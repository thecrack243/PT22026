"""Tests for the investigation-case store (SQLite, no Streamlit involved)."""
from __future__ import annotations

import time

import pytest

from src.cases import (ACTIVE, PRIORITIES, STATUSES, CaseStore, case_id, parse_case_id,
                       priority_for)

FRAUD_EXPLANATION = "The sender was emptied into an account that held no money"


@pytest.fixture()
def store(tmp_path):
    db = CaseStore(tmp_path / "cases.db")
    yield db
    db.close()


def make(store, txn_id=101, risk=97.4, decision="review", **kwargs):
    return store.create(txn_id, risk, decision, FRAUD_EXPLANATION, **kwargs)


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #
def test_case_ids_are_readable_and_parse_back():
    assert case_id(7) == "C-0007"
    assert case_id(1234) == "C-1234"
    for value in ("C-0007", "c-7", "7", 7, " c7 "):
        assert parse_case_id(value) == 7
    with pytest.raises(ValueError):
        parse_case_id("not-a-case")


def test_default_priority_follows_the_score():
    assert priority_for(99.9, "block") == "urgent"
    assert priority_for(97.0) == "urgent"
    assert priority_for(93.0) == "high"
    assert priority_for(88.0) == "medium"
    assert priority_for(70.0) == "low"
    assert priority_for(70.0, "block") == "urgent"        # a blocked transaction always matters


# --------------------------------------------------------------------------- #
# Creating and persisting
# --------------------------------------------------------------------------- #
def test_a_case_keeps_every_field_it_was_created_with(store):
    case = make(store, txn_id=42, risk=93.25, decision="review", priority="high",
                amount=250_000.0, tx_type="TRANSFER", source="live")

    assert case.case_id == "C-0001"
    assert (case.txn_id, case.risk, case.decision) == (42, 93.25, "review")
    assert case.explanation == FRAUD_EXPLANATION
    assert case.status == "open"
    assert case.priority == "high"
    assert (case.amount, case.tx_type, case.source) == (250_000.0, "TRANSFER", "live")
    assert case.notes == ""
    assert case.created_at == case.updated_at
    assert case.is_active and case.status_label == "Open"


def test_a_case_survives_reopening_the_database(tmp_path):
    path = tmp_path / "cases.db"
    first = CaseStore(path)
    created = make(first, txn_id=77)
    first.close()

    reopened = CaseStore(path)                  # what a restart does
    found = reopened.get(created.case_id)
    assert found == created
    assert reopened.counts()["open"] == 1
    reopened.close()


def test_the_database_and_its_folder_are_created_on_demand(tmp_path):
    path = tmp_path / "nested" / "deeper" / "cases.db"
    CaseStore(path)
    assert path.exists()


def test_cases_start_open_and_can_be_created_with_another_status(store):
    assert make(store).status == "open"
    confirmed = make(store, txn_id=102, status="confirmed_fraud")
    assert confirmed.status == "confirmed_fraud"
    assert not confirmed.is_active


def test_unknown_status_or_priority_is_rejected(store):
    with pytest.raises(ValueError, match="Unknown status"):
        make(store, status="pending")
    with pytest.raises(ValueError, match="Unknown priority"):
        make(store, priority="critical")


# --------------------------------------------------------------------------- #
# Reading and filtering
# --------------------------------------------------------------------------- #
def test_lookup_by_case_id_and_transaction_id(store):
    case = make(store, txn_id=555)
    assert store.get(case.case_id) == case
    assert store.get(case.number) == case
    assert store.get("C-9999") is None
    assert [c.number for c in store.for_transaction(555)] == [case.number]


def test_open_for_finds_only_the_case_still_being_worked_on(store):
    case = make(store, txn_id=9)
    assert store.open_for(9) == case

    store.set_status(case.case_id, "closed")
    assert store.open_for(9) is None                    # a closed case does not block a new one

    second = make(store, txn_id=9)
    assert store.open_for(9) == second
    assert len(store.for_transaction(9)) == 2


def test_listing_filters_and_orders_by_last_update(store):
    a = make(store, txn_id=1, priority="urgent")
    b = make(store, txn_id=2, priority="low")
    store.update(a.case_id, status="investigating")     # a becomes the most recently updated

    assert [c.number for c in store.list()][0] == a.number
    assert [c.number for c in store.list(status="open")] == [b.number]
    assert [c.number for c in store.list(status="investigating")] == [a.number]
    assert [c.number for c in store.list(priority="urgent")] == [a.number]
    assert [c.number for c in store.list() if c.number == b.number]
    assert store.list(limit=1)[0].number == a.number


def test_search_matches_both_ids(store):
    a = make(store, txn_id=1234)
    b = make(store, txn_id=99)
    assert [c.number for c in store.list(search="C-0001")] == [a.number]
    assert [c.number for c in store.list(search="1234")] == [a.number]
    assert [c.number for c in store.list(search=str(b.number))] == [b.number]
    assert store.list(search="0000421") == []


def test_counts_cover_every_status(store):
    make(store, txn_id=1)
    make(store, txn_id=2, status="investigating")
    make(store, txn_id=3, status="confirmed_fraud")
    counts = store.counts()
    assert set(counts) == set(STATUSES) | {"all", "active"}
    assert counts["open"] == 1 and counts["investigating"] == 1
    assert counts["all"] == 3 and counts["active"] == 2


# --------------------------------------------------------------------------- #
# Updating
# --------------------------------------------------------------------------- #
def test_updating_bumps_updated_at_but_not_created_at(store):
    case = make(store)
    time.sleep(0.01)
    moved = store.update(case.case_id, status="investigating", notes="Asked the sender for documents.")

    assert moved.status == "investigating"
    assert moved.notes.startswith("Asked the sender")
    assert moved.created_at == case.created_at
    assert moved.updated_at > case.updated_at
    assert store.get(case.case_id) == moved


def test_updating_with_the_same_values_does_not_touch_the_timestamp(store):
    case = make(store, notes="nothing new")
    again = store.update(case.case_id, status=case.status, notes=case.notes, priority=case.priority)
    assert again.updated_at == case.updated_at


def test_updating_a_missing_case_fails(store):
    with pytest.raises(KeyError):
        store.update("C-0404", status="closed")


def test_case_can_be_moved_through_every_status(store):
    case = make(store)
    for status in STATUSES:
        case = store.set_status(case.case_id, status)
        assert case.status == status
        assert (case.status in ACTIVE) == case.is_active


def test_delete_removes_the_case(store):
    case = make(store)
    assert store.delete(case.case_id) is True
    assert store.get(case.case_id) is None
    assert store.delete(case.case_id) is False


def test_in_memory_store_is_supported():
    store = CaseStore(":memory:")
    case = make(store)
    assert store.get(case.case_id) == case
    store.close()
