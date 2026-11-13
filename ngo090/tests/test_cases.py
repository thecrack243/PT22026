"""Tests for the investigation-case store (SQLite, no Streamlit involved)."""
from __future__ import annotations

import time

import pytest

from src.cases import PRIORITIES, STATUSES, CaseStore, case_id, parse_case_id, priority_for

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
    assert case.status == "investigating"                  # the default resolution
    assert case.closed is False                            # and it starts open
    assert case.priority == "high"
    assert (case.amount, case.tx_type, case.source) == (250_000.0, "TRANSFER", "live")
    assert case.notes == ""
    assert case.created_at == case.updated_at
    assert case.is_open and case.status_label == "Investigating" and case.state_label == "Open"


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


def test_cases_start_investigating_and_can_be_created_with_a_resolution(store):
    assert make(store).status == "investigating"
    confirmed = make(store, txn_id=102, status="confirmed_fraud")
    assert confirmed.status == "confirmed_fraud"
    assert confirmed.is_open                             # decided, but not closed yet


def test_unknown_status_or_priority_is_rejected(store):
    with pytest.raises(ValueError, match="Unknown status"):
        make(store, status="pending")
    with pytest.raises(ValueError, match="Unknown priority"):
        make(store, priority="critical")


def test_a_legacy_database_is_migrated_on_startup(tmp_path):
    """A database from the first version (status open/closed, no closed column)."""
    import sqlite3

    path = tmp_path / "cases.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT, txn_id INTEGER NOT NULL, risk REAL NOT NULL,
            decision TEXT NOT NULL, explanation TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'open', notes TEXT NOT NULL DEFAULT '',
            priority TEXT NOT NULL DEFAULT 'medium', source TEXT NOT NULL DEFAULT 'live',
            amount REAL NOT NULL DEFAULT 0, tx_type TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
    """)
    conn.execute("INSERT INTO cases (txn_id, risk, decision, status, created_at, updated_at)"
                 " VALUES (1, 95.0, 'block', 'open', 't', 't')")           # old 'open'
    conn.execute("INSERT INTO cases (txn_id, risk, decision, status, created_at, updated_at)"
                 " VALUES (2, 90.0, 'review', 'investigating', 't', 't')") # same name, same meaning
    conn.execute("INSERT INTO cases (txn_id, risk, decision, status, created_at, updated_at)"
                 " VALUES (3, 88.0, 'approve', 'closed', 't', 't')")       # old 'closed'
    conn.commit()
    conn.close()

    store = CaseStore(path)                     # opening the store performs the migration
    rows = {c.txn_id: c for c in store.list()}
    assert rows[1].status == "investigating" and rows[1].closed is False
    assert rows[2].status == "investigating" and rows[2].closed is False
    assert rows[3].closed is True                          # the old Closed state became the flag
    assert store.counts()["open"] == 2 and store.counts()["closed"] == 1
    store.close()


# --------------------------------------------------------------------------- #
# Reading and filtering
# --------------------------------------------------------------------------- #
def test_lookup_by_case_id_and_transaction_id(store):
    case = make(store, txn_id=555)
    assert store.get(case.case_id) == case
    assert store.get(case.number) == case
    assert store.get("C-9999") is None
    assert [c.number for c in store.for_transaction(555)] == [case.number]


def test_open_for_finds_only_a_case_still_open(store):
    case = make(store, txn_id=9)
    assert store.open_for(9) == case

    store.update(case.case_id, closed=True)
    assert store.open_for(9) is None                    # a closed case does not block a new one

    second = make(store, txn_id=9)
    assert store.open_for(9) == second
    assert len(store.for_transaction(9)) == 2


def test_listing_filters_and_orders_by_last_update(store):
    a = make(store, txn_id=1, priority="urgent")
    b = make(store, txn_id=2, priority="low", status="confirmed_fraud")
    store.update(a.case_id, notes="a note")             # a becomes the most recently updated

    assert [c.number for c in store.list()][0] == a.number
    assert [c.number for c in store.list(status="confirmed_fraud")] == [b.number]
    assert [c.number for c in store.list(status="investigating")] == [a.number]
    assert [c.number for c in store.list(priority="urgent")] == [a.number]
    assert [c.number for c in store.list(state="open")] == [a.number, b.number]
    assert [c.number for c in store.list(state="closed")] == []
    assert [c.number for c in store.list() if c.number == b.number]
    assert store.list(limit=1)[0].number == a.number


def test_status_and_state_filters_combine(store):
    decided = make(store, txn_id=1, status="confirmed_fraud")
    store.update(decided.case_id, closed=True)
    working = make(store, txn_id=2, status="confirmed_fraud")

    found = store.list(status="confirmed_fraud", state="closed")
    assert [c.number for c in found] == [decided.number]
    assert [c.number for c in store.list(status="confirmed_fraud", state="open")] == [working.number]


def test_search_matches_both_ids(store):
    a = make(store, txn_id=1234)
    b = make(store, txn_id=99)
    assert [c.number for c in store.list(search="C-0001")] == [a.number]
    assert [c.number for c in store.list(search="1234")] == [a.number]
    assert [c.number for c in store.list(search=str(b.number))] == [b.number]
    assert store.list(search="0000421") == []


def test_counts_separate_the_resolution_from_the_state(store):
    make(store, txn_id=1)                                            # investigating, open
    make(store, txn_id=2, status="confirmed_fraud")                  # decided, open
    closed = make(store, txn_id=3, status="false_positive")
    store.update(closed.case_id, closed=True)

    counts = store.counts()
    assert set(counts) == set(STATUSES) | {"closed", "open", "all"}
    assert counts["investigating"] == 1 and counts["confirmed_fraud"] == 1
    assert counts["false_positive"] == 0                  # the open ones only
    assert counts["closed"] == 1
    assert counts["open"] == 2 and counts["all"] == 3


# --------------------------------------------------------------------------- #
# Updating
# --------------------------------------------------------------------------- #
def test_updating_bumps_updated_at_but_not_created_at(store):
    case = make(store)
    time.sleep(0.01)
    moved = store.update(case.case_id, status="confirmed_fraud", notes="Asked the sender for documents.")

    assert moved.status == "confirmed_fraud"
    assert moved.notes.startswith("Asked the sender")
    assert moved.created_at == case.created_at
    assert moved.updated_at > case.updated_at
    assert store.get(case.case_id) == moved


def test_saving_and_closing_in_one_update(store):
    case = make(store)
    closed = store.update(case.case_id, status="confirmed_fraud", notes="Mule flow", closed=True)
    assert closed.closed and closed.is_open is False and closed.state_label == "Closed"

    reopened = store.update(case.case_id, closed=False)   # the Reopen case button
    assert reopened.closed is False and reopened.status == "confirmed_fraud"


def test_updating_with_the_same_values_does_not_touch_the_timestamp(store):
    case = make(store, notes="nothing new")
    again = store.update(case.case_id, status=case.status, notes=case.notes,
                         priority=case.priority, closed=case.closed)
    assert again.updated_at == case.updated_at


def test_updating_a_missing_case_fails(store):
    with pytest.raises(KeyError):
        store.update("C-0404", status="confirmed_fraud")


def test_case_can_be_moved_through_every_status(store):
    case = make(store)
    for status in STATUSES:
        case = store.set_status(case.case_id, status)
        assert case.status == status


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


# --------------------------------------------------------------------------- #
# Automatic case creation
# --------------------------------------------------------------------------- #
from collections import namedtuple

Alert = namedtuple("Alert", "id risk decision reason amount type")


def alerts(*specs):
    """(txn, risk, decision) tuples turned into the row shape auto_open expects."""
    return [Alert(txn, risk, decision, "explanation", 500.0, "TRANSFER")
            for txn, risk, decision in specs]


def test_auto_open_creates_cases_for_alerts_at_or_above_the_threshold(store):
    from src.cases import auto_open

    made = auto_open(store, alerts((1, 99.0, "block"), (2, 99.5, "review"),
                                   (3, 98.9, "block")), min_risk=99.0)
    assert made == 2
    assert {c.txn_id for c in store.list()} == {1, 2}
    created = store.list()[0]
    assert created.source == "auto" and created.status == "investigating"


def test_auto_open_ignores_approved_transactions(store):
    from src.cases import auto_open

    made = auto_open(store, alerts((1, 99.9, "approve"), (2, 99.9, "block")), min_risk=99.0)
    assert made == 1
    assert [c.txn_id for c in store.list()] == [2]


def test_auto_open_never_duplicates_an_existing_case(store):
    from src.cases import auto_open

    existing = store.create(11, 99.5, "block", FRAUD_EXPLANATION, source="live")
    store.update(existing.case_id, closed=True, status="false_positive")   # closed cases count too
    made = auto_open(store, alerts((11, 99.9, "block"), (12, 99.9, "block")), min_risk=99.0)
    assert made == 1
    assert len(store.for_transaction(11)) == 1
    assert len(store.for_transaction(12)) == 1


def test_auto_open_stops_at_the_batch_size(store):
    from src.cases import auto_open

    rows = alerts(*[(i, 99.5, "block") for i in range(1, 11)])
    assert auto_open(store, rows, min_risk=99.0, limit=3) == 3
    assert store.counts()["all"] == 3


def test_auto_open_with_limit_zero_has_no_cap(store):
    from src.cases import auto_open

    rows = alerts(*[(i, 99.5, "block") for i in range(1, 26)])
    assert auto_open(store, rows, min_risk=99.0, limit=0) == 25


# --------------------------------------------------------------------------- #
# Investigation history (case_events)
# --------------------------------------------------------------------------- #
def test_opening_a_case_writes_the_first_history_row(store):
    case = make(store, txn_id=42, risk=97.4, decision="block", source="live")
    events = store.events(case.case_id)

    assert len(events) == 1
    assert events[0].kind == "created"
    assert "alert #42" in events[0].detail and "risk 97.4" in events[0].detail
    assert events[0].case_number == case.number and events[0].created_at == case.created_at


def test_auto_opened_cases_say_so_in_their_history(store):
    from src.cases import auto_open

    auto_open(store, alerts((5, 99.9, "block")), min_risk=99.0)
    events = store.events(store.list()[0].case_id)
    assert events[0].kind == "created" and events[0].detail.startswith("Auto-opened")


def test_every_edit_appends_without_overwriting(store):
    case = make(store)                                             # risk 97.4 -> priority starts urgent
    store.update(case.case_id, status="confirmed_fraud", priority="high", closed=True)
    store.update(case.case_id, notes="cleared with the customer")
    store.update(case.case_id, closed=False)                       # reopen

    events = store.events(case.case_id)
    kinds = [e.kind for e in events]                               # newest first
    assert kinds == ["reopened", "notes", "closed", "priority", "status", "created"]
    assert "Investigating -> Confirmed fraud" in events[-2].detail  # the status row
    assert events[0].detail == "Case reopened"


def test_history_survives_a_restart(tmp_path):
    path = tmp_path / "cases.db"
    first = CaseStore(path)
    case = make(first, txn_id=9)
    first.update(case.case_id, status="false_positive", closed=True)
    first.close()

    reopened = CaseStore(path)
    events = reopened.events(case.case_id)
    assert [e.kind for e in events] == ["closed", "status", "created"]
    reopened.close()


def test_deleting_a_case_removes_its_history(store):
    case = make(store)
    store.update(case.case_id, notes="temp")
    assert store.delete(case.case_id)
    assert store.events(case.case_id) == []


def test_a_new_case_has_history_but_an_empty_edit_writes_nothing(store):
    case = make(store)
    before = store.events(case.case_id)
    same = store.update(case.case_id, status=case.status, notes=case.notes,
                        priority=case.priority, closed=case.closed)
    assert same.updated_at == case.updated_at                      # no real change
    assert len(store.events(case.case_id)) == len(before)          # and no new rows


def test_cases_from_before_the_history_table_get_a_backfilled_row(tmp_path):
    # Simulate an old database: build one, drop the events table, reopen.
    import sqlite3

    path = tmp_path / "cases.db"
    old = CaseStore(path)
    case = make(old, txn_id=3)
    old.close()
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE case_events")
    conn.commit()
    conn.close()

    upgraded = CaseStore(path)                                     # what the app does on startup
    events = upgraded.events(case.case_id)
    assert len(events) == 1
    assert events[0].kind == "created" and "alert #3" in events[0].detail
    assert events[0].created_at == case.created_at                 # backdated to the open time
    upgraded.close()
