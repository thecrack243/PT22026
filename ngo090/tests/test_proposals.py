"""Tests for the proposal queue (src/proposals.py).

A proposal is the boundary between "the assistant thinks" and "the case
changes". These tests pin the boundary: only a pending proposal can be decided,
only once, and a case can never have two contradictory recommendations waiting.
"""
from __future__ import annotations

import pytest

from src.proposals import KINDS, ProposalStore


@pytest.fixture()
def queue(tmp_path):
    store = ProposalStore(tmp_path / "proposals.db")
    yield store
    store.close()


def make(queue, case="C-0001", kind="escalate", **kwargs):
    defaults = dict(rationale="because", risk=97.0, policy_decision="review",
                    proposed_priority="urgent", agent="rules:v1/template")
    defaults.update(kwargs)
    return queue.create(case, kind, **defaults)


def test_create_is_pending_and_keeps_its_evidence(queue):
    item = make(queue, evidence={"evidence": [{"kind": "risk", "text": "risk 97"}]})
    assert item.pending and item.status == "pending"
    assert item.number == 1
    assert item.case_id == "C-0001"
    assert item.evidence["evidence"][0]["text"] == "risk 97"
    assert item.changes == ["Priority -> Urgent"]


def test_unknown_kind_is_refused(queue):
    with pytest.raises(ValueError):
        queue.create("C-0001", "do_something_else")
    # `priority`, `reopen` and `resolution` exist because the chat records what
    # the analyst asked for the same way the rules record what they recommend.
    assert set(KINDS) == {"escalate", "close_fraud", "close_false_positive", "keep_open",
                          "priority", "reopen", "resolution"}


def test_a_second_proposal_supersedes_the_first(queue):
    first = make(queue, kind="escalate")
    second = make(queue, kind="close_false_positive", proposed_status="false_positive",
                  proposed_state="closed")
    assert queue.get(first.number).status == "superseded"
    assert queue.get(second.number).status == "pending"
    assert [p.number for p in queue.pending()] == [second.number]
    assert queue.pending_for("C-0001").number == second.number


def test_pending_is_oldest_first_and_list_is_newest_first(queue):
    a = make(queue, case="C-0001")
    b = make(queue, case="C-0002")
    assert [p.number for p in queue.pending()] == [a.number, b.number]
    assert [p.number for p in queue.list()] == [b.number, a.number]


def test_decide_records_who_and_why(queue):
    item = make(queue)
    decided = queue.decide(item.number, True, by="ana", note="checked with the customer")
    assert decided.status == "approved"
    assert decided.decided_by == "ana"
    assert decided.decision_note == "checked with the customer"
    assert decided.decided_at


def test_decide_twice_is_refused(queue):
    item = make(queue)
    queue.decide(item.number, True)
    with pytest.raises(ValueError):
        queue.decide(item.number, False)


def test_rejecting_keeps_the_row(queue):
    item = make(queue)
    queue.decide(item.number, False, by="ana")
    stored = queue.get(item.number)
    assert stored.status == "rejected" and not stored.pending
    assert queue.counts()["rejected"] == 1 and queue.counts()["pending"] == 0


def test_counts(queue):
    make(queue, case="C-0001")
    rejected = make(queue, case="C-0002")
    queue.decide(rejected.number, False)
    make(queue, case="C-0003")
    counts = queue.counts()
    assert counts["pending"] == 2 and counts["rejected"] == 1 and counts["all"] == 3


def test_survives_a_restart(tmp_path):
    path = tmp_path / "proposals.db"
    first = ProposalStore(path)
    item = make(first)
    first.close()

    second = ProposalStore(path)
    try:
        again = second.get(item.number)
        assert again is not None and again.pending
        assert again.rationale == "because"
        assert again.requires_override is False
    finally:
        second.close()


def test_requires_override_round_trips(queue):
    item = make(queue, requires_override=True, policy_decision="approve")
    assert item.requires_override is True
    assert queue.get(item.number).requires_override is True


def test_unknown_status_filter_is_refused(queue):
    with pytest.raises(ValueError):
        queue.list(status="waiting")
