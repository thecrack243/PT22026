"""Tests for the audit log: every decision, with the policy that produced it."""
from __future__ import annotations

import pandas as pd
import pytest

from src.audit import AuditLog, policy_snapshot, snapshot_hash, threshold_text
from src.cases import CaseStore
from src.decision import Policy
from src.live import LiveStream
from tests.helpers import make_engine

POLICY = Policy(block_threshold=96.0, review_min_score=90.0, daily_capacity=240)
TIGHTER = Policy(block_threshold=99.0, review_min_score=95.0, daily_capacity=120)


@pytest.fixture()
def log(tmp_path):
    db = AuditLog(tmp_path / "audit.db")
    yield db
    db.close()


@pytest.fixture(scope="module")
def engine():
    return make_engine()[0]


def batch(*specs) -> pd.DataFrame:
    """A decided batch shaped like what the live stream hands over."""
    rows = [{"id": txn, "minute": minute, "amount": amount, "type": tx_type, "risk": risk,
             "decision": decision, "reason": reason}
            for txn, minute, amount, tx_type, risk, decision, reason in specs]
    return pd.DataFrame(rows)


def three_rows() -> pd.DataFrame:
    return batch((1600, 500, 215_000.0, "CASH_OUT", 99.7, "block", "sender emptied into a new account"),
                 (1601, 500, 48_000.0, "TRANSFER", 91.2, "review", "amount far above the sender's habit"),
                 (1602, 501, 1_200.0, "CASH_OUT", 12.4, "approve", ""))


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def test_every_transaction_gets_a_row_with_the_policy_that_decided_it(log):
    made = log.record(three_rows(), POLICY, model_tag="v3", session="s1")

    assert made == 3
    rows = log.list()
    assert {r.txn_id for r in rows} == {1600, 1601, 1602}
    blocked = next(r for r in rows if r.txn_id == 1600)
    assert blocked.decision == "block" and blocked.score == 99.7
    assert blocked.amount == 215_000.0 and blocked.tx_type == "CASH_OUT"
    assert blocked.model_tag == "v3" and blocked.session == "s1"
    assert "sender emptied" in blocked.reason
    assert blocked.decision_label == "Blocked"
    assert blocked.policy_hash == snapshot_hash(policy_snapshot(POLICY))


def test_recording_the_same_batch_twice_adds_nothing(log):
    log.record(three_rows(), POLICY, session="s1")
    again = log.record(three_rows(), POLICY, session="s1")

    assert again == 0
    assert len(log.list()) == 3                      # idempotent, not doubled


def test_a_policy_change_creates_a_new_version_and_history_keeps_the_old(log):
    log.record(three_rows(), POLICY, session="s1")
    log.record(batch((1603, 502, 9_000.0, "TRANSFER", 99.4, "block", "x")), TIGHTER, session="s1")

    versions = log.policies()
    assert len(versions) == 2
    by_hash = {v["hash"]: v for v in versions}
    old, new = by_hash[snapshot_hash(policy_snapshot(POLICY))], by_hash[snapshot_hash(policy_snapshot(TIGHTER))]
    assert old["decisions"] == 3 and new["decisions"] == 1
    assert log.policy(old["hash"])["block_threshold"] == 96.0
    assert log.policy(new["hash"])["block_threshold"] == 99.0


def test_thresholds_that_never_fire_are_stored_as_null(log):
    never = Policy(block_threshold=float("inf"), review_min_score=float("-inf"))
    snapshot = policy_snapshot(never)
    assert snapshot["block_threshold"] is None and snapshot["review_min_score"] is None
    assert threshold_text(None) == "none" and threshold_text(90.5) == "90.5"

    log.record(three_rows(), never)
    assert log.policy(log.list()[0].policy_hash) == snapshot


def test_a_failed_write_does_not_lose_the_earlier_rows(log):
    log.record(three_rows(), POLICY)
    before = len(log.list())
    assert log.record(pd.DataFrame(), POLICY) == 0     # nothing to write
    assert len(log.list()) == before


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def test_entry_finds_the_decision_of_one_transaction(log):
    log.record(three_rows(), POLICY, session="s1")

    record = log.entry(1601)
    assert record is not None and record.decision == "review"
    assert "above the sender's habit" in record.reason
    assert log.entry(9999) is None                     # never decided


def test_filters_by_decision_policy_and_search(log):
    log.record(three_rows(), POLICY, session="s1")
    log.record(batch((1603, 502, 9_000.0, "TRANSFER", 99.4, "block", "x")), TIGHTER, session="s1")

    assert {r.txn_id for r in log.list(decision="block")} == {1600, 1603}
    assert [r.txn_id for r in log.list(search="1602")] == [1602]
    assert [r.txn_id for r in log.list(search="sender emptied")] == [1600]
    assert len(log.list(policy_hash=snapshot_hash(policy_snapshot(TIGHTER)))) == 1
    assert len(log.list(decision="review", search="1600")) == 0


def test_counts_add_up(log):
    log.record(three_rows(), POLICY)
    log.record(batch((1603, 502, 900.0, "TRANSFER", 99.9, "block", "x")), TIGHTER)

    counts = log.counts()
    assert counts == {"all": 4, "block": 2, "review": 1, "approve": 1, "policies": 2}


def test_the_log_survives_a_restart(tmp_path):
    path = tmp_path / "audit.db"
    first = AuditLog(path)
    first.record(three_rows(), POLICY, model_tag="v3", session="s1")
    first.close()

    reopened = AuditLog(path)
    assert len(reopened.list()) == 3
    assert reopened.entry(1600).model_tag == "v3"
    assert reopened.policies()[0]["snapshot"]["block_threshold"] == 96.0
    reopened.close()


def test_two_sessions_can_use_the_same_transaction_numbers(log):
    # Every new simulation restarts the ids at 1: the pair (session, txn) is the identity.
    log.record(batch((1, 500, 10.0, "TRANSFER", 99.0, "block", "first run")), POLICY, session="aaa")
    log.record(batch((1, 500, 20.0, "TRANSFER", 20.0, "approve", "second run")), POLICY, session="bbb")

    assert len(log.list()) == 2
    assert log.entry(1, session="aaa").reason == "first run"
    assert log.entry(1, session="bbb").decision == "approve"
    assert log.entry(1).session == "bbb"               # latest when no session is given


# --------------------------------------------------------------------------- #
# The whole chain
# --------------------------------------------------------------------------- #
def test_the_chain_links_decision_policy_and_case_history(tmp_path):
    log = AuditLog(tmp_path / "audit.db")
    store = CaseStore(tmp_path / "cases.db")
    log.record(three_rows(), POLICY, session="s1")
    case = store.create(1600, 99.7, "block", "sender emptied into a new account")
    store.update(case.case_id, status="confirmed_fraud", notes="customer confirmed", closed=True)

    chain = log.chain(1600, cases=store, session="s1")
    assert chain["decision"].txn_id == 1600
    assert chain["policy"]["block_threshold"] == 96.0
    assert [item["case"].case_id for item in chain["cases"]] == ["C-0001"]
    kinds = [e.kind for e in chain["cases"][0]["events"]]
    assert kinds == ["closed", "notes", "status", "created"]     # newest first

    assert log.chain(1601, cases=store)["cases"] == []           # reviewed, never cased
    log.close()
    store.close()


# --------------------------------------------------------------------------- #
# The live stream writes as it goes
# --------------------------------------------------------------------------- #
def test_the_live_stream_audits_everything_it_decides(tmp_path, engine):
    log = AuditLog(tmp_path / "audit.db")
    stream = LiveStream(engine, seed=3, now=0.0, audit=log, model_tag="v3-test")
    stream.advance(POLICY, now=30.0)                    # 60 simulated minutes of traffic

    rows = log.list(limit=5000)
    assert len(rows) == len(stream.rows) == stream.processed
    assert {r.txn_id for r in rows} == set(stream.rows["id"])
    assert {r.model_tag for r in rows} == {"v3-test"}
    assert {r.session for r in rows} == {stream.session}
    assert stream.audit_error == ""
    # the stored decision is the decision the stream made
    decided = dict(zip(stream.rows["id"], stream.rows["decision"]))
    assert all(r.decision == decided[r.txn_id] for r in rows)
    log.close()


def test_a_changed_policy_is_audited_as_a_new_version(tmp_path, engine):
    log = AuditLog(tmp_path / "audit.db")
    stream = LiveStream(engine, seed=5, now=0.0, audit=log, model_tag="v3-test")
    stream.advance(POLICY, now=20.0)
    stream.advance(TIGHTER, now=40.0)

    versions = {v["hash"] for v in log.policies()}
    assert versions == {snapshot_hash(policy_snapshot(POLICY)), snapshot_hash(policy_snapshot(TIGHTER))}
    newest = log.list(limit=1)[0]                       # the last decision was made under TIGHTER
    assert newest.policy_hash == snapshot_hash(policy_snapshot(TIGHTER))
    log.close()
