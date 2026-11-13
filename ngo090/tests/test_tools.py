"""Tests for the tool layer (src/tools.py).

The point of the layer is that it runs without Streamlit and always answers the
same way, so these tests import it directly, never start an app, and check the
same call twice.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.artifacts import discover_runs
from src.cases import CaseStore
from src.decision import APPROVE, BLOCK, REVIEW, Policy
from src.predict import predict_one
from src.tools import Toolkit, ToolError, as_transaction, open_toolkit

from tests.helpers import make_engine

ROOT = Path(__file__).resolve().parent.parent
ENGINE, _, _ = make_engine()

FRAUD_TX = {"id": 1600, "type": "TRANSFER", "amount": 250_000, "oldbalanceOrg": 250_000,
            "oldbalanceDest": 0, "hour": 3}
NORMAL_TX = {"id": 1601, "type": "TRANSFER", "amount": 3_000, "oldbalanceOrg": 60_000,
             "oldbalanceDest": 40_000, "hour": 14}
POLICY = Policy(block_threshold=99.0, review_min_score=90.0, daily_capacity=150)


@pytest.fixture()
def tools(tmp_path):
    store = CaseStore(tmp_path / "cases.db")
    yield Toolkit(ENGINE, policy=POLICY, cases=store, model_tag="test")
    store.close()


# --------------------------------------------------------------------------- #
# The layer is independent of the UI
# --------------------------------------------------------------------------- #
def test_the_tool_layer_imports_without_streamlit():
    """A fresh process that imports src.tools must not drag Streamlit in."""
    code = "import sys, src.tools; print('streamlit' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


# --------------------------------------------------------------------------- #
# Scoring and assessment
# --------------------------------------------------------------------------- #
def test_scoring_is_deterministic(tools):
    first, second = tools.score_transaction(FRAUD_TX), tools.score_transaction(FRAUD_TX)
    assert first.to_dict() == second.to_dict()
    assert 0.0 <= first.risk <= 100.0
    assert first.applicable


def test_a_type_without_fraud_examples_is_not_scored(tools):
    score = tools.score_transaction({"type": "PAYMENT", "amount": 120, "oldbalanceOrg": 5400,
                                     "oldbalanceDest": 0, "hour": 11})
    assert not score.applicable
    assert score.risk == 0.0 and score.signal == ""
    assert score.notes and "no fraud examples" in score.notes[0]
    assert tools.assess_risk({"type": "PAYMENT", "amount": 120, "oldbalanceOrg": 5400,
                              "oldbalanceDest": 0}).rule.startswith("not scored")


def test_the_score_is_the_same_one_the_dashboard_uses(tools):
    """The tool layer must not become a second scoring implementation."""
    tool_risk = tools.score_transaction(FRAUD_TX).risk
    dashboard = predict_one(ENGINE, FRAUD_TX, POLICY)
    assert tool_risk == pytest.approx(dashboard.risk)


def test_assess_risk_names_the_rule_that_fired(tools):
    blocked = tools.assess_risk(FRAUD_TX)
    assert blocked.decision == BLOCK
    assert "auto-block threshold" in blocked.rule
    assert blocked.policy["block_threshold"] == 99.0

    reviewed = tools.assess_risk(FRAUD_TX, Policy(block_threshold=float("inf"),
                                                  review_min_score=90.0, daily_capacity=10))
    assert reviewed.decision == REVIEW
    assert "review threshold" in reviewed.rule

    no_capacity = tools.assess_risk(FRAUD_TX, Policy(block_threshold=float("inf"),
                                                     review_min_score=90.0, daily_capacity=0))
    assert no_capacity.decision == APPROVE
    assert "no review capacity" in no_capacity.rule


def test_the_assessment_carries_a_policy_version(tools):
    a = tools.assess_risk(FRAUD_TX, POLICY)
    b = tools.assess_risk(FRAUD_TX, Policy(block_threshold=95.0, review_min_score=90.0, daily_capacity=150))
    assert a.policy_hash != b.policy_hash
    assert a.to_dict()["policy"]["capacity_mode"] == "rolling"


def test_explanations_list_what_raised_the_risk(tools):
    explanation = tools.explain_transaction(FRAUD_TX)
    assert explanation.risk == pytest.approx(tools.score_transaction(FRAUD_TX).risk)
    assert explanation.decision == BLOCK
    assert explanation.reasons_up, "a drained account should have reasons"
    assert all(r.contribution > 0 for r in explanation.reasons_up)
    listed = explanation.to_dict()["reasons"]["raised"]
    assert listed and {"feature", "text", "contribution"} <= set(listed[0])


def test_transactions_are_accepted_in_every_shape():
    for value in (FRAUD_TX, pd.Series(FRAUD_TX), pd.DataFrame([FRAUD_TX]).itertuples(index=False).__next__()):
        tx = as_transaction(value)
        assert (tx.type, tx.amount, tx.hour, tx.id) == ("TRANSFER", 250_000.0, 3, 1600)


def test_a_bad_transaction_says_what_is_wrong(tools):
    with pytest.raises(ToolError) as exc:
        tools.score_transaction({"type": "TRANSFER", "amount": -5, "oldbalanceOrg": 1, "oldbalanceDest": 0})
    assert "amount" in str(exc.value)


# --------------------------------------------------------------------------- #
# Related transactions
# --------------------------------------------------------------------------- #
CORPUS = pd.DataFrame([
    {"id": 1, "step": 4, "type": "TRANSFER", "amount": 250_000, "oldbalanceOrg": 250_000, "oldbalanceDest": 0},
    {"id": 2, "step": 4, "type": "TRANSFER", "amount": 251_000, "oldbalanceOrg": 251_000, "oldbalanceDest": 0},
    {"id": 3, "step": 90, "type": "CASH_OUT", "amount": 900, "oldbalanceOrg": 9000, "oldbalanceDest": 5000},
    {"id": 1600, "step": 4, "type": "TRANSFER", "amount": 250_000, "oldbalanceOrg": 250_000, "oldbalanceDest": 0},
])


def test_related_transactions_rank_shared_evidence():
    tools = Toolkit(ENGINE, policy=POLICY, corpus=CORPUS, model_tag="test")
    result = tools.find_related_transactions(FRAUD_TX, limit=5)
    ids = [item.transaction["id"] for item in result.items]
    assert set(ids) == {1, 2}                       # the CASH_OUT has nothing in common
    assert ids[0] in (1, 2)
    assert "same type (TRANSFER)" in result.items[0].relation
    assert "both send into an empty account" in result.items[0].relation
    assert result.items[0].risk is not None and result.items[0].decision in (BLOCK, REVIEW, APPROVE)


def test_related_transactions_never_return_the_transaction_itself():
    tools = Toolkit(ENGINE, corpus=CORPUS)
    result = tools.find_related_transactions(FRAUD_TX, limit=10)
    assert 1600 not in [item.transaction["id"] for item in result.items]


def test_related_transactions_are_deterministic_and_need_a_corpus():
    tools = Toolkit(ENGINE, corpus=CORPUS)
    assert (tools.find_related_transactions(FRAUD_TX).to_dict()
            == tools.find_related_transactions(FRAUD_TX).to_dict())
    with pytest.raises(ToolError) as exc:
        Toolkit(ENGINE).find_related_transactions(FRAUD_TX)
    assert "corpus" in str(exc.value)


def test_an_empty_corpus_is_an_empty_answer_not_an_error():
    result = Toolkit(ENGINE, corpus=CORPUS.iloc[0:0]).find_related_transactions(FRAUD_TX)
    assert result.items == () and result.examined == 0


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
def test_create_case_is_idempotent(tools):
    first = tools.create_case(FRAUD_TX, source="test")
    assert first.created and first.case.case_id == "C-0001"
    assert first.case.risk == pytest.approx(tools.score_transaction(FRAUD_TX).risk)
    assert first.case.explanation and first.case.priority == "urgent"
    again = tools.create_case(FRAUD_TX)
    assert not again.created and again.case.case_id == first.case.case_id
    assert tools.cases.counts()["all"] == 1
    assert [e.kind for e in again.events] == ["created"]


def test_create_case_needs_an_id(tools):
    with pytest.raises(ToolError) as exc:
        tools.create_case(NORMAL_TX | {"id": None})
    assert "transaction id" in str(exc.value)


def test_update_case_moves_the_case_and_keeps_its_history(tools):
    case = tools.create_case(FRAUD_TX, source="test").case
    tools.update_case(case.case_id, status="confirmed_fraud", note="Called the customer.")
    updated = tools.update_case(case.case_id, state="closed")
    assert updated.case.closed and updated.case.status == "confirmed_fraud"
    assert updated.case.notes == "Called the customer."
    kinds = [e.kind for e in updated.events]
    assert kinds[:3] == ["closed", "notes", "status"]

    reopened = tools.update_case(case.case_id, state="open")
    assert not reopened.case.closed and reopened.case.notes == "Called the customer."
    assert reopened.events[0].kind == "reopened"


def test_closing_a_case_needs_a_resolution(tools):
    case = tools.create_case(FRAUD_TX, source="test").case
    with pytest.raises(ToolError) as exc:
        tools.update_case(case.case_id, state="closed")
    assert "resolution" in str(exc.value)


def test_update_case_replaces_notes_when_asked(tools):
    case = tools.create_case(FRAUD_TX, source="test").case
    tools.update_case(case.case_id, note="first")
    tools.update_case(case.case_id, notes="rewritten")
    assert tools.get_case(case.case_id).case.notes == "rewritten"


def test_get_case_returns_none_when_it_does_not_exist(tools):
    assert tools.get_case("C-9999") is None
    with pytest.raises(ToolError):
        tools.update_case("C-9999", note="hello")


def test_a_toolkit_without_stores_says_so():
    with pytest.raises(ToolError) as exc:
        Toolkit(ENGINE).get_case("C-0001")
    assert "case store" in str(exc.value)


# --------------------------------------------------------------------------- #
# Agent-facing contract
# --------------------------------------------------------------------------- #
def test_every_result_is_json_safe(tools):
    case = tools.create_case(FRAUD_TX, source="test").case
    results = [
        tools.score_transaction(FRAUD_TX).to_dict(),
        tools.assess_risk(FRAUD_TX).to_dict(),
        tools.explain_transaction(FRAUD_TX).to_dict(),
        Toolkit(ENGINE, corpus=CORPUS).find_related_transactions(FRAUD_TX).to_dict(),
        tools.get_case(case.case_id).to_dict(),
        tools.update_case(case.case_id, status="false_positive", note="ok").to_dict(),
    ]
    for result in results:
        assert json.loads(json.dumps(result)) == result


@pytest.mark.slow
@pytest.mark.skipif(not discover_runs(ROOT), reason="no trained risk engine")
def test_open_toolkit_builds_a_working_system_from_the_project(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUD_CASES_DB", str(tmp_path / "cases.db"))
    monkeypatch.setenv("FRAUD_AUDIT_DB", str(tmp_path / "audit.db"))
    tools = open_toolkit(ROOT)
    score = tools.score_transaction(FRAUD_TX)
    assert score.model_tag and score.applicable and 0.0 <= score.risk <= 100.0
    assert tools.cases is not None and tools.audit is not None
