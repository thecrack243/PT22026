"""Tests for the investigation assistant (src/agent.py).

Two properties matter more than any single rule, and most of these tests are
about them:

    * the assistant never changes a case -- investigating twice leaves the case
      byte for byte the same, and only `file()` writes anything at all;
    * the recommendation is a rule, not a language model -- the same case gives
      the same answer, and a writer that invents numbers is discarded.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.agent import (CLUSTER_MIN, STEP_LABELS, InvestigationAgent, TemplateWriter,
                       approve, reject)
from src.cases import CaseStore
from src.decision import Policy
from src.proposals import ProposalStore
from src.tools import Toolkit, ToolError

from tests.helpers import make_engine

ROOT = Path(__file__).resolve().parent.parent
ENGINE, _, _ = make_engine()

FRAUD = {"id": 1600, "type": "TRANSFER", "amount": 250_000, "oldbalanceOrg": 250_000,
         "oldbalanceDest": 0, "hour": 3}
MID = {"id": 1601, "type": "CASH_OUT", "amount": 5_000, "oldbalanceOrg": 9_000,
       "oldbalanceDest": 4_000, "hour": 10}
PAYMENT = {"id": 1602, "type": "PAYMENT", "amount": 800, "oldbalanceOrg": 9_000,
           "oldbalanceDest": 5_000, "hour": 9}
BIG_PAYMENT = {"id": 1603, "type": "PAYMENT", "amount": 250_000, "oldbalanceOrg": 300_000,
               "oldbalanceDest": 5_000, "hour": 10}

# Three look-alikes that drain the sender: the cluster the rules key on.
CLUSTER = pd.DataFrame([
    {"id": 800, "step": 11, "type": "CASH_OUT", "amount": 5_000,
     "oldbalanceOrg": 5_000, "oldbalanceDest": 0},
    {"id": 801, "step": 11, "type": "CASH_OUT", "amount": 5_020,
     "oldbalanceOrg": 5_020, "oldbalanceDest": 0},
    {"id": 802, "step": 12, "type": "CASH_OUT", "amount": 4_990,
     "oldbalanceOrg": 4_990, "oldbalanceDest": 0},
])

DEFAULT = Policy(block_threshold=99.0, review_min_score=90.0, daily_capacity=150)
NEVER_BLOCK = Policy(block_threshold=float("inf"), review_min_score=90.0, daily_capacity=150)
EAGER_REVIEW = Policy(block_threshold=float("inf"), review_min_score=80.0, daily_capacity=150)


def build(tmp_path, policy=DEFAULT, tx=FRAUD, priority=None, cases=None, queue=None):
    store = cases or CaseStore(tmp_path / "cases.db")
    tools = Toolkit(ENGINE, policy=policy, cases=store, model_tag="test")
    result = tools.create_case(tx, source="test", priority=priority)
    return tools, store, result.case, tx


def agent_for(tools, tx, queue=None, corpus=None, lookup=True, writer=None):
    """An agent whose live lookup returns the row the case was actually built from."""
    rows = (lambda txn_id: {**tx, "id": txn_id}) if lookup else None
    return InvestigationAgent(tools, proposals=queue, corpus=corpus, lookup=rows, writer=writer)


# --------------------------------------------------------------------------- #
# The seven steps
# --------------------------------------------------------------------------- #
def test_trace_has_the_seven_steps_in_order(tmp_path):
    tools, _, case, tx = build(tmp_path)
    investigation = agent_for(tools, tx, corpus=CLUSTER).investigate(case.number)
    assert [s.key for s in investigation.steps] == [key for key, _ in STEP_LABELS]
    assert all(s.label for s in investigation.steps)
    # every step is a real tool call: it found something or it says why not
    assert all(s.detail for s in investigation.steps)


def test_unknown_case_is_refused(tmp_path):
    tools, _, _, _ = build(tmp_path)
    with pytest.raises(ToolError):
        InvestigationAgent(tools).investigate("C-9999")


def test_evidence_names_the_tool_it_came_from(tmp_path):
    tools, _, case, tx = build(tmp_path, tx=MID)
    investigation = agent_for(tools, tx, corpus=CLUSTER).investigate(case.number)
    sources = {e.source for e in investigation.evidence}
    assert {"get_case", "assess_risk", "find_related_transactions"} <= sources
    assert all(e.text for e in investigation.evidence)


# --------------------------------------------------------------------------- #
# The assistant cannot change anything
# --------------------------------------------------------------------------- #
def test_investigating_twice_leaves_the_case_untouched(tmp_path):
    tools, store, case, tx = build(tmp_path)
    before_case = store.get(case.number)
    before_events = [e.detail for e in store.events(case.number)]

    for _ in range(2):
        agent_for(tools, tx, corpus=CLUSTER).investigate(case.number)

    after = store.get(case.number)
    assert after == before_case
    assert [e.detail for e in store.events(case.number)] == before_events


def test_filing_writes_a_proposal_and_not_a_case_change(tmp_path):
    queue = ProposalStore(tmp_path / "proposals.db")
    # A reviewed case that is not yet urgent: the advice is real, so it files.
    tools, store, case, tx = build(tmp_path, policy=NEVER_BLOCK, priority="low", queue=queue)
    before = store.get(case.number)

    agent = agent_for(tools, tx, queue=queue, corpus=CLUSTER)
    filed = agent.file(agent.investigate(case.number))

    assert filed.pending
    assert filed.case_number == case.number
    assert filed.evidence["evidence"] and filed.evidence["steps"]
    assert store.get(case.number) == before
    queue.close()


def test_filing_without_a_queue_is_refused(tmp_path):
    tools, _, case, tx = build(tmp_path)
    agent = agent_for(tools, tx, corpus=CLUSTER)
    investigation = agent.investigate(case.number)
    with pytest.raises(ToolError):
        agent.file(investigation)


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #
def rule_of(tmp_path, policy=DEFAULT, tx=FRAUD, priority=None, corpus=None):
    tools, _, case, tx = build(tmp_path, policy=policy, tx=tx, priority=priority)
    return agent_for(tools, tx, corpus=corpus).investigate(case.number)


def test_r1_blocks_escalate_a_non_urgent_case(tmp_path):
    investigation = rule_of(tmp_path, priority="low")
    assert investigation.policy_decision == "block"
    assert investigation.recommendation.kind == "escalate"
    assert investigation.recommendation.rule == "R1"
    assert investigation.recommendation.proposed_priority == "urgent"
    assert not investigation.recommendation.requires_override


def test_r2_a_blocked_urgent_case_is_left_alone(tmp_path):
    investigation = rule_of(tmp_path, priority="urgent")
    assert (investigation.recommendation.kind, investigation.recommendation.rule) == ("keep_open", "R2")
    assert not investigation.recommendation.changes_anything


def test_r3_very_high_review_is_escalated(tmp_path):
    investigation = rule_of(tmp_path, policy=NEVER_BLOCK, priority="low")
    assert investigation.policy_decision == "review"
    assert investigation.recommendation.rule == "R3"
    assert investigation.recommendation.proposed_priority == "urgent"


def test_r3_does_not_escalate_a_case_that_is_already_urgent(tmp_path):
    """The case in the screenshot: R3 said "escalate" to an already-urgent case.

    A recommendation to do what is already true is not advice, it is noise in
    the approval queue, so the same rule reports the queue position instead.
    """
    investigation = rule_of(tmp_path, policy=NEVER_BLOCK, priority="urgent")
    assert investigation.policy_decision == "review"
    assert (investigation.recommendation.kind, investigation.recommendation.rule) == ("keep_open", "R3")
    assert not investigation.recommendation.changes_anything
    assert "nothing to change" in investigation.recommendation.rationale


def test_r4_never_downgrades_an_urgent_case(tmp_path):
    investigation = rule_of(tmp_path, policy=EAGER_REVIEW, tx=MID, corpus=CLUSTER, priority="urgent")
    assert investigation.recommendation.rule == "R4"
    assert investigation.recommendation.kind == "keep_open"
    assert not investigation.recommendation.changes_anything


def test_a_proposal_that_changes_nothing_is_not_filed(tmp_path):
    queue = ProposalStore(tmp_path / "proposals.db")
    tools, store, case, tx = build(tmp_path, policy=NEVER_BLOCK, priority="urgent", queue=queue)
    agent = agent_for(tools, tx, queue=queue, corpus=CLUSTER)
    with pytest.raises(ToolError) as err:
        agent.file(agent.investigate(case.number))
    assert "nothing to approve" in str(err.value)
    assert queue.pending() == []
    assert store.get(case.number) == case
    queue.close()


def test_r4_a_cluster_escalates_a_review(tmp_path):
    investigation = rule_of(tmp_path, policy=EAGER_REVIEW, tx=MID, corpus=CLUSTER)
    assert investigation.policy_decision == "review"
    assert investigation.recommendation.rule == "R4"
    assert investigation.recommendation.proposed_priority == "high"
    assert not investigation.recommendation.requires_override


def test_r5_escalating_an_approved_transaction_is_an_override(tmp_path):
    investigation = rule_of(tmp_path, tx=MID, corpus=CLUSTER)
    assert investigation.policy_decision == "approve"
    assert investigation.recommendation.rule == "R5"
    assert investigation.recommendation.requires_override
    assert "overrides the policy" in investigation.recommendation.rationale


def test_r6_an_approved_lonely_case_is_a_false_positive(tmp_path):
    investigation = rule_of(tmp_path, tx=PAYMENT)
    assert investigation.recommendation.rule == "R6"
    assert investigation.recommendation.proposed_status == "false_positive"
    assert investigation.recommendation.proposed_state == "closed"
    assert not investigation.recommendation.requires_override


def test_r7_an_unscoreable_type_in_a_cluster_is_left_open(tmp_path):
    # The engine has no examples of PAYMENT, so it cannot score it; but the
    # neighbours are suspicious. Closing it would be a guess, so it stays open.
    investigation = rule_of(tmp_path, tx=BIG_PAYMENT, corpus=CLUSTER.assign(
        amount=250_000, oldbalanceOrg=250_000))
    assert not investigation.assessment.applicable
    assert investigation.recommendation.rule == "R7"
    assert investigation.recommendation.kind == "keep_open"


def test_cluster_threshold_is_the_documented_one():
    assert CLUSTER_MIN == 2


# --------------------------------------------------------------------------- #
# Honesty about what it could not check
# --------------------------------------------------------------------------- #
def test_missing_live_row_is_reported_not_invented(tmp_path):
    tools, _, case, tx = build(tmp_path)
    investigation = InvestigationAgent(tools, corpus=None, lookup=None).investigate(case.number)
    assert investigation.gaps and "#1600" in investigation.gaps[0]
    # the steps that would have been wrong are marked as skipped, not faked
    skipped = {s.key for s in investigation.steps if not s.ok}
    assert skipped == {"transaction_details", "related_activity", "risk_assessment"}
    # and the recorded score is what the recommendation uses
    assert investigation.risk == case.risk
    assert "as recorded on the case" in investigation.narrative


def test_no_corpus_is_a_gap_not_an_empty_result(tmp_path):
    tools, _, case, tx = build(tmp_path)
    investigation = agent_for(tools, tx, corpus=None).investigate(case.number)
    assert any("no corpus" in gap for gap in investigation.gaps)
    assert investigation.related is None


def test_policy_change_since_open_is_reported(tmp_path):
    tools, _, case, tx = build(tmp_path, policy=NEVER_BLOCK)
    # the case was opened under a policy that never blocks; now it blocks
    tools.policy = DEFAULT
    investigation = agent_for(tools, tx, corpus=None).investigate(case.number)
    assert any("The case was opened at" in gap for gap in investigation.gaps)


# --------------------------------------------------------------------------- #
# The writer is replaceable, the guard is not
# --------------------------------------------------------------------------- #
class LyingWriter:
    name = "llm"

    def write(self, brief, facts=None):
        return "The customer sent 987654 to an account in Lagos."


class FailingWriter:
    name = "llm"

    def write(self, brief, facts=None):
        raise RuntimeError("the model endpoint failed: connection reset")


class PlainWriter:
    name = "llm"

    def write(self, brief, facts=None):
        return "A transfer [1] of 250,000 was blocked by the policy."


def test_template_is_deterministic(tmp_path):
    tools, _, case, tx = build(tmp_path)
    first = agent_for(tools, tx, corpus=CLUSTER).investigate(case.number)
    second = agent_for(tools, tx, corpus=CLUSTER).investigate(case.number)
    assert first.narrative == second.narrative
    assert first.recommendation.to_dict() == second.recommendation.to_dict()
    assert first.writer == "template"


def test_a_writer_that_invents_numbers_is_discarded(tmp_path):
    tools, _, case, tx = build(tmp_path)
    investigation = agent_for(tools, tx, corpus=CLUSTER, writer=LyingWriter()).investigate(case.number)
    assert investigation.writer == "template"
    assert "987654" not in investigation.narrative
    assert any("introduced numbers" in gap for gap in investigation.gaps)


def test_a_failing_writer_falls_back_without_crashing(tmp_path):
    tools, _, case, tx = build(tmp_path)
    investigation = agent_for(tools, tx, corpus=CLUSTER, writer=FailingWriter()).investigate(case.number)
    assert investigation.writer == "template"
    assert any("was not used" in gap for gap in investigation.gaps)


def test_a_faithful_writer_is_kept(tmp_path):
    tools, _, case, tx = build(tmp_path)
    investigation = agent_for(tools, tx, corpus=CLUSTER, writer=PlainWriter()).investigate(case.number)
    assert investigation.writer == "llm"
    assert investigation.narrative.startswith("A transfer [1]")


def test_investigation_is_json_safe(tmp_path):
    tools, _, case, tx = build(tmp_path)
    payload = agent_for(tools, tx, corpus=CLUSTER).investigate(case.number).to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["case"] == case.case_id
    assert payload["recommendation"]["rule"]


# --------------------------------------------------------------------------- #
# Approval: the only path to a changed case
# --------------------------------------------------------------------------- #
def test_approval_applies_through_the_tool_layer(tmp_path):
    queue = ProposalStore(tmp_path / "proposals.db")
    tools, store, case, tx = build(tmp_path, priority="low", queue=queue)
    agent = agent_for(tools, tx, queue=queue, corpus=None)
    proposal = agent.file(agent.investigate(case.number))
    assert not proposal.requires_override            # R1 agrees with the policy

    result = approve(queue, tools, proposal, by="ana", note="same pattern as last week")
    assert result.case.priority == "urgent"
    assert "Assistant (rules:v1/template)" in result.case.notes
    assert "Approved by ana: same pattern as last week" in result.case.notes
    assert queue.get(proposal.number).status == "approved"
    # and the change is in the case's own history, like any other edit
    assert any("Priority" in e.detail for e in store.events(case.number))
    queue.close()


def test_an_override_needs_the_extra_tick(tmp_path):
    queue = ProposalStore(tmp_path / "proposals.db")
    tools, store, case, tx = build(tmp_path, tx=MID, queue=queue)
    agent = agent_for(tools, tx, queue=queue, corpus=CLUSTER)
    proposal = agent.file(agent.investigate(case.number))
    assert proposal.requires_override

    with pytest.raises(ToolError):
        approve(queue, tools, proposal)
    assert store.get(case.number).priority == case.priority     # nothing happened

    approve(queue, tools, proposal, override=True)
    assert store.get(case.number).priority == "high"
    queue.close()


def test_rejecting_leaves_the_case_alone(tmp_path):
    queue = ProposalStore(tmp_path / "proposals.db")
    tools, store, case, tx = build(tmp_path, tx=MID, queue=queue)
    before = store.get(case.number)
    agent = agent_for(tools, tx, queue=queue, corpus=CLUSTER)
    proposal = agent.file(agent.investigate(case.number))

    decided = reject(queue, proposal, by="ana", note="the customer confirmed it")
    assert decided.status == "rejected"
    assert store.get(case.number) == before
    queue.close()


def test_a_decided_proposal_cannot_be_decided_again(tmp_path):
    queue = ProposalStore(tmp_path / "proposals.db")
    tools, _, case, tx = build(tmp_path, priority="low", queue=queue)
    agent = agent_for(tools, tx, queue=queue, corpus=None)
    proposal = agent.file(agent.investigate(case.number))
    approve(queue, tools, proposal)
    with pytest.raises(ToolError):
        approve(queue, tools, proposal)
    queue.close()


def test_approving_a_proposal_that_does_not_exist_is_refused(tmp_path):
    queue = ProposalStore(tmp_path / "proposals.db")
    tools, _, case, tx = build(tmp_path)
    with pytest.raises(ToolError):
        approve(queue, tools, 99)
    queue.close()


# --------------------------------------------------------------------------- #
# Headless
# --------------------------------------------------------------------------- #
def test_importing_the_agent_does_not_load_streamlit():
    code = ("import sys; import src.agent; "
            "print('streamlit' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", out.stderr
