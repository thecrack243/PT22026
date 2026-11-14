"""The investigation assistant: one agent, one case, evidence first.

What this is
------------
An assistant that investigates a case the way a careful analyst would: it reads
the case, pulls the transaction, looks for related activity, asks the Risk
Manager what it thinks, and then writes up what it found and what it recommends.
It is deliberately *one* assistant on *one* case -- not a swarm of autonomous
agents. Every step it takes is a call to `src/tools.py`, so the assistant cannot
do anything a script could not do, and every step is recorded in a trace the
analyst can read.

What this is not
----------------
It is not the decision maker. Two rules keep that honest:

    * The assistant never writes to a case. `investigate()` only reads; the
      result is a *proposal* (`src/proposals.py`) that a person approves.
    * The decision stays with the Risk Manager. `assess_risk` applies the
      policy; the assistant's recommendation is computed by the explicit rules
      in `_recommend()` and is shown *next to* the policy's decision, not
      instead of it. When they disagree the proposal is marked
      `requires_override`, and approving it takes a separate, deliberate tick.

The language layer
------------------
The recommendation is arithmetic, not vibes: `_recommend()` is a fixed rule list
over the evidence. The prose comes from a writer, and the default writer is a
deterministic template, so the same case always produces the same text -- which
is what makes it testable. An optional LLM writer (`FRAUD_LLM_URL`) can rephrase
the same facts, with two guards: it is given only the evidence list and told to
use nothing else, and its output is rejected if it contains a number that is not
in the evidence. When either guard trips, the template is used instead and the
reason is recorded. The model never chooses the recommendation; it only rewords
it.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from src.cases import Case, CaseEvent, STATUS_LABELS, PRIORITY_LABELS
from src.decision import APPROVE, BLOCK, REVIEW
from src.proposals import Proposal, ProposalStore
from src.tools import (Assessment, CaseResult, Explanation, RelatedResult, Toolkit,
                       ToolError, as_transaction)

AGENT_TAG = "rules:v2"                    # bump when _recommend() changes behaviour
                                          # v2: priority rules never propose a change already true
LLM_URL = "FRAUD_LLM_URL"                 # OpenAI-compatible chat completions endpoint
LLM_KEY = "FRAUD_LLM_KEY"
LLM_MODEL = "FRAUD_LLM_MODEL"
LLM_TIMEOUT = 20.0

# A related transaction only counts as cluster evidence at this risk. The number
# is the same order as the auto-block threshold, so "cluster" means "several
# things the policy would itself have blocked".
CLUSTER_RISK = 90.0
CLUSTER_MIN = 2

STEP_LABELS = (
    ("receive_case", "Case received"),
    ("transaction_details", "Transaction details"),
    ("related_activity", "Related activity"),
    ("risk_assessment", "Risk Manager assessment"),
    ("evidence_summary", "Evidence summary"),
    ("recommendation", "Recommendation"),
    ("proposal", "Proposal"),
)


# --------------------------------------------------------------------------- #
# Pieces of an investigation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Step:
    """One thing the assistant did, with what it learned, in order."""
    key: str
    label: str
    detail: str
    ms: float = 0.0
    ok: bool = True

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "detail": self.detail,
                "ms": round(float(self.ms), 2), "ok": bool(self.ok)}


@dataclass(frozen=True)
class Evidence:
    """One fact the recommendation rests on, and where it came from."""
    kind: str                 # fact / risk / related / history / gap
    text: str
    source: str = ""          # the tool call that produced it

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, "source": self.source}


@dataclass(frozen=True)
class Recommendation:
    """What the assistant suggests, and which rule suggested it."""
    kind: str                 # escalate / close_fraud / close_false_positive / keep_open
    rationale: str
    rule: str = ""
    proposed_status: str | None = None
    proposed_state: str | None = None
    proposed_priority: str | None = None
    requires_override: bool = False
    support: int = 0          # how many evidence items point this way

    @property
    def label(self) -> str:
        return {"escalate": "Escalate", "close_fraud": "Close as confirmed fraud",
                "close_false_positive": "Close as false positive",
                "keep_open": "Keep open"}.get(self.kind, self.kind)

    @property
    def changes_anything(self) -> bool:
        return any((self.proposed_status, self.proposed_state, self.proposed_priority))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "label": self.label, "rationale": self.rationale,
                "rule": self.rule, "proposed_status": self.proposed_status,
                "proposed_state": self.proposed_state,
                "proposed_priority": self.proposed_priority,
                "requires_override": bool(self.requires_override),
                "support": int(self.support)}


@dataclass(frozen=True)
class Investigation:
    """Everything one investigation produced. Nothing here is a side effect."""
    case: Case
    transaction: dict
    assessment: Assessment | None
    explanation: Explanation | None
    related: RelatedResult | None
    recommendation: Recommendation
    narrative: str
    evidence: tuple[Evidence, ...] = ()
    steps: tuple[Step, ...] = ()
    gaps: tuple[str, ...] = ()
    history: tuple[CaseEvent, ...] = ()
    writer: str = "template"
    agent: str = AGENT_TAG

    @property
    def risk(self) -> float:
        return float(self.assessment.risk) if self.assessment is not None else float(self.case.risk)

    @property
    def policy_decision(self) -> str:
        return self.assessment.decision if self.assessment is not None else self.case.decision

    def to_dict(self) -> dict:
        return {
            "case": self.case.case_id, "transaction": self.transaction,
            "risk": self.risk, "policy_decision": self.policy_decision,
            "recommendation": self.recommendation.to_dict(),
            "narrative": self.narrative, "writer": self.writer, "agent": self.agent,
            "evidence": [e.to_dict() for e in self.evidence],
            "steps": [s.to_dict() for s in self.steps],
            "gaps": list(self.gaps),
            "history": [{"kind": e.kind, "detail": e.detail, "at": e.created_at}
                        for e in self.history],
            "assessment": None if self.assessment is None else self.assessment.to_dict(),
            "related": None if self.related is None else self.related.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Writers: the prose layer, never the decision layer
# --------------------------------------------------------------------------- #
class TemplateWriter:
    """Deterministic English over the evidence list. The default."""

    name = "template"

    def write(self, brief: dict) -> str:
        c = brief["case"]
        tx = brief["transaction"]
        kind = str(tx["type"]).replace("_", "-").lower()
        if brief["exact"]:
            opening = (f"{c['case_id']} covers transaction #{tx['id']}: a {kind} of "
                       f"{tx['amount']:,.0f} at simulated hour {tx['hour']:02d}:00.")
        else:
            opening = (f"{c['case_id']} covers transaction #{tx['id']}: a {kind} of "
                       f"{tx['amount']:,.0f}, as recorded on the case.")
        sentences = [opening]

        rm = brief["risk_manager"]
        if rm is None:
            sentences.append(f"The case was opened at risk {c['risk']:.1f} with the "
                             f"decision \"{c['decision']}\", and the engine could not be asked for a "
                             f"fresh assessment, so those recorded numbers are all we have.")
        elif not rm["applicable"]:
            sentences.append(f"The engine does not score {kind} transactions, so the policy "
                             f"approved it without a score.")
        else:
            sentences.append(f"The engine scores it {rm['risk']:.1f} and the policy "
                             f"{rm['verb']} ({rm['rule']}).")

        if brief["reasons"]:
            sentences.append("The score was driven by: " + "; ".join(brief["reasons"]) + ".")

        if brief["related_skipped"]:
            sentences.append("Related activity was not searched: "
                             + brief["related_skipped"].rstrip(".") + ".")
        elif brief["related"] is None:
            sentences.append("Related activity was not searched.")
        elif brief["related"]["count"] == 0:
            sentences.append(f"Of {brief['related']['examined']:,} transactions examined, none shared "
                             f"enough evidence to count as related.")
        else:
            sentences.append(brief["related"]["sentence"])

        if brief["history"]:
            sentences.append(brief["history"])

        if brief["drift"]:
            sentences.append(brief["drift"])

        sentences.append(f"Recommendation: {brief['recommendation'].lower()} — {brief['rationale']}")
        return " ".join(sentences)


class LLMWriter:
    """Optional rephrasing through an OpenAI-compatible endpoint.

    This class is only the transport: it sends the numbered facts and the
    recommendation, and refuses an empty or malformed answer. The trust guard --
    rejecting a reply that contains a number the evidence does not -- lives in
    `InvestigationAgent._narrate`, so it cannot be skipped by swapping the
    writer out.
    """

    name = "llm"

    def __init__(self, url: str | None = None, key: str | None = None,
                 model: str | None = None, timeout: float = LLM_TIMEOUT):
        self.url = url or os.environ.get(LLM_URL, "")
        self.key = key if key is not None else os.environ.get(LLM_KEY, "")
        self.model = model or os.environ.get(LLM_MODEL, "gpt-4o-mini")
        self.timeout = timeout

    @classmethod
    def available(cls) -> bool:
        return bool(os.environ.get(LLM_URL))

    def write(self, brief: dict, facts: list[str] | None = None) -> str:
        if not self.url:
            raise RuntimeError(f"{LLM_URL} is not set.")
        facts = facts or []
        numbered = "\n".join(f"[{i}] {text}" for i, text in enumerate(facts, 1))
        system = (
            "You are a fraud investigation assistant writing a short case note. "
            "Use ONLY the numbered facts given to you. Do not invent facts, do not "
            "introduce any number that is not in the facts, and do not make or change "
            "the recommendation: it is given to you and you must repeat it as stated. "
            "Write 3 to 5 sentences of plain English, no headings, no bullet points, "
            "no markdown. Refer to facts as [1], [2] where natural."
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": numbered + "\n\nRecommendation: "
                 + brief["recommendation"] + " — " + brief["rationale"]},
            ],
        }
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.key}"} if self.key else {})},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"the model endpoint failed: {exc}") from exc

        try:
            text = body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("the model endpoint returned an unexpected shape") from exc
        if not text:
            raise RuntimeError("the model returned an empty answer")
        return text


def _numbers(text: str) -> list[str]:
    """Numeric tokens, normalised so '1,250', '1250.0' and '1250' compare equal."""
    out = []
    for token in re.findall(r"\d[\d,]*(?:\.\d+)?", str(text)):
        try:
            out.append(f"{float(token.replace(',', '')):g}")
        except ValueError:                        # unreachable for this pattern, kept safe
            out.append(token)
    return out


# --------------------------------------------------------------------------- #
# The recommendation rules
# --------------------------------------------------------------------------- #
def _rank(priority: str) -> int:
    """Where a priority sits in the queue. Unknown values sort lowest."""
    from src.cases import PRIORITIES

    return PRIORITIES.index(priority) if priority in PRIORITIES else -1


def _recommend(case: Case, assessment: Assessment | None, related: RelatedResult | None,
               cluster: int) -> Recommendation:
    """The explicit rule list. Deterministic, order-sensitive, no model involved.

    Each rule states what it saw, so a reader can disagree with the rule rather
    than with a black box. The Risk Manager's decision is an input, never an
    output: when a rule recommends something the policy did not, the proposal is
    flagged as an override.

    No rule proposes a change that is already true. A recommendation to do what
    the case already says is worse than no recommendation: it fills the approval
    queue with no-ops and teaches the analyst to stop reading.
    """
    decision = assessment.decision if assessment is not None else case.decision
    risk = float(assessment.risk) if assessment is not None else float(case.risk)

    # R1: the policy blocks it, the case is not at the top of the queue -> say so.
    # The check is on the case's own priority, not on the risk: a blocked case
    # whose analyst already marked it urgent (R2) needs no further advice.
    if decision == BLOCK and case.priority != "urgent":
        return Recommendation(
            kind="escalate", rule="R1",
            rationale=(f"the policy already blocks this transaction (risk {risk:.1f}), so the case "
                       f"should carry urgent priority rather than {case.priority_label.lower()}"),
            proposed_priority="urgent", support=2)

    # R2: blocked and already urgent -> the queue is right, leave the records alone.
    if decision == BLOCK:
        return Recommendation(
            kind="keep_open", rule="R2",
            rationale=(f"the transaction is blocked and the case is already urgent; there is "
                       f"nothing to change while the investigation is open"),
            support=1)

    # R3: very high risk sent to review -> it belongs at the top of the queue.
    # If it already is, say that instead of filing a proposal that changes nothing.
    if decision == REVIEW and risk >= 95.0:
        if _rank(case.priority) >= _rank("urgent"):
            return Recommendation(
                kind="keep_open", rule="R3",
                rationale=(f"risk {risk:.1f} is far above the review threshold and the case is "
                           f"already urgent; the queue position is right, so there is nothing to "
                           f"change"),
                support=2)
        return Recommendation(
            kind="escalate", rule="R3",
            rationale=(f"risk {risk:.1f} is far above the review threshold, so the case should be "
                       f"worked before anything else"),
            proposed_priority="urgent", support=2)

    # R4: a cluster of near-threshold transactions around a reviewed one.
    if decision == REVIEW and cluster >= CLUSTER_MIN:
        if _rank(case.priority) >= _rank("high"):
            return Recommendation(
                kind="keep_open", rule="R4",
                rationale=(f"{cluster} related transactions also score at or above "
                           f"{CLUSTER_RISK:.0f}, but the case already carries "
                           f"{case.priority_label.lower()} priority; nothing to change"),
                support=1 + cluster)
        return Recommendation(
            kind="escalate", rule="R4",
            rationale=(f"{cluster} related transactions also score at or above {CLUSTER_RISK:.0f}; "
                       f"several look-alikes matter more than one"),
            proposed_priority="high", support=1 + cluster)

    # R5: the engine scored it and the policy approved it, but it sits in a
    # suspicious cluster. This is the one rule that can contradict the policy on
    # purpose -- and it says so, so approving it takes a deliberate override.
    if decision == APPROVE and cluster >= CLUSTER_MIN and getattr(assessment, "applicable", False):
        if _rank(case.priority) >= _rank("high"):
            return Recommendation(
                kind="keep_open", rule="R5",
                rationale=(f"the policy approves this transaction but {cluster} related "
                           f"transactions score at or above {CLUSTER_RISK:.0f}; the case already "
                           f"carries {case.priority_label.lower()} priority, so it stays where "
                           f"it is"),
                support=cluster)
        return Recommendation(
            kind="escalate", rule="R5", requires_override=True,
            rationale=(f"the policy approves this transaction, but {cluster} related transactions "
                       f"score at or above {CLUSTER_RISK:.0f}; escalating overrides the policy"),
            proposed_priority="high", support=cluster)

    # R6: the policy approves it and nothing else points at it -> likely a false positive.
    if decision == APPROVE and cluster < CLUSTER_MIN and not case.closed:
        return Recommendation(
            kind="close_false_positive", rule="R6",
            rationale=("the current policy approves this transaction and no related activity "
                       "supports it, so the alert looks like a false positive"),
            proposed_status="false_positive", proposed_state="closed", support=1)

    # R7: everything else -- leave it to the analyst. That includes a transaction
    # the engine cannot score (no fraud examples of its type) sitting next to
    # suspicious look-alikes: closing it would be a guess, so it stays open.
    return Recommendation(
        kind="keep_open", rule="R7",
        rationale="the evidence is not strong enough to close the case either way",
        support=0)


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
class InvestigationAgent:
    """One assistant over one toolkit: reads everything, writes only proposals.

    `proposals` is optional: without it `investigate()` still works and returns
    the investigation, which is what tests and scripts use. `file()` needs it.

    `lookup` is how the assistant gets the transaction's real feature values
    (type, amount, both balances, hour). The case record only keeps the amount
    and the type, and scoring or matching related transactions without the
    balances would quietly produce *wrong* evidence -- the engine keys on them.
    When `lookup` returns nothing the assistant says so and falls back to the
    numbers the case recorded, instead of inventing zeroes.
    """

    def __init__(self, tools: Toolkit, *, proposals: ProposalStore | None = None,
                 corpus: Any = None, lookup: Callable[[int], dict | None] | None = None,
                 writer: Any = None, agent: str = AGENT_TAG):
        self.tools = tools
        self.proposals = proposals
        self.corpus = corpus
        self.lookup = lookup
        self.writer = writer or TemplateWriter()
        self.agent = agent

    # -- the investigation ---------------------------------------------- #
    def investigate(self, case_id: str | int, *, corpus: Any = None,
                    related_limit: int = 5) -> Investigation:
        """Read the case and everything around it. Does not write anything."""
        steps: list[Step] = []
        evidence: list[Evidence] = []
        gaps: list[str] = []

        # 1 -- receive the case -----------------------------------------
        started = time.perf_counter()
        found = self.tools.get_case(case_id)
        if found is None:
            raise ToolError(f"Case {case_id!r} does not exist.")
        case, history = found.case, found.events
        steps.append(Step("receive_case", "Case received",
                          f"{case.case_id} · {case.priority_label} priority · {case.status_label} · "
                          f"{case.state_label.lower()}", _ms(started)))
        evidence.append(Evidence("history",
                                 f"{case.case_id} was opened for transaction #{case.txn_id} at risk "
                                 f"{case.risk:.1f} with the decision \"{case.decision}\" and is "
                                 f"{case.state_label.lower()} ({case.status_label.lower()}), with "
                                 f"{len(history)} recorded event(s).",
                                 "get_case"))
        if history:
            evidence.append(Evidence("history",
                                     "Last recorded event: " + history[0].detail + ".",
                                     "get_case"))

        # 2 -- transaction details ---------------------------------------
        started = time.perf_counter()
        transaction, exact = self._transaction(case)
        steps.append(Step("transaction_details", "Transaction details",
                          f"{transaction['type'].replace('_', ' ').title()} · "
                          f"{transaction['amount']:,.0f} · hour {transaction['hour']:02d}"
                          + ("" if exact else " · from the case record"),
                          _ms(started), ok=exact))
        evidence.append(Evidence("fact", self._transaction_text(transaction, exact),
                                 "live feed" if exact else "case record"))
        if not exact:
            gaps.append(
                f"The live row for transaction #{case.txn_id} is not available in this session, "
                f"so the engine was not asked to re-score it and related activity was not "
                f"searched. Without the account balances those answers would be wrong rather "
                f"than merely missing; the score recorded when the case was opened stands.")

        # 3 -- related activity ------------------------------------------
        started = time.perf_counter()
        source = corpus if corpus is not None else self.corpus
        related = None
        related_skipped = ""
        if not exact:
            related_skipped = ("the live row for this transaction is not available, and matching "
                               "on the amount alone would produce false matches")
            steps.append(Step("related_activity", "Related activity",
                              "skipped · no live row to search with", _ms(started), ok=False))
        elif source is None or (hasattr(source, "empty") and bool(source.empty)):
            related_skipped = ("there is no corpus of transactions to compare against in this "
                               "session")
            gaps.append("Related activity was not searched: " + related_skipped + ".")
            steps.append(Step("related_activity", "Related activity",
                              "skipped · no corpus available", _ms(started), ok=False))
        else:
            try:
                related = self.tools.find_related_transactions(transaction, corpus=source,
                                                               limit=related_limit)
            except ToolError as exc:
                related_skipped = str(exc)
                gaps.append(f"Related activity could not be searched: {exc}")
                steps.append(Step("related_activity", "Related activity", f"failed · {exc}",
                                  _ms(started), ok=False))
            else:
                steps.append(Step("related_activity", "Related activity",
                                  f"{len(related.items)} related of {related.examined:,} examined",
                                  _ms(started)))
                for item in related.items:
                    evidence.append(Evidence(
                        "related",
                        f"#{item.transaction['id']} · {item.transaction['type'].replace('_', ' ').title()} "
                        f"· {item.transaction['amount']:,.0f}"
                        + (f" · risk {item.risk:.1f} ({item.decision})" if item.risk is not None else "")
                        + (" — " + ", ".join(item.relation) if item.relation else ""),
                        "find_related_transactions"))

        # 4 -- ask the Risk Manager ---------------------------------------
        started = time.perf_counter()
        assessment = explanation = None
        if not exact:
            steps.append(Step("risk_assessment", "Risk Manager assessment",
                              "skipped · the transaction's feature values are not available",
                              _ms(started), ok=False))
        else:
            try:
                assessment = self.tools.assess_risk(transaction)
                explanation = self.tools.explain_transaction(transaction)
            except ToolError as exc:
                gaps.append(f"The Risk Manager could not score this transaction: {exc}")
                steps.append(Step("risk_assessment", "Risk Manager assessment", f"failed · {exc}",
                                  _ms(started), ok=False))
            else:
                steps.append(Step("risk_assessment", "Risk Manager assessment",
                                  f"risk {assessment.risk:.1f} · {assessment.decision} · "
                                  f"rule {assessment.policy_hash}",
                                  _ms(started)))
                evidence.append(Evidence("risk",
                                         f"The Risk Manager scores the transaction {assessment.risk:.1f} "
                                         f"and {_verb(assessment.decision)}: {assessment.rule}.",
                                         "assess_risk"))
                for reason in explanation.reasons_up[:3]:
                    evidence.append(Evidence("risk", f"{reason.label} — {reason.text}",
                                             "explain_transaction"))
                # The score at open time is part of the record; if the engine or
                # the policy has moved since, say so rather than quietly
                # ignoring it.
                if abs(assessment.risk - case.risk) >= 1.0 or assessment.decision != case.decision:
                    gaps.append(
                        f"The case was opened at risk {case.risk:.1f} with the decision "
                        f"\"{case.decision}\"; the current engine and policy give {assessment.risk:.1f} "
                        f"and \"{assessment.decision}\". Both are kept so the change is visible.")

        # 5 -- summarise the evidence --------------------------------------
        started = time.perf_counter()
        cluster = sum(1 for item in (related.items if related else ())
                      if item.risk is not None and item.risk >= CLUSTER_RISK)
        recommendation = _recommend(case, assessment, related, cluster)
        steps.append(Step("evidence_summary", "Evidence summary",
                          f"{len(evidence)} evidence item(s), {len(gaps)} gap(s), "
                          f"{cluster} in the cluster", _ms(started)))

        # 6 -- recommend ----------------------------------------------------
        started = time.perf_counter()
        steps.append(Step("recommendation", "Recommendation",
                          f"{recommendation.label} (rule {recommendation.rule})", _ms(started)))

        # 7 -- write it up (the proposal itself is filed separately) --------
        brief = self._brief(case, transaction, assessment, explanation, related,
                            recommendation, history, gaps, exact=exact,
                            related_skipped=related_skipped)
        started = time.perf_counter()
        narrative, writer_name, writer_note = self._narrate(brief, evidence)
        if writer_note:
            gaps.append(writer_note)
        steps.append(Step("proposal", "Proposal",
                          f"written by {writer_name} · awaiting a human decision", _ms(started)))

        return Investigation(
            case=case, transaction=transaction, assessment=assessment, explanation=explanation,
            related=related, recommendation=recommendation, narrative=narrative,
            evidence=tuple(evidence), steps=tuple(steps), gaps=tuple(gaps),
            history=tuple(history), writer=writer_name, agent=self.agent)

    # -- filing it ------------------------------------------------------ #
    def file(self, investigation: Investigation) -> Proposal:
        """Turn an investigation into a pending proposal. The only write.

        Note what this does *not* do: it does not touch the case. Filing is the
        assistant's whole output; `approve()` is what changes anything, and it
        needs a person.
        """
        if self.proposals is None:
            raise ToolError("This agent has no proposal store, so it cannot file anything.")
        rec = investigation.recommendation
        case = investigation.case
        # Second gate, independent of the rules: a proposal nobody needs to
        # approve is noise in the queue. If every proposed value already holds,
        # there is nothing to decide, so nothing is filed.
        pending = [
            (rec.proposed_status, case.status),
            ("closed" if rec.proposed_state == "closed" else "open" if rec.proposed_state else "",
             "closed" if case.closed else "open"),
            (rec.proposed_priority, case.priority),
        ]
        if not any(value and value != current for value, current in pending):
            raise ToolError(
                f"{case.case_id} already matches this recommendation "
                f"({rec.label.lower()}), so there is nothing to approve. "
                f"The investigation is still shown; only the proposal was skipped.")
        return self.proposals.create(
            investigation.case.case_id, rec.kind, rationale=rec.rationale,
            narrative=investigation.narrative,
            evidence=_evidence_payload(investigation),
            risk=investigation.risk, policy_decision=investigation.policy_decision,
            proposed_status=rec.proposed_status, proposed_state=rec.proposed_state,
            proposed_priority=rec.proposed_priority, agent=f"{self.agent}/{investigation.writer}",
            requires_override=rec.requires_override)

    # -- internals ------------------------------------------------------ #
    def _transaction(self, case: Case) -> tuple[dict, bool]:
        """The transaction's real values, and whether they are the real ones.

        `exact=False` means the row is gone from the feed (trimmed, or the case
        came from another mode) and we only have what the case recorded. The
        caller uses that to skip the steps that would be wrong without balances.
        """
        row = None
        if self.lookup is not None:
            try:
                row = self.lookup(int(case.txn_id))
            except Exception:                     # a lookup failure must not sink the investigation
                row = None
        if row is not None:
            try:                                  # never `if row:` -- a Series has no truth value
                return as_transaction(row).to_dict(), True
            except ToolError:
                pass
        return ({"id": int(case.txn_id), "type": case.tx_type or "UNKNOWN",
                 "amount": float(case.amount), "oldbalanceOrg": 0.0, "oldbalanceDest": 0.0,
                 "hour": 0, "step": 1, "nameOrig": "", "nameDest": ""}, False)

    @staticmethod
    def _transaction_text(transaction: dict, exact: bool = True) -> str:
        head = (f"Transaction #{transaction['id']}: "
                f"{transaction['type'].replace('_', ' ').title()} of "
                f"{transaction['amount']:,.0f}")
        if not exact:
            return f"{head}, as recorded on the case."
        return (f"{head} at simulated hour {transaction['hour']:02d}:00, "
                f"sender balance {transaction['oldbalanceOrg']:,.0f}, "
                f"receiver balance {transaction['oldbalanceDest']:,.0f}.")

    def _brief(self, case: Case, transaction: dict, assessment: Assessment | None,
               explanation: Explanation | None, related: RelatedResult | None,
               recommendation: Recommendation, history, gaps, *, exact: bool = True,
               related_skipped: str = "") -> dict:
        reasons = [f"{r.label} — {r.text}" for r in (explanation.reasons_up[:3] if explanation else [])]
        rm = None
        if assessment is not None:
            rm = {"risk": assessment.risk, "decision": assessment.decision,
                  "applicable": assessment.applicable, "rule": assessment.rule,
                  "verb": _verb(assessment.decision)}
        return {
            "case": {"case_id": case.case_id, "risk": float(case.risk),
                     "decision": case.decision, "priority": case.priority,
                     "status": case.status, "state": case.state_label.lower()},
            "transaction": transaction,
            "exact": bool(exact),
            "risk_manager": rm,
            "reasons": reasons,
            "related": self._related_brief(related),
            "related_skipped": related_skipped,
            "recommendation": recommendation.label,
            "rationale": recommendation.rationale,
            "history": (f"The case carries {len(history)} recorded event(s) and is "
                        f"{case.state_label.lower()}.") if history else "",
            "drift": next((g for g in gaps if g.startswith("The case was opened at")), ""),
        }

    @staticmethod
    def _related_brief(related: RelatedResult | None) -> dict | None:
        if related is None:
            return None
        if not related.items:
            return {"count": 0, "examined": related.examined, "sentence": ""}
        best = related.items[0]
        why = ", ".join(best.relation) if best.relation else "shared evidence"
        sentence = (f"{len(related.items)} related transaction(s) turned up; the closest is "
                    f"#{best.transaction['id']} "
                    + (f"at risk {best.risk:.1f} " if best.risk is not None else "")
                    + f"because it has {why}.")
        return {"count": len(related.items), "examined": related.examined, "sentence": sentence}

    def _narrate(self, brief: dict, evidence: list[Evidence]) -> tuple[str, str, str]:
        """Template prose, or the model's rephrasing when it passes both guards.

        The number guard lives here rather than in the writer on purpose: it has
        to hold for *any* writer plugged in, not just the one that remembers to
        check. A writer is a text function; the trust boundary is the agent.
        """
        template = TemplateWriter().write(brief)
        writer = self.writer
        if writer is None or getattr(writer, "name", "template") == "template":
            return template, "template", ""
        facts = [e.text for e in evidence if e.kind != "gap"]
        try:
            text = writer.write(brief, facts=facts)
        except (RuntimeError, ToolError, ValueError) as exc:
            return (template, "template",
                    f"The language model was not used ({exc}); the template wording is shown instead.")

        allowed = set(_numbers(" ".join(facts) + " " + json.dumps(brief["transaction"])))
        invented = [n for n in _numbers(text) if n not in allowed]
        if invented:
            return (template, "template",
                    f"The language model was not used because it introduced numbers that are not "
                    f"in the evidence ({', '.join(invented[:3])}); the template wording is shown "
                    f"instead.")
        return text, getattr(writer, "name", "llm"), ""


def _verb(decision: str) -> str:
    return {BLOCK: "blocks it", REVIEW: "sends it to review",
            APPROVE: "approves it"}.get(decision, f"decides \"{decision}\"")


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _evidence_payload(investigation: Investigation) -> dict:
    """What gets stored with a proposal: enough to re-read the reasoning later."""
    return {
        "evidence": [e.to_dict() for e in investigation.evidence],
        "steps": [{"label": s.label, "detail": s.detail} for s in investigation.steps],
        "gaps": list(investigation.gaps),
        "related": [] if investigation.related is None else [
            {"id": item.transaction.get("id"), "amount": item.transaction.get("amount"),
             "risk": item.risk, "relation": list(item.relation)}
            for item in investigation.related.items],
        "transaction": investigation.transaction,
    }


# --------------------------------------------------------------------------- #
# Approval: the only path from a proposal to a changed case
# --------------------------------------------------------------------------- #
def approve(proposals: ProposalStore, tools: Toolkit, proposal: int | Proposal, *,
            by: str = "analyst", note: str = "", override: bool = False) -> CaseResult:
    """Execute an approved proposal through the tool layer, then record the verdict.

    This is the function item 8 of the roadmap was about: the assistant asked,
    a person said yes, and only now does the case move -- through `update_case`,
    the same call the analyst's own buttons make, so the audit trail cannot tell
    an approved agent action from a hand edit and does not need to.
    """
    # Again, re-read: the caller's copy may be older than somebody else's decision.
    number = proposal.number if isinstance(proposal, Proposal) else proposal
    item = proposals.get(number)
    if item is None:
        raise ToolError("That proposal does not exist.")
    if not item.pending:
        raise ToolError(f"Proposal {item.number} is already {item.status_label.lower()}.")
    if item.requires_override and not override:
        raise ToolError(
            f"Proposal {item.number} recommends {item.kind_label.lower()} against the risk "
            f"policy's own decision ({item.policy_decision}). Approving it needs the explicit "
            f"override.")

    text = f"Assistant ({item.agent}): {item.rationale}"
    if note.strip():
        text += f"\n\nApproved by {by}: {note.strip()}"
    result = tools.update_case(
        item.case_id, status=item.proposed_status, state=item.proposed_state,
        priority=item.proposed_priority, note=text)
    proposals.decide(item.number, True, by=by, note=note.strip())
    return result


def reject(proposals: ProposalStore, proposal: int | Proposal, *, by: str = "analyst",
           note: str = "") -> Proposal:
    """Turn a proposal down. The case is deliberately left exactly as it was."""
    number = proposal.number if isinstance(proposal, Proposal) else proposal
    item = proposals.get(number)
    if item is None:
        raise ToolError("That proposal does not exist.")
    if not item.pending:
        raise ToolError(f"Proposal {item.number} is already {item.status_label.lower()}.")
    return proposals.decide(item.number, False, by=by, note=note.strip())
