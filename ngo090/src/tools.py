"""The tool layer: one deterministic, Streamlit-free API over the whole system.

The dashboard, the command line and (later) an agent should all reach the same
behaviour. Instead of each of them importing the engine, the policy, the case
store and the audit log separately, they call the functions here:

    score_transaction(transaction)          how suspicious is it?          (pure)
    assess_risk(transaction)               what does the policy do?       (pure)
    explain_transaction(transaction)       why? which facts mattered?     (pure)
    find_related_transactions(transaction) what else looks like it?       (pure)
    create_case(transaction)               keep it for investigation      (writes)
    get_case(case_id)                      the case and its history       (reads)
    update_case(case_id, ...)              move it along                  (writes)

Design rules
------------
    * No Streamlit, no session state, no UI imports. Everything here runs in a
      plain Python process, a test or a worker.
    * Same inputs, same outputs. The scoring functions have no randomness and
      no wall-clock: the same engine, transaction and policy always give the
      same numbers. Only create_case / update_case write, and they write to
      SQLite in one transaction each.
    * Inputs are forgiving, outputs are plain. A transaction may be a dict, a
      pandas Series, a row from itertuples() or a Transaction; results are
      dataclasses with to_dict() and contain only JSON-safe values.
    * The decision is explicit. score_transaction never decides; assess_risk
      applies a Policy (the toolkit's own by default) and says which rule fired.

A toolkit is one configured system: a risk engine, a policy, a case store and
(optionally) a corpus of transactions to compare against. The app builds one
per run; a script builds one with open_toolkit():

    from src.tools import open_toolkit

    tools = open_toolkit(".")
    print(tools.assess_risk({"type": "TRANSFER", "amount": 250000,
                             "oldbalanceOrg": 250000, "oldbalanceDest": 0}).to_dict())
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.artifacts import discover_runs, load_engine
from src.audit import AuditLog, policy_snapshot, snapshot_hash
from src.cases import Case, CaseEvent, CaseStore, priority_for
from src.decision import APPROVE, BLOCK, REVIEW, Policy, decide_one
from src.engine import RiskEngine
from src.explain import Reason
from src.predict import DEFAULT_POLICY, evaluate_one, validate_transaction
from src.processing import DEFAULT_HOUR, score_clean

CASES_DB = "FRAUD_CASES_DB"        # environment overrides, same names as app/common.py
AUDIT_DB = "FRAUD_AUDIT_DB"
RELATED_LIMIT = 5
RELATED_MIN_SCORE = 2.0


class ToolError(ValueError):
    """The caller asked for something the tools cannot do with what they were given."""


# --------------------------------------------------------------------------- #
# Transactions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Transaction:
    """One transaction, in the shape the models and the case store understand.

    `id` is optional because a transaction typed by hand has none; `step` is
    optional because the app thinks in hours while PaySim thinks in steps.
    """
    type: str
    amount: float
    oldbalanceOrg: float
    oldbalanceDest: float
    hour: int = DEFAULT_HOUR          # 0-23
    id: int | None = None
    step: int | None = None           # 1-based simulated hour, derived from `hour` if absent
    nameOrig: str = ""                # account ids: only present in raw PaySim data
    nameDest: str = ""

    def to_dict(self) -> dict:
        return {
            "id": None if self.id is None else int(self.id),
            "type": self.type,
            "amount": float(self.amount),
            "oldbalanceOrg": float(self.oldbalanceOrg),
            "oldbalanceDest": float(self.oldbalanceDest),
            "hour": int(self.hour),
            "step": None if self.step is None else int(self.step),
            "nameOrig": self.nameOrig,
            "nameDest": self.nameDest,
        }

    def as_model_input(self) -> dict:
        """The dict predict.evaluate_one expects."""
        return {"type": self.type, "amount": self.amount,
                "oldbalanceOrg": self.oldbalanceOrg, "oldbalanceDest": self.oldbalanceDest,
                "hour": self.hour}


def _as_mapping(value: Any) -> Mapping:
    """A dict-like view of a dict, Series, namedtuple, dataclass or sqlite row."""
    if isinstance(value, Mapping):
        return value
    if isinstance(value, pd.Series):
        return value.to_dict()
    if hasattr(value, "_asdict"):                  # namedtuple / itertuples row
        return value._asdict()
    if hasattr(value, "__dict__"):
        return vars(value)
    try:
        return dict(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(
            "A transaction must be a dict, a pandas Series or a row with fields; "
            f"got {type(value).__name__}."
        ) from exc


def _pick(data: Mapping, *names: str, default: Any = None) -> Any:
    for name in names:
        if name in data:
            value = data[name]
            if value is None:
                continue
            if isinstance(value, (float, np.floating)) and math.isnan(value):
                continue
            return value
    return default


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"{field_name} must be a whole number (got {value!r}).") from exc


def as_transaction(value: Any) -> Transaction:
    """Normalise anything transaction-shaped into a validated Transaction.

    Raises ToolError (a ValueError) listing every problem, so a caller can show
    the message as it is.
    """
    if isinstance(value, Transaction):
        return value
    data = _as_mapping(value)

    step = _optional_int(_pick(data, "step"), "step")
    if step is not None and step < 1:
        raise ToolError("step must be 1 or more.")
    hour = _optional_int(_pick(data, "hour"), "hour")
    if hour is None:
        hour = DEFAULT_HOUR if step is None else (step - 1) % 24
    if step is None:
        step = hour + 1

    raw = {
        "type": _pick(data, "type", "tx_type", default=""),
        "amount": _pick(data, "amount"),
        "oldbalanceOrg": _pick(data, "oldbalanceOrg", "oldbalance_org"),
        "oldbalanceDest": _pick(data, "oldbalanceDest", "oldbalance_dest"),
        "hour": hour,
    }
    clean, errors = validate_transaction(raw)
    if errors:
        raise ToolError(" ".join(errors))

    txn_id = _pick(data, "id", "txn_id", "transaction_id")
    return Transaction(
        type=clean["type"], amount=clean["amount"],
        oldbalanceOrg=clean["oldbalanceOrg"], oldbalanceDest=clean["oldbalanceDest"],
        hour=clean["hour"], id=None if txn_id is None else _optional_int(txn_id, "id"),
        step=step,
        nameOrig=str(_pick(data, "nameOrig", "name_orig", default="") or ""),
        nameDest=str(_pick(data, "nameDest", "name_dest", default="") or ""),
    )


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def _reason_dict(reason: Reason) -> dict:
    return {"feature": reason.feature, "label": reason.label, "text": reason.text,
            "contribution": float(reason.contribution), "share": float(reason.share)}


@dataclass(frozen=True)
class Score:
    """How suspicious one transaction is, and which detector drove that."""
    transaction: Transaction
    risk: float
    applicable: bool
    signal: str = ""
    components: dict = field(default_factory=dict)
    model_tag: str = ""
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"transaction": self.transaction.to_dict(), "risk": float(self.risk),
                "applicable": bool(self.applicable), "signal": self.signal,
                "components": {k: float(v) for k, v in self.components.items()},
                "model_tag": self.model_tag, "notes": list(self.notes)}


@dataclass(frozen=True)
class Assessment:
    """The risk plus the action the policy takes and the rule that fired."""
    score: Score
    decision: str
    rule: str
    policy: dict
    policy_hash: str

    @property
    def risk(self) -> float:
        return self.score.risk

    @property
    def applicable(self) -> bool:
        return self.score.applicable

    def to_dict(self) -> dict:
        out = self.score.to_dict()
        out.update({"decision": self.decision, "rule": self.rule,
                    "policy": self.policy, "policy_hash": self.policy_hash})
        return out


@dataclass(frozen=True)
class Explanation:
    """Why the score is what it is: what raised it, what lowered it.

    Deliberately mirrors predict.Prediction's field names, so the dashboard's
    existing result card can show a tool result unchanged.
    """
    transaction: Transaction
    risk: float
    decision: str
    applicable: bool
    signal: str
    reasons_up: tuple[Reason, ...] = ()
    reasons_down: tuple[Reason, ...] = ()
    notes: tuple[str, ...] = ()
    model_tag: str = ""

    def to_dict(self) -> dict:
        return {"transaction": self.transaction.to_dict(), "risk": float(self.risk),
                "decision": self.decision, "applicable": bool(self.applicable),
                "signal": self.signal, "model_tag": self.model_tag,
                "reasons": {"raised": [_reason_dict(r) for r in self.reasons_up],
                            "lowered": [_reason_dict(r) for r in self.reasons_down]},
                "notes": list(self.notes)}


@dataclass(frozen=True)
class RelatedTransaction:
    """One transaction that resembles the one being investigated, and why."""
    transaction: dict
    relation: tuple[str, ...]
    similarity: float
    risk: float | None = None
    decision: str | None = None

    def to_dict(self) -> dict:
        return {"transaction": self.transaction, "relation": list(self.relation),
                "similarity": float(self.similarity),
                "risk": None if self.risk is None else float(self.risk),
                "decision": self.decision}


@dataclass(frozen=True)
class RelatedResult:
    transaction: dict
    examined: int
    items: tuple[RelatedTransaction, ...] = ()

    def to_dict(self) -> dict:
        return {"transaction": self.transaction, "examined": int(self.examined),
                "related": [item.to_dict() for item in self.items]}


@dataclass(frozen=True)
class CaseResult:
    """A case plus its history; `created` is False when it already existed."""
    case: Case
    created: bool = False
    events: tuple[CaseEvent, ...] = ()

    def to_dict(self) -> dict:
        case = self.case
        return {
            "case": {
                "number": int(case.number), "case_id": case.case_id,
                "transaction": int(case.txn_id), "risk": float(case.risk),
                "decision": case.decision, "status": case.status,
                "status_label": case.status_label, "state": "closed" if case.closed else "open",
                "state_label": case.state_label, "priority": case.priority,
                "priority_label": case.priority_label, "notes": case.notes,
                "explanation": case.explanation, "amount": float(case.amount),
                "tx_type": case.tx_type, "source": case.source,
                "created_at": case.created_at, "updated_at": case.updated_at,
            },
            "created": bool(self.created),
            "history": [{"id": e.number, "kind": e.kind, "detail": e.detail,
                         "at": e.created_at} for e in self.events],
        }


# --------------------------------------------------------------------------- #
# The toolkit
# --------------------------------------------------------------------------- #
def _policy_text(value: float) -> str:
    value = float(value)
    if math.isinf(value):
        return "never" if value > 0 else "off"
    return f"{value:g}"


def _rule_text(score: Score, decision: str, policy: Policy) -> str:
    """One sentence an analyst (or an agent's log) can read without the code."""
    if not score.applicable:
        return "not scored: this transaction type has no fraud examples in the dataset"
    if decision == BLOCK:
        return (f"risk {score.risk:.2f} is at or above the auto-block threshold "
                f"({_policy_text(policy.block_threshold)})")
    if decision == REVIEW:
        return (f"risk {score.risk:.2f} is at or above the review threshold "
                f"({_policy_text(policy.review_min_score)}) and review capacity is "
                f"{int(policy.daily_capacity)}")
    if policy.daily_capacity == 0:
        return (f"risk {score.risk:.2f} is below the block threshold "
                f"({_policy_text(policy.block_threshold)}) and the policy has no review capacity")
    return (f"risk {score.risk:.2f} is below the review threshold "
            f"({_policy_text(policy.review_min_score)})")


def _column(frame: pd.DataFrame, *names: str) -> pd.Series | None:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return None


def _numeric(frame: pd.DataFrame, *names: str) -> pd.Series | None:
    """The first column that exists, as numbers; None when none of them exists."""
    values = _column(frame, *names)
    return None if values is None else pd.to_numeric(values, errors="coerce")


def _parse_state(value: Any) -> bool:
    """'open' / 'closed' (or a bool) -> the closed flag."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("closed", "close", "true", "1"):
        return True
    if text in ("open", "reopened", "false", "0"):
        return False
    raise ToolError(f"state must be 'open' or 'closed' (got {value!r}).")


class Toolkit:
    """One configured system: engine + policy + case store + audit log + corpus.

    Nothing is cached and nothing is global, so several toolkits (say, one per
    policy) can exist side by side in the same process without interfering.
    """

    def __init__(self, engine: RiskEngine | None = None, *,
                 policy: Policy | None = None,
                 cases: CaseStore | None = None,
                 audit: AuditLog | None = None,
                 corpus: pd.DataFrame | Iterable | None = None,
                 model_tag: str = ""):
        self.engine = engine
        self.policy = policy
        self.cases = cases
        self.audit = audit
        self.corpus = corpus
        self.model_tag = model_tag

    # -- plumbing ------------------------------------------------------- #
    def _policy(self, policy: Policy | None) -> Policy:
        return policy or self.policy or DEFAULT_POLICY

    def _case_store(self) -> CaseStore:
        if self.cases is None:
            raise ToolError("This toolkit has no case store. Create it with cases=CaseStore(path).")
        return self.cases

    def _result(self, case: Case, created: bool = False) -> CaseResult:
        store = self._case_store()
        return CaseResult(case, created=created, events=tuple(store.events(case.number)))

    # -- reading -------------------------------------------------------- #
    def score_transaction(self, transaction: Any) -> Score:
        """Risk score, engine coverage and the detector behind it. Pure."""
        if self.engine is None:
            raise ToolError("This toolkit has no risk engine. Create it with engine=load_engine(...).")
        tx = as_transaction(transaction)
        evaluation = evaluate_one(self.engine, tx.as_model_input(), explain=False)
        prediction = evaluation.prediction
        return Score(transaction=tx, risk=prediction.risk, applicable=prediction.applicable,
                     signal=evaluation.signal, components=evaluation.components,
                     model_tag=self.model_tag, notes=tuple(prediction.notes))

    def assess_risk(self, transaction: Any, policy: Policy | None = None) -> Assessment:
        """The action the policy takes for this transaction, and why. Pure."""
        score = self.score_transaction(transaction)
        policy = self._policy(policy)
        decision = decide_one(score.risk, policy) if score.applicable else APPROVE
        snapshot = policy_snapshot(policy)
        return Assessment(score=score, decision=decision, rule=_rule_text(score, decision, policy),
                          policy=snapshot, policy_hash=snapshot_hash(snapshot))

    def explain_transaction(self, transaction: Any, policy: Policy | None = None,
                            top_k: int = 3) -> Explanation:
        """The facts that raised and lowered the risk, with the decision. Pure."""
        if self.engine is None:
            raise ToolError("This toolkit has no risk engine. Create it with engine=load_engine(...).")
        tx = as_transaction(transaction)
        policy = self._policy(policy)
        evaluation = evaluate_one(self.engine, tx.as_model_input(), policy, top_k=top_k, explain=True)
        prediction = evaluation.prediction
        return Explanation(transaction=tx, risk=prediction.risk, decision=prediction.decision,
                           applicable=prediction.applicable, signal=evaluation.signal,
                           reasons_up=tuple(prediction.reasons_up),
                           reasons_down=tuple(prediction.reasons_down),
                           notes=tuple(prediction.notes), model_tag=self.model_tag)

    def find_related_transactions(self, transaction: Any, *,
                                  corpus: pd.DataFrame | Iterable | None = None,
                                  limit: int = RELATED_LIMIT,
                                  min_score: float = RELATED_MIN_SCORE) -> RelatedResult:
        """Transactions that resemble this one, most similar first. Pure.

        "Related" is evidence, not proof: it ranks by shared account (when the
        data has account ids), same type, a close amount, the same
        empty-receiver or emptied-sender pattern and nearby time. Each result
        carries the reasons it matched, so a person can judge them.
        """
        tx = as_transaction(transaction)
        frame = self._corpus_frame(corpus if corpus is not None else self.corpus)
        if len(frame) == 0:
            return RelatedResult(transaction=tx.to_dict(), examined=0)

        scores, signals = self._similarity(frame, tx)
        keep = scores >= float(min_score)
        if tx.id is not None:
            keep &= frame["id"].to_numpy(dtype=float) != int(tx.id)
        positions = np.flatnonzero(keep)
        if len(positions) == 0:
            return RelatedResult(transaction=tx.to_dict(), examined=len(frame))

        order = np.lexsort((positions, -scores[positions]))[:max(int(limit), 0)]
        chosen = positions[order]
        risks, decisions = self._score_shortlist(frame.iloc[chosen])

        items = []
        for row_pos, risk, decision in zip(chosen, risks, decisions):
            items.append(RelatedTransaction(
                transaction=self._frame_transaction(frame.iloc[row_pos]),
                relation=self._relation_text(signals, row_pos, tx),
                similarity=float(scores[row_pos]),
                risk=risk, decision=decision,
            ))
        return RelatedResult(transaction=tx.to_dict(), examined=len(frame), items=tuple(items))

    # -- case work ------------------------------------------------------ #
    def create_case(self, transaction: Any, *, policy: Policy | None = None,
                    priority: str | None = None, note: str = "", source: str = "tool") -> CaseResult:
        """Keep this transaction for investigation. Idempotent per open case.

        If the transaction already has an open case, that case is returned with
        created=False: a double click or a retried agent call cannot open two
        cases for the same alert.
        """
        store = self._case_store()
        tx = as_transaction(transaction)
        if tx.id is None:
            raise ToolError("create_case needs a transaction id (the 'id' field of the transaction).")
        existing = store.open_for(int(tx.id))
        if existing is not None:
            return self._result(existing, created=False)

        explanation = self.explain_transaction(tx, policy=policy)
        main = explanation.reasons_up[0] if explanation.reasons_up else None
        if main is not None:
            text = f"{main.label} - {main.text}"
        else:
            text = explanation.notes[0] if explanation.notes else ""
        case = store.create(int(tx.id), explanation.risk, explanation.decision, text,
                            priority=priority or priority_for(explanation.risk, explanation.decision),
                            source=source, amount=tx.amount, tx_type=tx.type, notes=note)
        return self._result(case, created=True)

    def get_case(self, case_id: str | int) -> CaseResult | None:
        """One case with its history, or None when it does not exist."""
        store = self._case_store()
        case = store.get(case_id)
        return None if case is None else self._result(case)

    def update_case(self, case_id: str | int, *, status: str | None = None,
                    state: str | bool | None = None, note: str | None = None,
                    notes: str | None = None, priority: str | None = None) -> CaseResult:
        """Move a case along: resolution, open/closed, priority and notes.

        `note` appends one note (with a blank line), which is what an analyst or
        an agent adds after checking something; `notes` replaces the whole text,
        which is what the edit form does. A case can only be closed once a
        resolution other than "investigating" is recorded.
        """
        store = self._case_store()
        if note is not None and notes is not None:
            raise ToolError("Pass either note (append) or notes (replace), not both.")
        current = store.get(case_id)
        if current is None:
            raise ToolError(f"Case {case_id!r} does not exist.")

        closed = None if state is None else _parse_state(state)
        if note is not None:
            text = str(note).strip()
            notes = f"{current.notes}\n\n{text}".strip() if current.notes else text
        resulting_status = status if status is not None else current.status
        if closed and not current.closed and resulting_status == "investigating":
            raise ToolError("A case needs a resolution (confirmed_fraud or false_positive) before it can be closed.")

        updated = store.update(current.case_id, status=status, notes=notes,
                               priority=priority, closed=closed)
        return self._result(updated)

    # -- internals ------------------------------------------------------ #
    def _corpus_frame(self, corpus: pd.DataFrame | Iterable | None) -> pd.DataFrame:
        if corpus is None:
            raise ToolError("find_related_transactions needs a corpus: pass corpus=... "
                            "or create the toolkit with one.")
        if isinstance(corpus, pd.DataFrame):
            source = corpus.reset_index(drop=True)
        else:
            rows = [as_transaction(row) for row in corpus]
            source = pd.DataFrame([{
                "id": t.id, "type": t.type, "amount": t.amount,
                "oldbalanceOrg": t.oldbalanceOrg, "oldbalanceDest": t.oldbalanceDest,
                "hour": t.hour, "step": t.step, "nameOrig": t.nameOrig, "nameDest": t.nameDest,
            } for t in rows])

        frame = pd.DataFrame(index=range(len(source)))
        ids = _numeric(source, "id", "txn_id")
        frame["id"] = np.nan if ids is None else ids.to_numpy(dtype=float)

        types = _column(source, "type", "tx_type")
        frame["type"] = (types.fillna("").astype(str).str.strip().str.upper()
                         if types is not None else "")
        for name in ("amount", "oldbalanceOrg", "oldbalanceDest"):
            values = _numeric(source, name)
            frame[name] = 0.0 if values is None else values.fillna(0.0).to_numpy(dtype=float)

        hour = _numeric(source, "hour")
        step = _numeric(source, "step")
        if step is None and hour is None:
            frame["step"] = 1
            frame["hour"] = 0
        elif step is None:
            frame["hour"] = hour.fillna(0).astype(int).to_numpy() % 24
            frame["step"] = frame["hour"] + 1
        else:
            frame["step"] = step.fillna(1).astype(int).clip(lower=1).to_numpy()
            frame["hour"] = ((frame["step"] - 1) % 24 if hour is None
                             else hour.fillna((frame["step"] - 1) % 24).astype(int).to_numpy() % 24)

        for name in ("nameOrig", "nameDest"):
            values = _column(source, name)
            frame[name] = (values.fillna("").astype(str).str.strip()
                           if values is not None else "")
        return frame

    @staticmethod
    def _similarity(frame: pd.DataFrame, tx: Transaction):
        """Similarity score per corpus row, and the signals that produced it."""
        amount = frame["amount"].to_numpy(dtype=float)
        dest = frame["oldbalanceDest"].to_numpy(dtype=float)
        sender = frame["oldbalanceOrg"].to_numpy(dtype=float)
        same_type = frame["type"].to_numpy() == tx.type
        origins = frame["nameOrig"].to_numpy()
        receivers = frame["nameDest"].to_numpy()
        same_orig = (tx.nameOrig != "") & (origins == tx.nameOrig)
        same_dest = (tx.nameDest != "") & (receivers == tx.nameDest)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(amount > 0, np.maximum(amount / tx.amount, tx.amount / amount), np.inf)
        amount_points = np.select([ratio <= 1.01, ratio <= 1.05, ratio <= 1.25], [3.0, 2.0, 1.0], 0.0)

        empty_pair = (dest == 0) & (float(tx.oldbalanceDest) == 0.0)
        target_drains = abs(float(tx.oldbalanceOrg) - tx.amount) <= max(0.01 * tx.amount, 1.0)
        drains = np.abs(sender - amount) <= np.maximum(0.01 * amount, 1.0)
        drain_pair = drains & target_drains

        gap = np.abs(frame["step"].to_numpy(dtype=float) - float(tx.step or (tx.hour + 1)))
        time_points = np.select([gap <= 1, gap <= 3, gap <= 24], [0.75, 0.5, 0.25], 0.0)

        scores = (same_type.astype(float) * 1.0 + same_orig.astype(float) * 4.0
                  + same_dest.astype(float) * 4.0 + amount_points
                  + empty_pair.astype(float) * 0.5 + drain_pair.astype(float) * 1.0 + time_points)
        signals = {"type": same_type, "orig": same_orig, "dest": same_dest,
                   "amount": amount_points, "empty": empty_pair, "drain": drain_pair, "gap": gap}
        return scores, signals

    @staticmethod
    def _relation_text(signals: dict, position: int, tx: Transaction) -> tuple[str, ...]:
        """Why one transaction is in the result list, in plain language."""
        why = []
        if signals["orig"][position]:
            why.append(f"same sender account ({tx.nameOrig})")
        if signals["dest"][position]:
            why.append(f"same receiver account ({tx.nameDest})")
        if signals["type"][position]:
            why.append(f"same type ({tx.type})")
        points = signals["amount"][position]
        if points >= 3:
            why.append("similar amount (within 1%)")
        elif points >= 2:
            why.append("similar amount (within 5%)")
        elif points >= 1:
            why.append("similar amount (within 25%)")
        if signals["empty"][position]:
            why.append("both send into an empty account")
        if signals["drain"][position]:
            why.append("both empty the sender's account")
        gap = signals["gap"][position]
        if gap <= 1:
            why.append("same simulated hour")
        elif gap <= 3:
            why.append("within 3 simulated hours")
        elif gap <= 24:
            why.append("within a day")
        return tuple(why)

    def _score_shortlist(self, rows: pd.DataFrame):
        """Risk and decision of the selected related transactions (one engine call)."""
        n = len(rows)
        if self.engine is None or n == 0:
            return [None] * n, [None] * n
        clean = pd.DataFrame({
            "step": rows["step"].astype(int).clip(lower=1),
            "type": rows["type"].astype(str),
            "amount": rows["amount"].astype(float),
            "oldbalanceOrg": rows["oldbalanceOrg"].astype(float),
            "oldbalanceDest": rows["oldbalanceDest"].astype(float),
        }).reset_index(drop=True)
        risk, _, applicable = score_clean(self.engine, clean, min_explain=101.0)
        policy = self._policy(None)
        risks = [float(r) if ok else None for r, ok in zip(risk, applicable)]
        decisions = [decide_one(r, policy) if ok else None for r, ok in zip(risk, applicable)]
        return risks, decisions

    @staticmethod
    def _frame_transaction(row: pd.Series) -> dict:
        txn_id = row["id"]
        return {
            "id": None if pd.isna(txn_id) else int(txn_id),
            "type": str(row["type"]), "amount": float(row["amount"]),
            "oldbalanceOrg": float(row["oldbalanceOrg"]),
            "oldbalanceDest": float(row["oldbalanceDest"]),
            "hour": int(row["hour"]), "step": int(row["step"]),
            "nameOrig": str(row["nameOrig"]), "nameDest": str(row["nameDest"]),
        }


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def cases_path(root: str | Path = ".") -> Path:
    """Where cases live: $FRAUD_CASES_DB, or data/cases.db under the project root."""
    override = os.environ.get(CASES_DB)
    return Path(override) if override else Path(root) / "data" / "cases.db"


def audit_path(root: str | Path = ".") -> Path:
    """Where the decision log lives: $FRAUD_AUDIT_DB, or data/audit.db."""
    override = os.environ.get(AUDIT_DB)
    return Path(override) if override else Path(root) / "data" / "audit.db"


def open_toolkit(root: str | Path = ".", tag: str | None = None, *,
                 policy: Policy | None = None,
                 corpus: pd.DataFrame | Iterable | None = None,
                 with_stores: bool = True) -> Toolkit:
    """A ready toolkit for a project folder: the preferred trained run plus its stores.

    This is what a script, a notebook or an agent runtime calls. The app builds
    its toolkit from the engines it has already loaded instead (app/common.py),
    so both paths share exactly the same functions.
    """
    root = Path(root)
    runs = discover_runs(root)
    if not runs:
        raise ToolError(f"No trained risk engine under {root / 'results'}. Run train/train.py first.")
    tag = tag or runs[0]
    engine = load_engine(root, tag)
    return Toolkit(engine, policy=policy, model_tag=tag, corpus=corpus,
                   cases=CaseStore(cases_path(root)) if with_stores else None,
                   audit=AuditLog(audit_path(root)) if with_stores else None)
