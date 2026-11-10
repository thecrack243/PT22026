"""One entry point for scoring a single transaction (form, API, generator).

    prediction = predict_one(engine, {"type": "TRANSFER", "amount": 250000,
                                      "oldbalanceOrg": 250000, "oldbalanceDest": 0, "hour": 3})

Returns the risk score, the decision under a policy, and the reasons.
The dashboard replay, the live generator and the manual test form all use the
same engine, so a transaction gets the same score wherever it comes from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.decision import APPROVE, Policy, decide_one
from src.engine import RiskEngine
from src.explain import Reason, explain_one
from src.features import KEEP_TYPES
from src.processing import ALL_TYPES, DEFAULT_HOUR, OTHER_TYPES, engine_frame  # noqa: F401  (OTHER_TYPES re-exported)

# The dashboard's default settings, used when no policy is given.
DEFAULT_POLICY = Policy(block_threshold=99.5, review_min_score=90.0, daily_capacity=150)


@dataclass
class Prediction:
    risk: float                       # 0-100, higher = more suspicious
    decision: str                     # approve / review / block
    applicable: bool                  # False when the engine does not cover this transaction type
    reasons_up: list[Reason] = field(default_factory=list)      # what raised the risk
    reasons_down: list[Reason] = field(default_factory=list)    # what lowered it
    notes: list[str] = field(default_factory=list)


def validate_transaction(tx: dict) -> tuple[dict, list[str]]:
    """Clean a raw transaction dict. Returns (clean values, list of error messages)."""
    errors: list[str] = []
    clean: dict = {}

    t = str(tx.get("type", "")).strip().upper().replace(" ", "_")
    if t not in ALL_TYPES:
        errors.append(f"type must be one of {', '.join(ALL_TYPES)}.")
    clean["type"] = t

    for name in ("amount", "oldbalanceOrg", "oldbalanceDest"):
        try:
            value = float(tx[name])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{name} is required and must be a number.")
            continue
        if value != value or value in (float("inf"), float("-inf")) or value < 0:
            errors.append(f"{name} must be a finite number that is zero or more.")
            continue
        clean[name] = value
    if "amount" in clean and clean["amount"] <= 0:
        errors.append("amount must be greater than zero.")

    try:
        hour = int(tx.get("hour", DEFAULT_HOUR))
        if not 0 <= hour <= 23:
            raise ValueError
        clean["hour"] = hour
    except (TypeError, ValueError):
        errors.append("hour must be a whole number from 0 to 23.")
    return clean, errors


def predict_one(engine: RiskEngine, tx: dict, policy: Policy | None = None, top_k: int = 3) -> Prediction:
    """Score one transaction. Raises ValueError listing every problem with the input."""
    clean, errors = validate_transaction(tx)
    if errors:
        raise ValueError(" ".join(errors))
    policy = policy or DEFAULT_POLICY

    if clean["type"] not in KEEP_TYPES:
        return Prediction(risk=0.0, decision=APPROVE, applicable=False,
                          notes=[f"{clean['type']} transactions have no fraud examples in this dataset, so the engine "
                                 "does not score them. This does not mean they are safe in real life."])

    df = engine_frame(pd.DataFrame([{
        "step": clean["hour"] + 1,                 # step 1 is hour 0 of day 1
        "type": clean["type"], "amount": clean["amount"],
        "oldbalanceOrg": clean["oldbalanceOrg"], "oldbalanceDest": clean["oldbalanceDest"],
    }]))
    out = engine.score(df)
    up, down = explain_one(engine, out.features, top_k=top_k)
    risk = float(out.risk[0])
    return Prediction(risk=risk, decision=decide_one(risk, policy), applicable=True,
                      reasons_up=up, reasons_down=down)