"""Explanations: which facts about a transaction pushed its risk up or down.

Method (works for any model, no extra library): for each model input, replace
that input with a typical value (the engine's feature_reference) and see how
much the engine's score drops. The drop is that input's contribution:

    contribution = score(transaction) - score(transaction with this input made typical)

Positive contributions push the transaction towards "fraud", negative ones
away from it. Contributions are measured on the stacker's raw (log-odds style)
score, not on the 0-100 risk, because the percentile saturates near 100.

These are explanations of the engine, not proof of fraud. They show what the
engine reacted to.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.engine import RiskEngine

NO_FACTOR = "No single dominant factor"

# Short names for charts and tables.
FEATURE_LABELS = {
    "amount": "Transaction amount",
    "oldbalanceOrg": "Sender balance",
    "oldbalanceDest": "Receiver balance",
    "is_transfer": "Transaction type",
    "hour_of_day": "Time of day",
    "amount_to_balance": "Amount vs. sender balance",
    "dest_balance_zero": "Empty receiving account",
    "newbalanceOrig": "Sender balance after",
    "newbalanceDest": "Receiver balance after",
    "errorBalanceOrig": "Sender balance mismatch",
    "errorBalanceDest": "Receiver balance mismatch",
}


@dataclass
class Reason:
    feature: str
    label: str            # short name, e.g. "Transaction amount"
    text: str             # plain-language fact, e.g. "Amount 453,515 (typical 90,000)"
    contribution: float   # change in the engine's raw score (positive = towards fraud)
    share: float          # share of the total push towards fraud (0 for reasons that lower the risk)


def describe(feature: str, value: float, reference: float) -> str:
    """One plain-language sentence about an input value, next to the typical value."""
    if feature == "amount":
        return f"Amount {value:,.0f} (typical {reference:,.0f})"
    if feature == "oldbalanceOrg":
        return f"Sender balance {value:,.0f} (typical {reference:,.0f})"
    if feature == "oldbalanceDest":
        return f"Receiver balance {value:,.0f} (typical {reference:,.0f})"
    if feature == "dest_balance_zero":
        return "Receiving account was empty" if value >= 0.5 else "Receiving account already held funds"
    if feature == "is_transfer":
        return "Transfer" if value >= 0.5 else "Cash out"
    if feature == "hour_of_day":
        return f"Made at {int(value):02d}:00"
    if feature == "amount_to_balance":
        return "Amount above 1000% of the sender's balance" if value > 10 else f"Amount is {value:.0%} of the sender's balance"
    return f"{FEATURE_LABELS.get(feature, feature)}: {value:,.2f} (typical {reference:,.2f})"


def _check_reference(engine: RiskEngine) -> pd.Series:
    if engine.feature_reference is None:
        raise ValueError("The engine has no feature_reference. Retrain, or load it with artifacts.load_engine.")
    return engine.feature_reference


def contributions(engine: RiskEngine, X: pd.DataFrame) -> np.ndarray:
    """Contribution of every input for every row (rows x inputs)."""
    ref = _check_reference(engine)
    n, k = X.shape
    cols = list(X.columns)
    # Score the originals and all counterfactuals in ONE call. The engine pays a
    # fixed cost per call of roughly 45 ms whatever the batch size, so one call
    # for k+1 batches is several times faster than k+1 separate calls.
    stacked = pd.concat([X] + [X.assign(**{col: ref[col]}) for col in cols], ignore_index=True)
    raw = engine.raw_from_features(stacked)
    base = raw[:n]
    out = np.zeros((n, k))
    for j in range(k):
        out[:, j] = base - raw[(j + 1) * n:(j + 2) * n]
    return out


def explain_matrix(engine: RiskEngine, X: pd.DataFrame, risk: np.ndarray, min_risk: float = 80.0):
    """Main reason for many transactions at once (used by the dashboard).

    Only rows with risk >= min_risk are explained, since those are the only
    ones the policy can flag. Returns (labels, contributions); rows that were
    not explained get an empty label and NaN contributions.
    """
    risk = np.asarray(risk, dtype=float)
    labels = np.full(len(X), "", dtype=object)
    contrib = np.full((len(X), X.shape[1]), np.nan)
    mask = risk >= min_risk
    if mask.any():
        c = contributions(engine, X[mask])
        contrib[mask] = c
        cols = list(X.columns)
        top = np.argmax(c, axis=1)
        best = c[np.arange(len(c)), top]
        labels[mask] = [FEATURE_LABELS.get(cols[i], cols[i]) if b > 0 else NO_FACTOR
                        for i, b in zip(top, best)]
    return labels, contrib


MIN_REASON_SHARE = 0.03     # reasons smaller than this share of the total effect are left out


def explain_one(engine: RiskEngine, X_row: pd.DataFrame, top_k: int = 3):
    """Reasons for one transaction: (raising the risk, lowering the risk).

    Reasons with a negligible effect (under 3% of the total) are left out.
    """
    ref = _check_reference(engine)
    c = contributions(engine, X_row.iloc[[0]])[0]
    positive_total = float(c[c > 0].sum())
    reasons = []
    for j, col in enumerate(X_row.columns):
        value = float(X_row.iloc[0][col])
        reasons.append(Reason(
            feature=col, label=FEATURE_LABELS.get(col, col),
            text=describe(col, value, float(ref[col])),
            contribution=float(c[j]),
            share=float(c[j] / positive_total) if c[j] > 0 and positive_total > 0 else 0.0,
        ))
    total = float(np.abs(c).sum()) or 1.0
    material = [r for r in reasons if abs(r.contribution) >= MIN_REASON_SHARE * total]
    up = sorted((r for r in material if r.contribution > 0), key=lambda r: -r.contribution)[:top_k]
    down = sorted((r for r in material if r.contribution < 0), key=lambda r: r.contribution)[:top_k]
    return up, down