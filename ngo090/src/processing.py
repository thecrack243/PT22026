"""Shared transaction processing: validate, score, decide and explain.

Every way of getting transactions into the system (the manual test form, an
uploaded file, and later the live generator) goes through the functions here, so
a transaction gets the same score wherever it comes from.

Accepted columns (header names are not case sensitive; extra columns are ignored):
    type            TRANSFER, CASH_OUT, PAYMENT, CASH_IN or DEBIT        (required)
    amount          greater than zero                                    (required)
    oldbalanceOrg   sender balance before, zero or more                  (required)
    oldbalanceDest  receiver balance before, zero or more                (required)
    step or hour    simulated hour (step 1 = day 1, 00:00) or 0-23       (optional, default hour 12)
    isFraud         0 or 1 if the true label is known                    (optional, for audits)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.decision import APPROVE, BLOCK, REVIEW, Policy
from src.engine import RiskEngine
from src.explain import explain_matrix
from src.features import KEEP_TYPES

# Types that exist in PaySim but never contain fraud there, so the engine was not trained on them.
OTHER_TYPES = ("PAYMENT", "CASH_IN", "DEBIT")
ALL_TYPES = KEEP_TYPES + OTHER_TYPES
DEFAULT_HOUR = 12
MAX_ROWS = 20_000
REQUIRED = ("type", "amount", "oldbalanceOrg", "oldbalanceDest")
_ALIASES = {c.lower(): c for c in (*REQUIRED, "step", "hour", "isFraud")}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename headers to the standard names (case-insensitive) and drop everything else."""
    rename = {c: _ALIASES[str(c).strip().lower()] for c in df.columns if str(c).strip().lower() in _ALIASES}
    return df[list(rename)].rename(columns=rename)


def engine_frame(clean: pd.DataFrame) -> pd.DataFrame:
    """The table the engine expects. Post-transaction balances are derived; they are
    only used by the leakage experiment, never by the recommended models."""
    out = clean[["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"]].copy()
    out["newbalanceOrig"] = (out["oldbalanceOrg"] - out["amount"]).clip(lower=0.0)
    out["newbalanceDest"] = out["oldbalanceDest"] + out["amount"]
    return out


def validate_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check every row. Returns (clean rows, rejected rows with a reason).

    Raises ValueError when a required column is missing altogether.
    """
    df = standardize_columns(df)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}.")

    work = pd.DataFrame({"row": np.arange(1, len(df) + 1)})
    work["type"] = df["type"].astype(str).str.strip().str.upper().str.replace(" ", "_", regex=False).to_numpy()
    for name in ("amount", "oldbalanceOrg", "oldbalanceDest"):
        work[name] = pd.to_numeric(df[name], errors="coerce").to_numpy()

    if "step" in df.columns:
        work["step"] = pd.to_numeric(df["step"], errors="coerce").to_numpy()
    elif "hour" in df.columns:
        work["step"] = pd.to_numeric(df["hour"], errors="coerce").to_numpy() + 1
    else:
        work["step"] = DEFAULT_HOUR + 1
    if "isFraud" in df.columns:
        label = pd.to_numeric(df["isFraud"], errors="coerce").to_numpy()
        work["isFraud"] = np.where(np.isin(label, (0, 1)), label, np.nan)

    reasons = pd.Series([""] * len(work), dtype=object)

    def flag(mask, text):
        mask = np.asarray(mask, dtype=bool)
        reasons[mask & (reasons == "")] = text

    flag(~work["type"].isin(ALL_TYPES), f"type must be one of {', '.join(ALL_TYPES)}")
    for name in ("amount", "oldbalanceOrg", "oldbalanceDest"):
        v = work[name]
        flag(v.isna() | np.isinf(v), f"{name} is missing or not a number")
        flag(v < 0, f"{name} must be zero or more")
    flag(work["amount"] <= 0, "amount must be greater than zero")
    s = work["step"]
    hour_given = "step" not in df.columns and "hour" in df.columns
    if hour_given:
        flag(s.isna() | (s < 1) | (s > 24) | (s != np.floor(s.fillna(0))), "hour must be a whole number from 0 to 23")
    else:
        flag(s.isna() | (s < 1) | (s != np.floor(s.fillna(0))), "step must be a whole number of 1 or more")

    bad = (reasons != "").to_numpy()
    clean = work[~bad].reset_index(drop=True)
    clean["step"] = clean["step"].astype(int)
    rejected = pd.DataFrame({"row": work["row"][bad].to_numpy(), "reason": reasons[bad].to_numpy()})
    return clean, rejected


def score_clean(engine: RiskEngine, clean: pd.DataFrame, min_explain: float = 80.0):
    """Risk and main reason for validated transactions.

    Returns (risk, reason, applicable). Types without fraud examples in the
    dataset are not scored: risk 0 and applicable False.
    """
    n = len(clean)
    risk = np.zeros(n)
    reason = np.full(n, "", dtype=object)
    applicable = clean["type"].isin(KEEP_TYPES).to_numpy()
    if applicable.any():
        out = engine.score(engine_frame(clean[applicable]))
        risk[applicable] = out.risk
        labels, _ = explain_matrix(engine, out.features, out.risk, min_risk=min_explain)
        reason[applicable] = labels
    return risk, reason, applicable


@dataclass
class ProcessedBatch:
    rows: pd.DataFrame        # scored transactions: risk, decision, reason, applicable
    rejected: pd.DataFrame    # rows that failed validation, with the reason
    n_input: int


def process_frame(engine: RiskEngine, df: pd.DataFrame, policy: Policy,
                  min_explain: float = 80.0, max_rows: int = MAX_ROWS) -> ProcessedBatch:
    """Validate, score, decide and explain a table of transactions.

    Decisions are made one transaction at a time, so the policy's daily review
    capacity is not applied (it needs a stream of transactions over time).
    """
    if len(df) > max_rows:
        raise ValueError(f"The file has {len(df):,} rows; the limit is {max_rows:,}. Split it into smaller files.")
    clean, rejected = validate_frame(df)

    n = len(clean)
    risk, reason, applicable = score_clean(engine, clean, min_explain)

    block = applicable & (risk >= policy.block_threshold)
    review = applicable & ~block & (risk >= policy.review_min_score) & (policy.daily_capacity > 0)
    decision = np.full(n, APPROVE, dtype=object)
    decision[review], decision[block] = REVIEW, BLOCK

    rows = clean.copy()
    rows["risk"], rows["decision"], rows["reason"], rows["applicable"] = risk, decision, reason, applicable
    return ProcessedBatch(rows=rows, rejected=rejected, n_input=len(df))


def template_csv() -> str:
    """A small example file users can download and fill in."""
    return (
        "type,amount,oldbalanceOrg,oldbalanceDest,hour\n"
        "TRANSFER,8000,120000,60000,14\n"
        "TRANSFER,250000,250000,0,3\n"
        "CASH_OUT,181000,181000,0,2\n"
        "PAYMENT,120,5400,0,11\n"
    )


def summarize_batch(batch: ProcessedBatch, costs) -> dict:
    """Counts and, when the file has true labels, detection quality and money impact."""
    from src.decision import evaluate  # local import keeps this module light

    rows = batch.rows
    flagged = (rows["decision"] != APPROVE).to_numpy()
    summary = {
        "read": batch.n_input, "scored": int(rows["applicable"].sum()), "rejected": len(batch.rejected),
        "alerts": int(flagged.sum()), "blocked": int((rows["decision"] == BLOCK).sum()),
        "reviewed": int((rows["decision"] == REVIEW).sum()),
        "max_risk": float(rows["risk"].max()) if len(rows) else 0.0, "labelled": False,
    }
    if "isFraud" in rows.columns and rows["isFraud"].notna().any():
        lab = rows[rows["isFraud"].notna()].copy()
        lab["isFraud"] = lab["isFraud"].astype(int)
        lab_flagged = (lab["decision"] != APPROVE).to_numpy()
        fraud = (lab["isFraud"] == 1).to_numpy()
        caught = int((fraud & lab_flagged).sum())
        summary.update(
            labelled=True, fraud_total=int(fraud.sum()), fraud_caught=caught,
            recall=caught / fraud.sum() if fraud.sum() else 0.0,
            precision=caught / lab_flagged.sum() if lab_flagged.sum() else 0.0,
            net_benefit=evaluate(lab, lab["decision"], costs)["net_benefit"],
        )
    return summary