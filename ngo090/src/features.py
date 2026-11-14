"""Shared data loading and feature engineering for FRAUDINTEL.

This module is imported by BOTH train.py and the Streamlit app, so features are
computed identically at training time and at scoring time (no training/serving
skew).

Leakage note
------------
PaySim's post-transaction balances (newbalanceOrig / newbalanceDest) reveal
fraud almost directly (a fraudster empties the account, leaving a zero balance),
but a live system would not know them at decision time. They are therefore
EXCLUDED by default and only added when include_balances=True, which is used
for the leakage experiment in the report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Fraud in PaySim only occurs in these two transaction types.
KEEP_TYPES = ("TRANSFER", "CASH_OUT")

LOAD_COLUMNS = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "isFraud",
]

# Small dtypes keep memory low on a laptop (the full CSV has ~6.4M rows).
LOAD_DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float32",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
}

# Columns known at decision time.
_BASE_FEATURES = [
    "amount", "oldbalanceOrg", "oldbalanceDest",
    "is_transfer", "hour_of_day", "amount_to_balance", "dest_balance_zero",
]
# Columns that leak the outcome (only used for the leakage experiment).
_LEAKY_FEATURES = [
    "newbalanceOrig", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest",
]

# Heavy-tailed columns that get a signed log transform for logistic regression.
_LOG_BASE = ["amount", "oldbalanceOrg", "oldbalanceDest", "amount_to_balance"]


def feature_columns(include_balances: bool = False, drop: tuple[str, ...] = ()) -> list[str]:
    """Ordered list of model input columns.

    `drop` removes named features (used for ablation experiments, e.g. dropping
    'amount_to_balance'). Unknown names raise ValueError.
    """
    cols = _BASE_FEATURES + (_LEAKY_FEATURES if include_balances else [])
    unknown = set(drop) - set(cols)
    if unknown:
        raise ValueError(
            f"Unknown feature(s) to drop: {sorted(unknown)}. Available: {cols}"
        )
    return [c for c in cols if c not in drop]


def log_columns(include_balances: bool = False, drop: tuple[str, ...] = ()) -> list[str]:
    """Columns that should be log-transformed before scaling (for LR)."""
    cols = list(_LOG_BASE)
    if include_balances:
        cols += _LEAKY_FEATURES
    return [c for c in cols if c not in drop]


def signed_log1p(x):
    """sign(x) * log(1 + |x|). Handles the negative 'error' features."""
    return np.sign(x) * np.log1p(np.abs(x))


def load_paysim(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load PaySim, keeping only the needed columns and fraud-relevant types."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download PaySim from Kaggle "
            "(dataset 'ealaxi/paysim1') and pass its location with --data-path."
        )
    try:
        df = pd.read_csv(path, usecols=LOAD_COLUMNS, dtype=LOAD_DTYPES, nrows=nrows)
    except ValueError as exc:  # missing columns / wrong format
        raise ValueError(
            f"'{path}' does not look like the PaySim CSV (expected columns: "
            f"{', '.join(LOAD_COLUMNS)}). Original error: {exc}"
        ) from exc

    df = df[df["type"].isin(KEEP_TYPES)].reset_index(drop=True)
    df["type"] = df["type"].astype(str)
    return df


def build_features(df: pd.DataFrame, include_balances: bool = False,
                   drop: tuple[str, ...] = ()) -> pd.DataFrame:
    """Turn raw transactions into the model's feature matrix (float32 DataFrame)."""
    needed = {"step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"}
    if include_balances:
        needed |= {"newbalanceOrig", "newbalanceDest"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = pd.DataFrame(index=df.index)
    out["amount"] = df["amount"]
    out["oldbalanceOrg"] = df["oldbalanceOrg"]
    out["oldbalanceDest"] = df["oldbalanceDest"]
    out["is_transfer"] = (df["type"] == "TRANSFER").astype("int8")
    out["hour_of_day"] = ((df["step"] - 1) % 24).astype("int8")
    out["amount_to_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    out["dest_balance_zero"] = (df["oldbalanceDest"] == 0).astype("int8")

    if include_balances:
        out["newbalanceOrig"] = df["newbalanceOrig"]
        out["newbalanceDest"] = df["newbalanceDest"]
        # Accounting mismatches: nonzero when money does not add up.
        out["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
        out["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    return out[feature_columns(include_balances, drop)].astype("float32")
