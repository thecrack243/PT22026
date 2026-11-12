"""Tests for src/features.py. Run from the project root: python -m pytest"""
import numpy as np
import pandas as pd
import pytest

from src.features import (
    KEEP_TYPES, build_features, feature_columns, load_paysim, log_columns, signed_log1p,
)


def sample_df():
    return pd.DataFrame({
        "step": [1, 25, 50],
        "type": ["TRANSFER", "CASH_OUT", "TRANSFER"],
        "amount": [100.0, 500.0, 40.0],
        "oldbalanceOrg": [100.0, 1000.0, 0.0],
        "oldbalanceDest": [0.0, 50.0, 10.0],
        "newbalanceOrig": [0.0, 500.0, 0.0],
        "newbalanceDest": [100.0, 550.0, 50.0],
    })


def test_default_features_exclude_the_leaky_columns():
    X = build_features(sample_df())
    assert list(X.columns) == feature_columns()
    assert not {"newbalanceOrig", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest"} & set(X.columns)
    assert (X.dtypes == "float32").all()


def test_feature_values():
    X = build_features(sample_df())
    assert list(X["is_transfer"]) == [1, 0, 1]
    assert list(X["hour_of_day"]) == [0, 0, 1]                       # step 1 -> 00h, 25 -> 00h, 50 -> 01h
    assert list(X["dest_balance_zero"]) == [1, 0, 0]
    assert float(X["amount_to_balance"].iloc[0]) == pytest.approx(100 / 101, abs=1e-5)
    assert float(X["amount_to_balance"].iloc[2]) == pytest.approx(40.0)     # zero sender balance does not divide by zero


def test_leakage_experiment_adds_balance_features():
    X = build_features(sample_df(), include_balances=True)
    assert set(feature_columns(True)) == set(X.columns) and len(X.columns) == len(feature_columns(False)) + 4
    assert float(X["errorBalanceOrig"].iloc[0]) == pytest.approx(0.0)       # 0 + 100 - 100


def test_dropping_features_and_unknown_names():
    X = build_features(sample_df(), drop=("amount_to_balance", "oldbalanceOrg"))
    assert "amount_to_balance" not in X.columns and "oldbalanceOrg" not in X.columns
    assert "amount_to_balance" not in log_columns(drop=("amount_to_balance",))
    with pytest.raises(ValueError):
        feature_columns(drop=("not_a_feature",))


def test_missing_columns_are_reported():
    with pytest.raises(ValueError):
        build_features(sample_df().drop(columns=["amount"]))


def test_signed_log_keeps_the_sign():
    out = signed_log1p(np.array([-9.0, 0.0, 9.0]))
    assert float(out[0]) == pytest.approx(-float(out[2])) and float(out[1]) == 0.0


def test_load_paysim_keeps_only_fraud_relevant_types(tmp_path):
    csv = tmp_path / "paysim.csv"
    pd.DataFrame({
        "step": [1, 1, 2, 2], "type": ["PAYMENT", "TRANSFER", "CASH_IN", "CASH_OUT"],
        "amount": [5.0, 6.0, 7.0, 8.0], "nameOrig": "a", "oldbalanceOrg": 1.0, "newbalanceOrig": 0.0,
        "nameDest": "b", "oldbalanceDest": 1.0, "newbalanceDest": 2.0,
        "isFraud": [0, 1, 0, 0], "isFlaggedFraud": 0,
    }).to_csv(csv, index=False)
    df = load_paysim(csv)
    assert set(df["type"]) <= set(KEEP_TYPES) and len(df) == 2
    with pytest.raises(FileNotFoundError):
        load_paysim(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_paysim(bad)