"""Tests for src/processing.py: validation, scoring and decisions for tables of transactions."""
import numpy as np
import pandas as pd
import pytest

from src.decision import APPROVE, BLOCK, REVIEW, Policy
from src.predict import predict_one
from src.processing import MAX_ROWS, process_frame, standardize_columns, template_csv, validate_frame
from tests.helpers import make_engine

ENGINE, X_ALL, Y_ALL = make_engine()
POLICY = Policy(block_threshold=99.0, review_min_score=90.0, daily_capacity=100)


def frame(**overrides):
    base = {"type": ["TRANSFER", "TRANSFER"], "amount": [80_000, 3_000],
            "oldbalanceOrg": [80_000, 60_000], "oldbalanceDest": [0, 40_000], "hour": [3, 14]}
    base.update(overrides)
    return pd.DataFrame(base)


def test_headers_are_case_insensitive_and_extra_columns_ignored():
    df = frame().rename(columns={"type": "TYPE", "oldbalanceOrg": "OldBalanceOrg"})
    df["nameOrig"] = "C123"
    out = standardize_columns(df)
    assert set(out.columns) == {"type", "amount", "oldbalanceOrg", "oldbalanceDest", "hour"}


def test_valid_rows_pass_and_defaults_apply():
    clean, rejected = validate_frame(frame().drop(columns="hour"))
    assert len(clean) == 2 and rejected.empty
    assert set(clean["step"]) == {13}                                  # default hour 12 -> step 13
    clean, _ = validate_frame(frame())
    assert list(clean["step"]) == [4, 15]                              # hour + 1


def test_bad_rows_are_rejected_with_reasons_and_row_numbers():
    df = frame(type=["TRANSFER", "gift", "cash out", "TRANSFER"], amount=[100, 50, -5, "abc"],
               oldbalanceOrg=[10, 10, 10, 10], oldbalanceDest=[0, 0, 0, 0], hour=[5, 5, 5, 30])
    clean, rejected = validate_frame(df)
    assert list(clean["row"]) == [1]
    assert list(rejected["row"]) == [2, 3, 4]
    assert all(rejected["reason"] != "")
    assert "type" in rejected.loc[rejected["row"] == 2, "reason"].iloc[0]


def test_missing_required_column_raises():
    with pytest.raises(ValueError):
        validate_frame(frame().drop(columns="amount"))


def test_labels_are_kept_only_when_valid():
    clean, _ = validate_frame(frame(isFraud=[1, "maybe"]))
    assert clean["isFraud"].iloc[0] == 1 and np.isnan(clean["isFraud"].iloc[1])


def test_batch_matches_single_transaction_scores():
    batch = process_frame(ENGINE, frame(), POLICY)
    for i, tx in enumerate(frame().to_dict("records")):
        single = predict_one(ENGINE, tx, POLICY)
        assert float(batch.rows["risk"].iloc[i]) == pytest.approx(single.risk, abs=1e-6)
        assert batch.rows["decision"].iloc[i] == single.decision


def test_types_without_fraud_are_approved_unscored():
    batch = process_frame(ENGINE, frame(type=["PAYMENT", "CASH_IN"]), POLICY)
    assert list(batch.rows["decision"]) == [APPROVE, APPROVE] and not batch.rows["applicable"].any()
    assert (batch.rows["risk"] == 0).all()


def test_decisions_follow_the_policy_and_risky_rows_get_reasons():
    big = pd.concat([frame()] * 300, ignore_index=True)
    batch = process_frame(ENGINE, big, Policy(block_threshold=100.0, review_min_score=0.0, daily_capacity=100))
    assert set(batch.rows["decision"]) <= {APPROVE, REVIEW}            # block threshold 100 ... never blocks
    no_review = process_frame(ENGINE, big, Policy(block_threshold=float("inf"), review_min_score=0.0, daily_capacity=0))
    assert set(no_review.rows["decision"]) == {APPROVE}
    high = batch.rows["risk"] >= 80
    assert (batch.rows.loc[high, "reason"] != "").all()


def test_size_limit_and_template():
    with pytest.raises(ValueError):
        process_frame(ENGINE, pd.DataFrame({"type": ["TRANSFER"] * (MAX_ROWS + 1)}), POLICY)
    import io
    template = pd.read_csv(io.StringIO(template_csv()))
    batch = process_frame(ENGINE, template, POLICY)
    assert len(batch.rows) == len(template) and batch.rejected.empty