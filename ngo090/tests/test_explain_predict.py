"""Tests for src/explain.py and src/predict.py using a small engine built from synthetic data.

Uses only scikit-learn models (a gradient-boosting classifier stands in for XGBoost),
so the tests run without XGBoost installed. Run from the project root: python -m pytest
"""
import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_engine
from src.decision import APPROVE, BLOCK, REVIEW, Policy, decide_one
from src.explain import NO_FACTOR, contributions, explain_matrix, explain_one
from src.predict import predict_one, validate_transaction


ENGINE, X_ALL, Y_ALL = make_engine()
FRAUD_TX = {"type": "TRANSFER", "amount": 80_000, "oldbalanceOrg": 80_000, "oldbalanceDest": 0, "hour": 3}
NORMAL_TX = {"type": "TRANSFER", "amount": 3_000, "oldbalanceOrg": 60_000, "oldbalanceDest": 40_000, "hour": 14}


def test_a_neutral_input_has_zero_contribution():
    X = X_ALL.iloc[[0]].copy()
    X["amount"] = ENGINE.feature_reference["amount"]          # already typical
    c = contributions(ENGINE, X)[0]
    assert c[list(X.columns).index("amount")] == pytest.approx(0.0, abs=1e-9)


def test_fraud_like_transaction_scores_higher_and_has_reasons():
    fraud, normal = predict_one(ENGINE, FRAUD_TX), predict_one(ENGINE, NORMAL_TX)
    assert fraud.applicable and fraud.risk > normal.risk
    assert fraud.reasons_up and fraud.reasons_up[0].contribution > 0
    contributions_sorted = [r.contribution for r in fraud.reasons_up]
    assert contributions_sorted == sorted(contributions_sorted, reverse=True)
    assert sum(r.share for r in fraud.reasons_up) <= 1.0 + 1e-9
    assert all(r.text for r in fraud.reasons_up)


def test_explain_matrix_labels_only_high_risk_rows():
    out = ENGINE.score_features(X_ALL.iloc[:300])
    labels, contrib = explain_matrix(ENGINE, X_ALL.iloc[:300], out.risk, min_risk=80.0)
    high = out.risk >= 80.0
    assert high.any() and (~high).any()
    assert all(labels[high] != "") and all(labels[~high] == "")
    assert np.isnan(contrib[~high]).all() and not np.isnan(contrib[high]).any()
    assert all((l == NO_FACTOR) or isinstance(l, str) for l in labels[high])


def test_explain_one_requires_a_reference():
    engine, _, _ = make_engine(seed=1, n=2000)
    engine.feature_reference = None
    with pytest.raises(ValueError):
        explain_one(engine, X_ALL.iloc[[0]])


def test_validation_reports_every_problem():
    clean, errors = validate_transaction({"type": "gift", "amount": -5, "oldbalanceOrg": "abc", "hour": 30})
    assert len(errors) >= 4
    clean, errors = validate_transaction({"type": "cash out", "amount": "100", "oldbalanceOrg": 5, "oldbalanceDest": 0})
    assert not errors and clean["type"] == "CASH_OUT" and clean["hour"] == 12
    with pytest.raises(ValueError):
        predict_one(ENGINE, {"type": "TRANSFER"})


def test_types_without_fraud_are_not_scored():
    p = predict_one(ENGINE, {**NORMAL_TX, "type": "PAYMENT"})
    assert not p.applicable and p.risk == 0.0 and p.decision == APPROVE and p.notes


def test_decide_one_follows_the_policy():
    policy = Policy(block_threshold=99.0, review_min_score=90.0, daily_capacity=100)
    assert [decide_one(r, policy) for r in (50.0, 95.0, 99.5)] == [APPROVE, REVIEW, BLOCK]
    assert decide_one(95.0, Policy(block_threshold=99.0, review_min_score=90.0, daily_capacity=0)) == APPROVE