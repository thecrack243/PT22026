"""Unit tests for src/decision.py. Run from the project root: python -m pytest"""
import numpy as np
import pandas as pd
import pytest

from src.decision import (
    APPROVE, BLOCK, REVIEW, CostConfig, Policy,
    decide, evaluate, threshold_from_top_fraction,
)


def make_df(steps, amounts=None, fraud=None):
    n = len(steps)
    return pd.DataFrame({
        "step": steps,
        "amount": amounts if amounts is not None else [100.0] * n,
        "isFraud": fraud if fraud is not None else [0] * n,
    })


def test_block_threshold_and_no_capacity():
    df = make_df([1, 1, 1])
    action = decide(df, [0.9, 0.5, 0.1], Policy(block_threshold=0.8, daily_capacity=0))
    assert list(action) == [BLOCK, APPROVE, APPROVE]


def test_daily_topk_reviews_exactly_k_best_per_day():
    df = make_df([1, 1, 1, 30, 30, 30])           # day 0 and day 1
    scores = [0.2, 0.9, 0.5, 0.4, 0.1, 0.8]
    action = decide(df, scores, Policy(daily_capacity=1, capacity_mode="daily_topk"))
    assert list(action) == [APPROVE, REVIEW, APPROVE, APPROVE, APPROVE, REVIEW]


def test_rolling_budget_is_one_review_per_hour():
    df = make_df([1, 1, 1, 2, 2])
    scores = [0.2, 0.9, 0.5, 0.7, 0.1]
    action = decide(df, scores, Policy(daily_capacity=24, capacity_mode="rolling"))
    assert list(action) == [APPROVE, REVIEW, APPROVE, REVIEW, APPROVE]


def test_review_min_score_leaves_capacity_unused():
    df = make_df([1, 1])
    action = decide(df, [0.9, 0.3], Policy(daily_capacity=48, review_min_score=0.5))
    assert list(action) == [REVIEW, APPROVE]


def test_evaluate_money_totals():
    df = make_df([1] * 6, amounts=[1000, 500, 200, 300, 300, 300], fraud=[1, 1, 1, 0, 0, 0])
    action = pd.Series([BLOCK, REVIEW, APPROVE, BLOCK, REVIEW, APPROVE])
    costs = CostConfig(review_cost=10, friction_cost=50, review_catch_rate=1.0)
    m = evaluate(df, action, costs)
    assert m["fraud_amount_saved"] == 1500
    assert m["fraud_amount_lost"] == 200
    assert m["net_benefit"] == 1500 - 20 - 50          # 2 reviews, 1 wrongly blocked
    assert m["baseline_cost"] - m["total_cost"] == m["net_benefit"]
    assert (m["fraud_caught"], m["fraud_missed"], m["false_alarms"]) == (2, 1, 2)


def test_input_validation():
    df = make_df([1, 1])
    with pytest.raises(ValueError):
        decide(df, [0.1], Policy())                      # wrong length
    with pytest.raises(ValueError):
        decide(df, [0.1, np.nan], Policy())              # NaN score
    with pytest.raises(ValueError):
        Policy(capacity_mode="weekly")
    with pytest.raises(ValueError):
        CostConfig(review_catch_rate=1.5)
    with pytest.raises(ValueError):
        decide(df.drop(columns="amount"), [0.1, 0.2], Policy())


def test_threshold_from_top_fraction():
    baseline = np.arange(100)
    assert threshold_from_top_fraction(baseline, 0.10) == pytest.approx(89.1)