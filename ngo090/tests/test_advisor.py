"""Tests for the presets and the policy advisor in src/decision.py."""
import numpy as np
import pandas as pd

from src.decision import (
    CAPACITY_CANDIDATES, PRESETS, CostConfig, Policy, recommend_policy,
)


def test_presets_fit_the_dashboard_sliders():
    for name, p in PRESETS.items():
        assert 90.0 <= p.block_threshold <= 100.0, name
        assert 80 <= p.review_min_score <= 99, name
        assert 0 <= p.daily_capacity <= 1000 and p.daily_capacity % 10 == 0, name
    assert PRESETS["Strict"].daily_capacity > PRESETS["Balanced"].daily_capacity > PRESETS["Light touch"].daily_capacity


def make_replay(n=4000, seed=0):
    """Fraud scores around 99.8, legitimate ones spread below 99."""
    rng = np.random.default_rng(seed)
    fraud = rng.random(n) < 0.02
    risk = np.where(fraud, rng.uniform(99.6, 100.0, n), rng.uniform(0, 99.0, n))
    df = pd.DataFrame({"step": np.sort(rng.integers(1, 73, n)), "amount": np.full(n, 10_000.0),
                       "isFraud": fraud.astype(int)})
    return df, risk


def test_advisor_prefers_blocking_only_the_clearly_risky_when_friction_is_costly():
    df, risk = make_replay()
    costs = CostConfig(review_cost=100.0, friction_cost=50_000.0)
    rec = recommend_policy(df, risk, costs, Policy(block_threshold=90.0, review_min_score=90.0, daily_capacity=100))
    assert rec.best_block >= 99.5            # blocking at 90 would block many honest customers
    assert rec.best_net >= rec.current_net
    assert rec.best_net == max(rec.capacity_net)
    assert rec.best_capacity in CAPACITY_CANDIDATES


def test_advisor_never_blocks_when_blocking_is_pointless():
    df, risk = make_replay()
    costs = CostConfig(review_cost=0.0, friction_cost=1e9)         # any wrong block is ruinous
    rec = recommend_policy(df, risk, costs, Policy(block_threshold=90.0, review_min_score=99.0, daily_capacity=1000))
    assert rec.best_block >= 99.9


def test_advisor_reports_the_current_policy_net_benefit():
    df, risk = make_replay()
    pol = Policy(block_threshold=99.5, review_min_score=90.0, daily_capacity=150)
    rec = recommend_policy(df, risk, CostConfig(), pol)
    from src.decision import decide, evaluate
    assert rec.current_net == evaluate(df, decide(df, risk, pol), CostConfig())["net_benefit"]