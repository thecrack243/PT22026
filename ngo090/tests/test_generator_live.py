"""Tests for src/generator.py and src/live.py (the live simulation)."""
import numpy as np
import pandas as pd
import pytest

import src.live as live
from src.decision import APPROVE, BLOCK, REVIEW, CostConfig, Policy
from src.generator import PATTERN_PAYSIM, PATTERN_SMALL, GeneratorConfig, TransactionGenerator
from src.live import LiveSettings, LiveStream, clock_label, minute_label
from src.monitor import compute_view
from src.processing import validate_frame
from tests.helpers import make_engine

ENGINE, _, _ = make_engine()
NEVER = Policy(block_threshold=float("inf"), review_min_score=100.0, daily_capacity=0)


def test_generator_volume_fraud_rate_and_columns():
    gen = TransactionGenerator(seed=1)
    df = gen.generate(0, 24 * 60, GeneratorConfig(volume_per_hour=300, fraud_rate=0.05))         # one full day
    assert 6400 < len(df) < 8000                                     # about 7200 (busier by day than by night)
    assert 0.03 < df["isFraud"].mean() < 0.075
    assert df["id"].is_unique and df["id"].is_monotonic_increasing
    assert set(df["type"]) <= {"PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"}
    clean, rejected = validate_frame(df)                              # the generator output is valid input
    assert rejected.empty and len(clean) == len(df)


def test_fraud_follows_the_chosen_pattern():
    gen = TransactionGenerator(seed=2)
    base = gen.generate(0, 3000, GeneratorConfig(volume_per_hour=300, fraud_rate=0.1, new_pattern=False))
    fraud = base[base["isFraud"] == 1]
    assert set(fraud["type"]) <= {"TRANSFER", "CASH_OUT"} and set(fraud["pattern"]) == {PATTERN_PAYSIM}
    assert (fraud["oldbalanceOrg"] >= fraud["amount"]).all()
    mixed = gen.generate(0, 3000, GeneratorConfig(volume_per_hour=300, fraud_rate=0.1, new_pattern=True))
    small = mixed[mixed["pattern"] == PATTERN_SMALL]
    assert len(small) > 0 and set(small["type"]) == {"TRANSFER"} and (small["amount"] <= 20_000).all()
    assert (small["oldbalanceDest"] > 0).all()                             # established receivers: the engine misses these
    assert (base[base["isFraud"] == 0]["pattern"] == "").all()


def test_generator_is_reproducible_and_handles_empty_ranges():
    a = TransactionGenerator(seed=5).generate(0, 200, GeneratorConfig())
    b = TransactionGenerator(seed=5).generate(0, 200, GeneratorConfig())
    pd.testing.assert_frame_equal(a, b)
    assert TransactionGenerator(seed=5).generate(10, 10, GeneratorConfig()).empty
    with pytest.raises(ValueError):
        GeneratorConfig(fraud_rate=2.0)


def test_stream_processes_what_the_simulated_clock_covers():
    s = LiveStream(ENGINE, seed=3, now=0.0)
    info = s.advance(NEVER, now=30.0)                                 # 30 s x 2 min/s = 60 simulated minutes
    assert s.minute == live.SIM_START_MINUTE + 60 and 150 < info["new"] < 500
    assert len(s.rows) == info["new"]
    assert {"risk", "reason", "decision", "proc_ms", "isFraud"} <= set(s.rows.columns)
    assert s.advance(NEVER, now=30.0)["new"] == 0                     # no time passed, nothing new
    assert s.avg_ms > 0


def test_decisions_follow_the_policy_and_unscored_types_are_approved():
    s = LiveStream(ENGINE, seed=4, now=0.0)
    s.advance(Policy(block_threshold=0.0, review_min_score=0.0, daily_capacity=0), now=60.0)
    scored = s.rows[s.rows["applicable"]]
    assert (scored["decision"] == BLOCK).all()
    assert (s.rows[~s.rows["applicable"]]["decision"] == APPROVE).all()
    s2 = LiveStream(ENGINE, seed=4, now=0.0)
    s2.advance(NEVER, now=60.0)
    assert (s2.rows["decision"] == APPROVE).all()


def test_review_capacity_is_a_budget_that_grows_with_time():
    s = LiveStream(ENGINE, seed=6, now=0.0)
    policy = Policy(block_threshold=float("inf"), review_min_score=0.0, daily_capacity=24)      # 1 review per hour
    s.advance(policy, now=300.0)                                                                 # 10 simulated hours
    reviews = int((s.rows["decision"] == REVIEW).sum())
    assert 8 <= reviews <= 11                                                                     # about one per hour


def test_a_policy_change_only_affects_later_transactions():
    s = LiveStream(ENGINE, seed=7, now=0.0)
    s.advance(NEVER, now=30.0)
    before = s.rows["decision"].copy()
    s.advance(Policy(block_threshold=0.0, review_min_score=0.0, daily_capacity=0), now=60.0)
    assert (s.rows["decision"].iloc[:len(before)] == before).all()                                # old decisions untouched
    assert (s.rows["decision"].iloc[len(before):][s.rows["applicable"].iloc[len(before):]] == BLOCK).all()


def test_long_absence_is_capped_and_runs_are_reproducible():
    s = LiveStream(ENGINE, seed=8, now=0.0, settings=LiveSettings(volume_per_hour=60))
    info = s.advance(NEVER, now=100_000.0)                             # 27 hours away from the screen
    assert info["caught_up_minutes"] == live.MAX_CATCHUP_MINUTES       # at most one simulated day passes
    assert s.minute == live.SIM_START_MINUTE + live.MAX_CATCHUP_MINUTES
    a = LiveStream(ENGINE, seed=9, now=0.0); b = LiveStream(ENGINE, seed=9, now=0.0)
    for t in (10.0, 25.0, 40.0):
        a.advance(NEVER, now=t); b.advance(NEVER, now=t)
    cols = ["id", "amount", "risk", "decision"]
    pd.testing.assert_frame_equal(a.rows[cols], b.rows[cols])


def test_view_has_the_same_structure_as_the_replay_view_and_handles_an_empty_stream():
    s = LiveStream(ENGINE, seed=10, now=0.0)
    empty = s.view(NEVER, CostConfig())
    assert empty["metrics"]["transactions"] == 0
    s.advance(Policy(), now=120.0)
    v = s.view(Policy(), CostConfig())
    assert v["metrics"]["transactions"] == len(s.rows) and len(v["hours"]) >= 1
    rows = s.rows.sort_values("step")
    steps = np.sort(rows["step"].unique())
    ref = compute_view(rows, rows["risk"].to_numpy(), rows["reason"].to_numpy(), steps, len(steps), Policy(), CostConfig())
    assert set(v) == set(ref)


def test_lookup_latest_alerts_and_trimming(monkeypatch):
    s = LiveStream(ENGINE, seed=11, now=0.0)
    s.advance(Policy(block_threshold=0.0, review_min_score=0.0, daily_capacity=0), now=60.0)
    assert len(s.latest(5)) == 5 and s.latest(5)["id"].is_monotonic_decreasing
    alerts = s.alerts(3)
    assert len(alerts) == 3 and (alerts["decision"] != APPROVE).all()
    assert s.get(int(alerts["id"].iloc[0]))["id"] == alerts["id"].iloc[0] and s.get(10**9) is None
    monkeypatch.setattr(live, "MAX_ROWS", 300)
    s.advance(NEVER, now=120.0)
    assert len(s.rows) <= 300


def test_time_labels():
    assert minute_label(live.SIM_START_MINUTE) == "D1 08:00"
    assert minute_label(1440 + 14 * 60 + 5) == "D2 14:05"
    assert clock_label(1440 + 14 * 60 + 5) == "Day 2 · 14:05"


# --- drift in the live stream ---
def test_psi_vs_baseline():
    from src.drift import psi_vs_baseline
    rng = np.random.default_rng(0)
    base = rng.normal(50, 10, 2000)
    assert psi_vs_baseline(base, rng.normal(50, 10, 2000)) < 0.05
    assert psi_vs_baseline(base, rng.normal(70, 10, 2000)) > 0.25
    assert psi_vs_baseline(np.full(100, 5.0), base) == 0.0                 # degenerate baseline


def test_market_shift_makes_amounts_larger_and_receivers_new():
    gen_a, gen_b = TransactionGenerator(seed=3), TransactionGenerator(seed=3)
    normal = gen_a.generate(0, 600, GeneratorConfig(volume_per_hour=300))
    shifted = gen_b.generate(0, 600, GeneratorConfig(volume_per_hour=300, market_shift=True))
    assert shifted["amount"].mean() > 8 * normal["amount"].mean()
    assert (shifted[shifted["type"] != "PAYMENT"]["oldbalanceDest"] == 0).all()


def test_drift_monitor_is_quiet_for_stable_traffic_and_reacts_to_a_shift():
    s = LiveStream(ENGINE, seed=12, now=0.0, settings=LiveSettings(volume_per_hour=600, fraud_rate=0.02))
    assert s.view(NEVER, CostConfig())["drift"] == "warming"
    for t in range(1, 81):                                              # 80 s x 2 = 160 simulated minutes
        s.advance(NEVER, now=float(t))
    for t in range(81, 121):                                            # 6 more hours of the same traffic
        s.advance(NEVER, now=float(t) * 3)
    assert s.view(NEVER, CostConfig())["psi"] is not None                  # both 3-hour windows are populated
    stable = s.view(NEVER, CostConfig())
    assert stable["drift"] in ("stable", "watch") and stable["psi"] is not None and stable["psi"] < 0.25
    shifted = LiveSettings(volume_per_hour=600, fraud_rate=0.02, market_shift=True)
    for t in range(121, 241):
        s.advance(NEVER, now=float(t) * 3, settings=shifted)
    assert s.view(NEVER, CostConfig())["drift"] == "drift"