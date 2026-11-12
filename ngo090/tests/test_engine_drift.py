"""Unit tests for src/engine.py helpers and src/drift.py. Run: python -m pytest"""
import numpy as np
import pandas as pd
import pytest

from src.drift import drift_status, psi_vs_uniform
from src.engine import STACK_ORDER, stack_features, to_risk


def test_to_risk_is_a_percentile_of_the_reference():
    ref = np.arange(100.0)
    risk = to_risk(np.array([-1.0, 49.5, 1000.0]), ref)
    assert list(risk) == [0.0, 50.0, 100.0]


def test_stack_features_order_and_logit():
    comp = pd.DataFrame({"isolation_forest": [0.7], "logistic_regression": [0.5], "xgboost": [1.0]})
    out = stack_features(comp)
    assert tuple(out.columns) == STACK_ORDER
    assert out["logistic_regression"].iloc[0] == pytest.approx(0.0)      # logit(0.5)
    assert np.isfinite(out["xgboost"].iloc[0])                           # p=1 is clipped
    assert out["isolation_forest"].iloc[0] == pytest.approx(0.7)


def test_psi_is_zero_for_even_spread_and_large_for_shift():
    even = np.linspace(0, 99.9, 1000)
    assert psi_vs_uniform(even) < 0.01
    assert psi_vs_uniform(np.full(1000, 99.0)) > 1.0


def test_drift_status_thresholds():
    assert drift_status(None) == "warming"
    assert drift_status(0.05) == "stable"
    assert drift_status(0.15) == "watch"
    assert drift_status(0.40) == "drift"