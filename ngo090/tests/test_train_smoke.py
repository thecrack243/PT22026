"""End-to-end check: train.py runs on a tiny synthetic PaySim file and writes every artifact
that the app needs. Skipped when XGBoost is not installed."""
import json
import runpy
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")


def make_csv(path, n=30_000, seed=0):
    rng = np.random.default_rng(seed)
    typ = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN"], n)
    bal = rng.lognormal(10, 1, n)
    fraud = np.isin(typ, ["TRANSFER", "CASH_OUT"]) & (rng.random(n) < 0.03)
    amount = np.where(fraud, bal, rng.lognormal(9, 1, n))
    pd.DataFrame({
        "step": rng.integers(1, 121, n), "type": typ, "amount": amount, "nameOrig": "a",
        "oldbalanceOrg": bal, "newbalanceOrig": np.where(fraud, 0.0, bal), "nameDest": "b",
        "oldbalanceDest": np.where(fraud, 0.0, rng.lognormal(10, 1, n)), "newbalanceDest": 1.0,
        "isFraud": fraud.astype(int), "isFlaggedFraud": 0,
    }).to_csv(path, index=False)


def test_training_writes_everything_the_app_needs(tmp_path, monkeypatch):
    csv = tmp_path / "paysim.csv"
    make_csv(csv)
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--data-path", str(csv), "--output-dir", str(tmp_path / "results"),
        "--models-dir", str(tmp_path / "models"), "--split-step", "80", "--val-steps", "30",
        "--daily-capacity", "20", "--replay-seed", "7", "--drop-features", "amount_to_balance", "oldbalanceOrg"])
    runpy.run_path("train/train.py", run_name="__main__")

    tag = "leakage_safe_no_amount_to_balance_oldbalanceOrg"
    results, models = tmp_path / "results" / tag, tmp_path / "models" / tag
    for name in ("results.csv", "replay_sample.csv.gz", "baseline_scores.npz", "meta.json"):
        assert (results / name).exists(), name
    for name in ("logistic_regression", "xgboost", "isolation_forest", "stacker"):
        assert (models / f"{name}.joblib").exists(), name

    table = pd.read_csv(results / "results.csv")
    assert table["model"].iloc[0] == "risk_engine"
    assert {"xgboost", "logistic_regression", "isolation_forest", "baseline_always_approve"} <= set(table["model"])
    meta = json.loads((results / "meta.json").read_text())
    assert meta["replay_seed"] == 7 and set(meta["feature_reference"]) == set(meta["features"])
    assert set(meta["engine_weights"]) == {"xgboost", "logistic_regression", "isolation_forest"}