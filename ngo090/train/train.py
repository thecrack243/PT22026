"""Train and evaluate three fraud models on PaySim with a time-based split.

Models compared (same split, same features, same metrics):
  1. Logistic regression  - interpretable baseline
  2. XGBoost              - stronger supervised model
  3. Isolation Forest     - unsupervised anomaly detector (never sees labels)

Usage
-----
Run from the project root (the folder that contains train/, src/ and data/):

    python train/train.py --data-path data/PS_20174392719_1491204439457_log.csv
    python train/train.py --data-path data/PS_...csv --include-balances   # leakage experiment
    python train/train.py --data-path data/PS_...csv --drop-features amount_to_balance  # ablation

Outputs (the folder name is the run tag: leakage_safe, with_balances, or
leakage_safe_no_<dropped features> for ablations):
  models/<tag>/
    *.joblib              trained models
  results/<tag>/
    results.csv           metrics table for the report
    replay_sample.csv     small slice of the test period for the app's replay
    baseline_scores.npz   training-period score samples (reference for drift checks)
    meta.json             settings used (split step, features, capacity, ...)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# This file lives in train/, while the shared code lives in ../src,
# so add the project root to the import path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    build_features, feature_columns, load_paysim, log_columns, signed_log1p,
)

SEED = 42
STEPS_PER_DAY = 24
log = logging.getLogger("train")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def make_logistic_regression(include_balances: bool, drop: tuple[str, ...] = ()) -> Pipeline:
    """Signed-log + scaling on money columns, scaling on the rest, balanced LR."""
    cols = feature_columns(include_balances, drop)
    logc = log_columns(include_balances, drop)
    rest = [c for c in cols if c not in logc]
    prep = ColumnTransformer([
        ("log", Pipeline([
            ("log", FunctionTransformer(signed_log1p)),
            ("scale", StandardScaler()),
        ]), logc),
        ("rest", StandardScaler(), rest),
    ])
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    return Pipeline([("prep", prep), ("clf", clf)])


def make_xgboost(scale_pos_weight: float):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        sys.exit("xgboost is not installed. Run: pip install -r requirements.txt")
    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,   # handles class imbalance
        tree_method="hist",                  # fast on CPU
        eval_metric="aucpr", n_jobs=-1, random_state=SEED,
    )


def make_isolation_forest() -> IsolationForest:
    return IsolationForest(n_estimators=200, contamination="auto",
                           n_jobs=-1, random_state=SEED)


def score(name: str, model, X: pd.DataFrame) -> np.ndarray:
    """Return a 'fraud score' where HIGHER = MORE SUSPICIOUS, for every model."""
    if name == "isolation_forest":
        return -model.score_samples(X)          # negate: low normality = suspicious
    return model.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def recall_at_precision(y: np.ndarray, s: np.ndarray, min_precision: float) -> float:
    prec, rec, _ = precision_recall_curve(y, s)
    ok = prec >= min_precision
    return float(rec[ok].max()) if ok.any() else 0.0


def capacity_flags(scores: np.ndarray, steps: np.ndarray, k_per_day: int) -> np.ndarray:
    """Flag the top-K highest-scored transactions in each simulated day.

    This mimics a review team that can only look at K cases per day.
    """
    day = (steps - 1) // STEPS_PER_DAY
    ranks = pd.Series(scores).groupby(day).rank(method="first", ascending=False)
    return (ranks.to_numpy() <= k_per_day)


def confusion(y: np.ndarray, flags: np.ndarray) -> dict:
    tp = int(((flags == 1) & (y == 1)).sum())
    fp = int(((flags == 1) & (y == 0)).sum())
    fn = int(((flags == 0) & (y == 1)).sum())
    tn = int(((flags == 0) & (y == 0)).sum())
    n = tp + fp + fn + tn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "accuracy": (tp + tn) / n,
    }


def evaluate_model(name, y, s, steps, k) -> dict:
    flags = capacity_flags(s, steps, k)
    row = {
        "model": name,
        "pr_auc": average_precision_score(y, s),
        "roc_auc": roc_auc_score(y, s),
        "recall_at_precision_90": recall_at_precision(y, s, 0.90),
    }
    row.update({f"{key}_at_k": val for key, val in confusion(y, flags).items()})
    return row


def evaluate_baselines(df_test: pd.DataFrame, y: np.ndarray) -> list[dict]:
    """Trivial baselines every model must beat."""
    prevalence = float(y.mean())
    rows = []

    flags = np.zeros(len(y), dtype=bool)
    r = {"model": "baseline_always_approve", "pr_auc": prevalence}
    r.update({f"{k}_at_k": v for k, v in confusion(y, flags).items()})
    rows.append(r)

    flags = ((df_test["type"] == "TRANSFER") & (df_test["amount"] > 200_000)).to_numpy()
    r = {"model": "baseline_rule_large_transfer",
         "pr_auc": average_precision_score(y, flags.astype(float))}
    r.update({f"{k}_at_k": v for k, v in confusion(y, flags).items()})
    rows.append(r)

    # PaySim fraudsters typically drain the whole account, so this one-line rule
    # is a very strong baseline on the simulated data (see the report).
    flags = ((df_test["amount"] == df_test["oldbalanceOrg"]) & (df_test["amount"] > 0)).to_numpy()
    r = {"model": "baseline_rule_full_balance_drain",
         "pr_auc": average_precision_score(y, flags.astype(float))}
    r.update({f"{k}_at_k": v for k, v in confusion(y, flags).items()})
    rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", required=True, help="Path to the PaySim CSV")
    p.add_argument("--output-dir", default=str(ROOT / "results"),
                   help="Where to save metrics and replay data (default: <project>/results)")
    p.add_argument("--models-dir", default=str(ROOT / "models"),
                   help="Where to save trained models (default: <project>/models)")
    p.add_argument("--split-step", type=int, default=500,
                   help="Steps <= this are training data; later steps are test/replay")
    p.add_argument("--include-balances", action="store_true",
                   help="Add post-transaction balance features (leakage experiment)")
    p.add_argument("--daily-capacity", type=int, default=300,
                   help="K: transactions the review team can check per simulated day")
    p.add_argument("--max-train-legit", type=int, default=400_000,
                   help="Cap on legitimate rows used by the supervised models")
    p.add_argument("--iforest-sample", type=int, default=200_000,
                   help="Unlabelled training rows used by Isolation Forest")
    p.add_argument("--replay-steps", type=int, default=72,
                   help="How many test-period steps go into replay_sample.csv")
    p.add_argument("--replay-legit-per-step", type=int, default=300,
                   help="Legitimate rows kept per step in replay_sample.csv")
    p.add_argument("--drop-features", nargs="*", default=[], metavar="NAME",
                   help="Feature(s) to remove for an ablation, e.g. amount_to_balance")
    p.add_argument("--nrows", type=int, default=None,
                   help="Read only the first N CSV rows (quick tests)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    inc = args.include_balances
    drop = tuple(args.drop_features)
    try:
        feats = feature_columns(inc, drop)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    tag = "with_balances" if inc else "leakage_safe"
    if drop:
        tag += "_no_" + "_".join(drop)
    out = Path(args.output_dir) / tag
    models_out = Path(args.models_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    models_out.mkdir(parents=True, exist_ok=True)

    # ---- Load and split by time (never randomly) ---------------------------
    log.info("Loading data (%s)...", args.data_path)
    try:
        df = load_paysim(args.data_path, nrows=args.nrows)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")

    train = df[df["step"] <= args.split_step].reset_index(drop=True)
    test = df[df["step"] > args.split_step].reset_index(drop=True)
    if train["isFraud"].sum() == 0 or test["isFraud"].sum() == 0:
        sys.exit("ERROR: train or test period has no fraud. Adjust --split-step.")
    log.info("Train: %d rows (%d fraud) | Test: %d rows (%d fraud) | features: %s%s",
             len(train), train.isFraud.sum(), len(test), test.isFraud.sum(),
             "with balances (LEAKY)" if inc else "leakage-safe",
             f" | dropped: {', '.join(drop)}" if drop else "")

    # ---- Supervised training set: all fraud + capped sample of legit -------
    fraud = train[train.isFraud == 1]
    legit = train[train.isFraud == 0]
    if len(legit) > args.max_train_legit:
        legit = legit.sample(args.max_train_legit, random_state=SEED)
    sup = pd.concat([fraud, legit]).sample(frac=1.0, random_state=SEED)
    X_sup, y_sup = build_features(sup, inc, drop), sup["isFraud"].to_numpy()
    pos_weight = len(legit) / len(fraud)

    # ---- Unlabelled training sample for Isolation Forest (natural ratio) ---
    unl = train.sample(min(args.iforest_sample, len(train)), random_state=SEED)
    X_unl = build_features(unl, inc, drop)

    X_test, y_test = build_features(test, inc, drop), test["isFraud"].to_numpy()
    steps_test = test["step"].to_numpy()

    # ---- Train -------------------------------------------------------------
    models, timings = {}, {}
    specs = {
        "logistic_regression": (make_logistic_regression(inc, drop), X_sup, y_sup),
        "xgboost": (make_xgboost(pos_weight), X_sup, y_sup),
        "isolation_forest": (make_isolation_forest(), X_unl, None),  # no labels
    }
    for name, (model, X, y) in specs.items():
        log.info("Training %s...", name)
        t0 = time.time()
        model.fit(X) if y is None else model.fit(X, y)
        timings[name] = round(time.time() - t0, 1)
        models[name] = model
        joblib.dump(model, models_out / f"{name}.joblib")

    # ---- Evaluate on the later period --------------------------------------
    rows = []
    for name, model in models.items():
        s = score(name, model, X_test)
        row = evaluate_model(name, y_test, s, steps_test, args.daily_capacity)
        row["train_seconds"] = timings[name]
        rows.append(row)
        log.info("%s: PR-AUC=%.4f recall@K=%.3f precision@K=%.3f",
                 name, row["pr_auc"], row["recall_at_k"], row["precision_at_k"])
    rows += evaluate_baselines(test, y_test)

    results = pd.DataFrame(rows)
    results.to_csv(out / "results.csv", index=False)
    print("\n", results.round(4).to_string(index=False))

    # ---- Reference score distributions for the app's drift panel -----------
    ref = train.sample(min(20_000, len(train)), random_state=SEED)
    X_ref = build_features(ref, inc, drop)
    np.savez_compressed(out / "baseline_scores.npz",
                        **{n: score(n, m, X_ref) for n, m in models.items()})

    # ---- Small replay sample (keeps repo light) ----------------------------
    window = test[test["step"] <= args.split_step + args.replay_steps]
    w_fraud = window[window.isFraud == 1]
    w_legit = window[window.isFraud == 0]
    n_legit = min(len(w_legit), args.replay_legit_per_step * args.replay_steps)
    replay = pd.concat([w_fraud, w_legit.sample(n_legit, random_state=SEED)])
    replay = replay.sort_values("step").reset_index(drop=True)
    replay.to_csv(out / "replay_sample.csv", index=False)
    log.info("Replay sample: %d rows (%.2f%% fraud - legit rows are subsampled, "
             "so this is higher than the true rate).",
             len(replay), 100 * replay.isFraud.mean())

    meta = {
        "feature_set": tag, "include_balances": inc,
        "features": feats, "dropped_features": list(drop),
        "split_step": args.split_step,
        "daily_capacity_k": args.daily_capacity,
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "test_fraud_rate": float(y_test.mean()),
        "supervised_pos_weight": float(pos_weight),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Done. Results in %s | models in %s", out, models_out)


if __name__ == "__main__":
    main()
