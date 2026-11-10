"""Train and evaluate the fraud models on PaySim with time-based windows.

Three detectors are trained, then combined into one risk engine:
  1. Logistic regression  - interpretable baseline
  2. XGBoost              - stronger supervised model
  3. Isolation Forest     - unsupervised anomaly detector (never sees labels)
  4. Risk engine          - a small stacking model that learns how to weight
                            the three (fitted on a held-out validation window)

Time windows (never random):
  steps 1 .. split-val_steps        base models are trained here
  steps split-val_steps+1 .. split  validation: stacker fitted, reference scores
  steps > split                     test and replay (never used for training)

Usage
-----
Run from the project root (the folder that contains train/, src/ and data/):

    python train/train.py --data-path data/PS_20174392719_1491204439457_log.csv --drop-features amount_to_balance oldbalanceOrg

Other options: --include-balances (leakage experiment), --drop-features (ablation).

Outputs (the folder name is the run tag: leakage_safe, with_balances, or
leakage_safe_no_<dropped features> for ablations):
  models/<tag>/
    logistic_regression.joblib, xgboost.joblib, isolation_forest.joblib, stacker.joblib
  results/<tag>/
    results.csv           metrics table for the report
    replay_sample.csv.gz  the test period (all fraud + a sample of legitimate rows) for the app's replay
    baseline_scores.npz   validation-window scores (reference for risk score and drift)
    meta.json             settings used (windows, features, learned engine weights, ...)
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

from src.engine import STACK_ORDER, make_stacker, stack_features  # noqa: E402
from src.scoring import score  # noqa: E402
from src.features import (  # noqa: E402
    build_features, feature_columns, load_paysim, log_columns, signed_log1p,
)

SEED = 42
STEPS_PER_DAY = 24
REFERENCE_SIZE = 50_000
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
                   help="Steps <= this are training+validation data; later steps are test/replay")
    p.add_argument("--val-steps", type=int, default=100,
                   help="Last N steps before the split held out to fit the risk engine")
    p.add_argument("--include-balances", action="store_true",
                   help="Add post-transaction balance features (leakage experiment)")
    p.add_argument("--daily-capacity", type=int, default=300,
                   help="K: transactions the review team can check per simulated day")
    p.add_argument("--max-train-legit", type=int, default=400_000,
                   help="Cap on legitimate rows used by the supervised models")
    p.add_argument("--iforest-sample", type=int, default=200_000,
                   help="Unlabelled training rows used by Isolation Forest")
    p.add_argument("--replay-steps", type=int, default=None,
                   help="Limit the replay file to the first N test-period steps (default: the whole test period)")
    p.add_argument("--replay-legit-per-step", type=int, default=150,
                   help="Legitimate rows kept per step in the replay file")
    p.add_argument("--replay-seed", type=int, default=None,
                   help="Seed for the legitimate-row sample (default: random, and saved in meta.json)")
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
    if not 1 <= args.val_steps < args.split_step:
        sys.exit("ERROR: --val-steps must be between 1 and --split-step minus 1.")

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

    fit_end = args.split_step - args.val_steps
    fit = df[df["step"] <= fit_end].reset_index(drop=True)
    val = df[(df["step"] > fit_end) & (df["step"] <= args.split_step)].reset_index(drop=True)
    test = df[df["step"] > args.split_step].reset_index(drop=True)
    for name, part in (("training", fit), ("validation", val), ("test", test)):
        if part["isFraud"].sum() == 0:
            sys.exit(f"ERROR: the {name} period has no fraud. Adjust --split-step / --val-steps.")
    log.info("Train: %d rows (%d fraud) | Validation: %d (%d fraud) | Test: %d (%d fraud) | features: %s%s",
             len(fit), fit.isFraud.sum(), len(val), val.isFraud.sum(), len(test), test.isFraud.sum(),
             "with balances (LEAKY)" if inc else "leakage-safe",
             f" | dropped: {', '.join(drop)}" if drop else "")

    # ---- Supervised training set: all fraud + capped sample of legit -------
    fraud = fit[fit.isFraud == 1]
    legit = fit[fit.isFraud == 0]
    if len(legit) > args.max_train_legit:
        legit = legit.sample(args.max_train_legit, random_state=SEED)
    sup = pd.concat([fraud, legit]).sample(frac=1.0, random_state=SEED)
    X_sup, y_sup = build_features(sup, inc, drop), sup["isFraud"].to_numpy()
    pos_weight = len(legit) / len(fraud)

    # ---- Unlabelled training sample for Isolation Forest (natural ratio) ---
    unl = fit.sample(min(args.iforest_sample, len(fit)), random_state=SEED)
    X_unl = build_features(unl, inc, drop)

    X_val, y_val = build_features(val, inc, drop), val["isFraud"].to_numpy()
    X_test, y_test = build_features(test, inc, drop), test["isFraud"].to_numpy()
    steps_test = test["step"].to_numpy()

    # ---- Train the three base models ---------------------------------------
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

    # ---- Fit the risk engine (stacker) on the validation window ------------
    log.info("Fitting the risk engine on the validation window...")
    comp_val = pd.DataFrame({n: score(n, m, X_val) for n, m in models.items()}, index=X_val.index)
    comp_test = pd.DataFrame({n: score(n, m, X_test) for n, m in models.items()}, index=X_test.index)
    stacker = make_stacker().fit(stack_features(comp_val), y_val)
    raw_val = stacker.decision_function(stack_features(comp_val))
    raw_test = stacker.decision_function(stack_features(comp_test))
    joblib.dump(stacker, models_out / "stacker.joblib")

    coefs = stacker.named_steps["clf"].coef_[0]
    weights = dict(zip(STACK_ORDER, (float(c) for c in coefs)))
    log.info("Engine weights (on standardised signals): %s",
             ", ".join(f"{k}={v:+.2f}" for k, v in weights.items()))

    # ---- Evaluate on the later period --------------------------------------
    rows = []
    row = evaluate_model("risk_engine", y_test, raw_test, steps_test, args.daily_capacity)
    row["train_seconds"] = float(sum(timings.values()))
    rows.append(row)
    for name in models:
        row = evaluate_model(name, y_test, comp_test[name].to_numpy(), steps_test, args.daily_capacity)
        row["train_seconds"] = timings[name]
        rows.append(row)
    for row in rows:
        log.info("%s: PR-AUC=%.4f recall@K=%.3f precision@K=%.3f",
                 row["model"], row["pr_auc"], row["recall_at_k"], row["precision_at_k"])
    rows += evaluate_baselines(test, y_test)

    results = pd.DataFrame(rows)
    results.to_csv(out / "results.csv", index=False)
    print("\n", results.round(4).to_string(index=False))

    # ---- Reference score distributions (validation window, out-of-sample) --
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(val), size=min(REFERENCE_SIZE, len(val)), replace=False)
    np.savez_compressed(
        out / "baseline_scores.npz",
        engine=raw_val[idx],
        **{n: comp_val[n].to_numpy()[idx] for n in models},
    )

    # ---- Small replay sample (keeps repo light) ----------------------------
    window = test if args.replay_steps is None else test[test["step"] <= args.split_step + args.replay_steps]
    w_fraud = window[window.isFraud == 1]
    w_legit = window[window.isFraud == 0]
    n_steps = int(window["step"].max()) - args.split_step
    n_legit = min(len(w_legit), args.replay_legit_per_step * n_steps)
    # The replay sample is random by default; the seed is saved so it can be reproduced.
    replay_seed = (args.replay_seed if args.replay_seed is not None
                   else int(np.random.SeedSequence().entropy % 2**32))
    replay = pd.concat([w_fraud, w_legit.sample(n_legit, random_state=replay_seed)])
    replay = replay.sort_values("step", kind="stable").reset_index(drop=True)
    replay.to_csv(out / "replay_sample.csv.gz", index=False)
    log.info("Replay sample: %d rows over %d hours (%.2f%% fraud - legit rows are subsampled, "
             "so this is higher than the true rate). Sample seed: %d",
             len(replay), n_steps, 100 * replay.isFraud.mean(), replay_seed)

    meta = {
        "feature_set": tag, "include_balances": inc,
        "features": feats, "dropped_features": list(drop),
        "split_step": args.split_step, "val_steps": args.val_steps,
        "daily_capacity_k": args.daily_capacity,
        "train_rows": int(len(fit)), "val_rows": int(len(val)), "test_rows": int(len(test)),
        "test_fraud_rate": float(y_test.mean()),
        "supervised_pos_weight": float(pos_weight),
        "engine_weights": weights,
        "engine_intercept": float(stacker.named_steps["clf"].intercept_[0]),
        # typical value of every model input (median of an unlabelled sample), used by explanations
        "feature_reference": {k: float(v) for k, v in X_unl.median().items()},
        "replay_seed": replay_seed, "replay_rows": int(len(replay)),
        "replay_first_step": int(replay["step"].min()), "replay_last_step": int(replay["step"].max()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Done. Results in %s | models in %s", out, models_out)


if __name__ == "__main__":
    main()