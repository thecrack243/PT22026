"""Finding and loading the files produced by train/train.py.

Each training run is saved under a tag (for example
'leakage_safe_no_amount_to_balance_oldbalanceOrg'):

    models/<tag>/*.joblib
    results/<tag>/meta.json, replay_sample.csv, baseline_scores.npz, results.csv
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.engine import EngineOutput, RiskEngine
from src.scoring import MODEL_NAMES

# The setting used by default in the app (hard problem, models make mistakes).
PREFERRED_RUN = "leakage_safe_no_amount_to_balance_oldbalanceOrg"
RESULT_FILES = ("meta.json", "replay_sample.csv.gz", "baseline_scores.npz")
REPLAY_COLUMNS = ("step", "type", "amount", "oldbalanceOrg", "oldbalanceDest", "isFraud")


@dataclass
class RunData:
    tag: str
    meta: dict
    replay: pd.DataFrame                 # sorted by step
    baseline: dict[str, np.ndarray]      # training-period scores per model


def discover_runs(root: str | Path) -> list[str]:
    """Tags of runs that have both results and models, preferred run first."""
    root = Path(root)
    results_dir, models_dir = root / "results", root / "models"
    if not results_dir.is_dir():
        return []
    tags = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        has_results = all((d / f).exists() for f in RESULT_FILES)
        has_models = all((models_dir / d.name / f"{n}.joblib").exists()
                         for n in (*MODEL_NAMES, "stacker"))
        if has_results and has_models:
            tags.append(d.name)
    return sorted(tags, key=lambda t: (t != PREFERRED_RUN, t))


def load_run(root: str | Path, tag: str) -> RunData:
    """Load metadata, replay transactions and baseline scores of one run."""
    d = Path(root) / "results" / tag
    meta = json.loads((d / "meta.json").read_text())
    replay = pd.read_csv(d / "replay_sample.csv.gz")
    missing = [c for c in REPLAY_COLUMNS if c not in replay.columns]
    if missing:
        raise ValueError(f"{d / 'replay_sample.csv.gz'} is missing columns: {missing}")
    replay = replay.sort_values("step", kind="stable").reset_index(drop=True)
    with np.load(d / "baseline_scores.npz") as npz:
        baseline = {name: npz[name] for name in npz.files}
    return RunData(tag=tag, meta=meta, replay=replay, baseline=baseline)


def load_engine(root: str | Path, tag: str) -> RiskEngine:
    """Load the risk engine (three models + stacker) of one run.

    Runs trained before reference values were saved in meta.json get one
    computed from the legitimate transactions of the replay file instead.
    """
    engine = RiskEngine.load(root, tag)
    if engine.feature_reference is None:
        replay = load_run(root, tag).replay
        legit = replay[replay["isFraud"] == 0]
        engine.feature_reference = engine.feature_matrix(legit).median()
    return engine


SHIFT_AMOUNT_FACTOR = 10.0


def score_rows(engine: RiskEngine, df: pd.DataFrame, shift_from_step: int | None = None) -> EngineOutput:
    """Score transactions with the engine.

    With shift_from_step set, the model inputs (not the money amounts used in
    the cost calculation) change from that hour on, to simulate the market
    changing under the model: amounts become 10x larger and the receiving
    accounts are all new (zero balance). Used to demonstrate the drift monitor.
    """
    if shift_from_step is not None:
        df = df.copy()
        later = df["step"] >= shift_from_step
        df.loc[later, "amount"] = df.loc[later, "amount"] * SHIFT_AMOUNT_FACTOR
        df.loc[later, "oldbalanceDest"] = 0.0
    return engine.score(df)


def score_replay(run: RunData, engine: RiskEngine, shift_from_step: int | None = None) -> EngineOutput:
    """Score the whole replay pool of a run."""
    return score_rows(engine, run.replay, shift_from_step)