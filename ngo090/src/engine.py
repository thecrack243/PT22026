"""The risk engine: three models combined into ONE risk score.

Users of the app never choose a model. Behind the scenes the engine runs
three detectors and a small "stacking" model learns how much to trust each:

    xgboost              known fraud patterns (supervised)
    logistic_regression  overall amount / balance profile (supervised, linear)
    isolation_forest     unusual behaviour (unsupervised, needs no labels)

How it is trained (see train/train.py)
--------------------------------------
    steps 1..fit_end      the three base models are trained
    steps fit_end..split  a validation window the base models never saw:
                          the stacker is fitted here, and the engine's score
                          distribution on this window becomes the reference
    steps > split         test and replay (never used for training)

Risk score
----------
The stacker's raw output is converted to a 0-100 risk score: the percentile
of that raw score among the reference (validation) traffic. A risk of 99.5
means "more suspicious than 99.5% of the traffic the engine saw while
validating". This puts every decision threshold on one intuitive scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import build_features
from src.scoring import MODEL_NAMES, load_models, score

SEED = 42
# Fixed column order of the stacker's inputs.
STACK_ORDER = ("xgboost", "logistic_regression", "isolation_forest")

# Plain-language names for what each detector contributes (shown in the app).
SIGNAL_LABELS = {
    "xgboost": "Known fraud pattern",
    "logistic_regression": "Amount and balance profile",
    "isolation_forest": "Unusual behaviour",
}


def stack_features(components: pd.DataFrame) -> pd.DataFrame:
    """Put the three raw model scores on a comparable footing for the stacker.

    Probabilities are turned into log-odds; the anomaly score is used as is.
    """
    eps = 1e-6
    out = pd.DataFrame(index=components.index)
    for name in ("xgboost", "logistic_regression"):
        p = components[name].clip(eps, 1 - eps)
        out[name] = np.log(p / (1 - p))
    out["isolation_forest"] = components["isolation_forest"]
    return out[list(STACK_ORDER)]


def make_stacker() -> Pipeline:
    """Small logistic regression that learns how to weight the three signals."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
    ])


def to_risk(raw: np.ndarray, reference_sorted: np.ndarray) -> np.ndarray:
    """Convert raw stacker scores to a 0-100 percentile against the reference."""
    ranks = np.searchsorted(reference_sorted, raw, side="right")
    return 100.0 * ranks / len(reference_sorted)


@dataclass
class EngineOutput:
    risk: np.ndarray          # 0-100, higher = more suspicious
    signal: np.ndarray        # plain-language main reason for each transaction
    raw: np.ndarray           # stacker decision value
    components: pd.DataFrame  # raw score of each base model
    features: pd.DataFrame | None = None   # the model inputs that produced these scores


class RiskEngine:
    """Loads the trained models and scores transactions with one risk score."""

    def __init__(self, models: dict, stacker: Pipeline, reference: np.ndarray,
                 include_balances: bool = False, drop: tuple[str, ...] = (),
                 feature_reference: pd.Series | None = None):
        self.models = models
        self.stacker = stacker
        self.reference = np.sort(np.asarray(reference, dtype=float))
        self.include_balances = include_balances
        self.drop = tuple(drop)
        # A "typical" value for every model input; explanations compare against it.
        self.feature_reference = feature_reference

    @classmethod
    def load(cls, root: str | Path, tag: str) -> "RiskEngine":
        root = Path(root)
        models_dir = root / "models" / tag
        results_dir = root / "results" / tag

        models = load_models(models_dir)
        stacker_path = models_dir / "stacker.joblib"
        baseline_path = results_dir / "baseline_scores.npz"
        if not stacker_path.exists():
            raise FileNotFoundError(
                f"{stacker_path} not found. Retrain with the latest train/train.py "
                "(see the README, section 4)."
            )
        with np.load(baseline_path) as npz:
            if "engine" not in npz.files:
                raise ValueError(
                    f"{baseline_path} has no engine reference. Retrain with the latest train/train.py."
                )
            reference = npz["engine"]

        import json
        meta = json.loads((results_dir / "meta.json").read_text())
        ref = meta.get("feature_reference")
        return cls(models, joblib.load(stacker_path), reference,
                   include_balances=meta.get("include_balances", False),
                   drop=tuple(meta.get("dropped_features", [])),
                   feature_reference=pd.Series(ref, dtype=float) if ref else None)

    def feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Model inputs for raw transactions (same code as at training time)."""
        return build_features(df, self.include_balances, self.drop)

    def _components_from(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {name: score(name, self.models[name], X) for name in MODEL_NAMES},
            index=X.index,
        )

    def components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw score of every base model for every transaction."""
        return self._components_from(self.feature_matrix(df))

    def raw_from_features(self, X: pd.DataFrame) -> np.ndarray:
        """Stacker output for a ready-made feature matrix (used by explanations)."""
        return self.stacker.decision_function(stack_features(self._components_from(X)))

    def score(self, df: pd.DataFrame) -> EngineOutput:
        return self.score_features(self.feature_matrix(df))

    def score_features(self, X: pd.DataFrame) -> EngineOutput:
        comp = self._components_from(X)
        feats = stack_features(comp)
        raw = self.stacker.decision_function(feats)
        risk = np.clip(to_risk(raw, self.reference), 0.0, 100.0)

        # Main reason: the signal contributing most to this transaction's score.
        scaled = self.stacker.named_steps["scale"].transform(feats)
        contrib = scaled * self.stacker.named_steps["clf"].coef_[0]
        top = np.argmax(contrib, axis=1)
        labels = np.array([SIGNAL_LABELS[STACK_ORDER[i]] for i in top], dtype=object)
        return EngineOutput(risk=risk, signal=labels, raw=raw, components=comp, features=X)