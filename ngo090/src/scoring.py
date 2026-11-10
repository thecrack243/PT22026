"""Model loading and scoring, shared by train/train.py and the app.

Every model is exposed through the same convention: a score where
HIGHER = MORE SUSPICIOUS, so the decision layer can treat them identically.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_NAMES = ("logistic_regression", "xgboost", "isolation_forest")


def score(name: str, model, X: pd.DataFrame) -> np.ndarray:
    """Return a 'fraud score' where HIGHER = MORE SUSPICIOUS, for every model."""
    if name == "isolation_forest":
        return -model.score_samples(X)          # negate: low normality = suspicious
    return model.predict_proba(X)[:, 1]


def load_models(models_dir: str | Path) -> dict:
    """Load the three trained models from a models/<tag>/ folder."""
    models_dir = Path(models_dir)
    models = {}
    for name in MODEL_NAMES:
        path = models_dir / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}. Run train/train.py first "
                "(see the README, section 4)."
            )
        models[name] = joblib.load(path)
    return models