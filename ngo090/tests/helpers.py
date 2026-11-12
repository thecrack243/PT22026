"""Shared test helper: a small risk engine built from synthetic data (scikit-learn models only)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.engine import RiskEngine, make_stacker, stack_features
from src.features import build_features


def make_engine(seed=0, n=6000):
    """Fraud = the sender's whole balance is moved into an empty account."""
    rng = np.random.default_rng(seed)
    bal = rng.lognormal(10, 1, n)
    fraud = rng.random(n) < 0.05
    df = pd.DataFrame({
        "step": rng.integers(1, 200, n),
        "type": rng.choice(["TRANSFER", "CASH_OUT"], n),
        "amount": np.where(fraud, bal, bal * rng.uniform(0.01, 0.5, n)),
        "oldbalanceOrg": bal,
        "oldbalanceDest": np.where(fraud, 0.0, rng.lognormal(10, 1, n)),
    })
    X, y = build_features(df), fraud.astype(int)
    models = {
        "xgboost": HistGradientBoostingClassifier(max_iter=60, random_state=0).fit(X, y),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(np.log1p(X.clip(lower=0)), y),
        "isolation_forest": IsolationForest(n_estimators=50, random_state=0).fit(X),
    }
    # the logistic pipeline above expects log inputs; wrap so it accepts the raw matrix
    lr = models["logistic_regression"]
    models["logistic_regression"] = type("LR", (), {"predict_proba": lambda self, M: lr.predict_proba(np.log1p(M.clip(lower=0)))})()
    comp = pd.DataFrame({
        "xgboost": models["xgboost"].predict_proba(X)[:, 1],
        "logistic_regression": models["logistic_regression"].predict_proba(X)[:, 1],
        "isolation_forest": -models["isolation_forest"].score_samples(X),
    }, index=X.index)
    stacker = make_stacker().fit(stack_features(comp), y)
    reference = stacker.decision_function(stack_features(comp))
    return RiskEngine(models, stacker, reference, feature_reference=X.median()), X, y