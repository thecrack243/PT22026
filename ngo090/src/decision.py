"""Risk decision layer for the Fraud Audit Console.

The models (see train/train.py) only produce a fraud SCORE per transaction.
This module turns scores into ACTIONS and measures what those actions would
have cost or saved. It contains no machine learning and no Streamlit code, so
it is easy to test and works with any of the three models.

Actions
-------
    block    score >= block_threshold: the transaction is stopped automatically.
    review   not blocked, score >= review_min_score, and the review team still
             has capacity: a human analyst checks it.
    approve  everything else.

Review capacity
---------------
The review team can only check a limited number of transactions.

    "rolling"     (default, realistic) A budget of daily_capacity / 24 reviews
                  is added every simulated hour, unused budget carries over
                  (capped at one day's worth). Each hour the top-scored
                  candidates are reviewed. This only uses information
                  available at that hour, as a live system would.
    "daily_topk"  The top K scored candidates of each simulated day are
                  reviewed. This matches recall@K in train/train.py, but it
                  needs the whole day's scores in advance, so it is an
                  offline (non-causal) view.

Money
-----
Units are whatever the `amount` column uses (PaySim currency units), so the
cost settings must be on the same scale as the amounts.

    fraud blocked          + amount saved
    fraud reviewed         + amount * review_catch_rate saved
    fraud approved         amount lost (nothing saved)
    any reviewed           - review_cost
    legitimate blocked     - friction_cost (an annoyed or lost customer)

    net_benefit = money saved - review costs - friction costs
                = cost of having no system - cost with this policy
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STEPS_PER_DAY = 24
APPROVE, REVIEW, BLOCK = "approve", "review", "block"
CAPACITY_MODES = ("rolling", "daily_topk")
REQUIRED_COLUMNS = ("step", "amount", "isFraud")


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostConfig:
    """Costs of the decisions, in the same units as the transaction amount.

    The defaults are placeholders: PaySim amounts are large (fraud averages
    about a million), so tune these in the app until the trade-off is
    meaningful.
    """
    review_cost: float = 1_000.0       # analyst time per reviewed transaction
    friction_cost: float = 5_000.0     # cost of blocking a legitimate customer
    review_catch_rate: float = 1.0     # share of reviewed fraud analysts stop

    def __post_init__(self) -> None:
        if self.review_cost < 0 or self.friction_cost < 0:
            raise ValueError("Costs must be non-negative.")
        if not 0.0 <= self.review_catch_rate <= 1.0:
            raise ValueError("review_catch_rate must be between 0 and 1.")


@dataclass(frozen=True)
class Policy:
    """How scores are turned into actions."""
    block_threshold: float = float("inf")     # inf = never auto-block
    review_min_score: float = float("-inf")   # -inf = review whatever fits capacity
    daily_capacity: int = 300                 # reviews the team can do per day
    capacity_mode: str = "rolling"

    def __post_init__(self) -> None:
        if self.daily_capacity < 0:
            raise ValueError("daily_capacity must be >= 0.")
        if self.capacity_mode not in CAPACITY_MODES:
            raise ValueError(f"capacity_mode must be one of {CAPACITY_MODES}.")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _check_inputs(df: pd.DataFrame, scores) -> np.ndarray:
    """Validate the transactions table and scores, return scores as float array."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Transactions are missing required columns: {missing}")
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or len(s) != len(df):
        raise ValueError(
            f"scores must be 1-D with one value per transaction "
            f"(got {s.shape}, expected ({len(df)},))."
        )
    if np.isnan(s).any():
        raise ValueError("scores contain NaN values.")
    return s


def threshold_from_top_fraction(baseline_scores, fraction: float) -> float:
    """Score threshold that would flag the top `fraction` of baseline scores.

    Model scores live on different scales (probabilities vs. anomaly scores),
    so the app lets the user think in "block the top 0.5% most suspicious"
    and converts that to a raw threshold using the training-period scores.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    return float(np.quantile(np.asarray(baseline_scores, dtype=float), 1.0 - fraction))


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #
def _review_daily_topk(scores, candidate, steps, k) -> np.ndarray:
    day = (steps - 1) // STEPS_PER_DAY
    ranked = pd.Series(np.where(candidate, scores, -np.inf))
    ranks = ranked.groupby(day).rank(method="first", ascending=False).to_numpy()
    return candidate & (ranks <= k)


def _review_rolling(scores, candidate, steps, daily_capacity) -> np.ndarray:
    review = np.zeros(len(scores), dtype=bool)
    per_step = daily_capacity / STEPS_PER_DAY
    positions = pd.Series(np.arange(len(scores))).groupby(steps).indices

    budget, prev_step = 0.0, None
    for step in sorted(positions):
        elapsed = 1 if prev_step is None else step - prev_step
        prev_step = step
        budget = min(budget + per_step * elapsed, float(daily_capacity))

        idx = positions[step]
        idx = idx[candidate[idx]]
        n_reviews = min(int(budget), len(idx))
        if n_reviews > 0:
            top = idx[np.argsort(-scores[idx])[:n_reviews]]
            review[top] = True
            budget -= n_reviews
    return review


def decide(df: pd.DataFrame, scores, policy: Policy) -> pd.Series:
    """Return an action ('approve' / 'review' / 'block') for every transaction.

    `df` must be sorted by step, with columns step, amount, isFraud (the true
    label is not used here, only in the money calculation).
    """
    s = _check_inputs(df, scores)
    steps = df["step"].to_numpy()

    block = s >= policy.block_threshold
    candidate = ~block & (s >= policy.review_min_score)

    if policy.daily_capacity == 0:
        review = np.zeros(len(s), dtype=bool)
    elif policy.capacity_mode == "daily_topk":
        review = _review_daily_topk(s, candidate, steps, policy.daily_capacity)
    else:
        review = _review_rolling(s, candidate, steps, policy.daily_capacity)

    action = np.full(len(s), APPROVE, dtype=object)
    action[review] = REVIEW
    action[block] = BLOCK
    return pd.Series(action, index=df.index, name="action")


def decide_one(risk: float, policy: Policy) -> str:
    """Action for a single transaction.

    Review capacity cannot be known for one isolated transaction, so it is
    assumed available unless the policy has no review capacity at all.
    """
    if risk >= policy.block_threshold:
        return BLOCK
    if risk >= policy.review_min_score and policy.daily_capacity > 0:
        return REVIEW
    return APPROVE


# --------------------------------------------------------------------------- #
# Money
# --------------------------------------------------------------------------- #
def row_outcomes(df: pd.DataFrame, action: pd.Series, costs: CostConfig) -> pd.DataFrame:
    """Per-transaction money outcome, so totals and charts use the same numbers."""
    amount = df["amount"].to_numpy(dtype=float)
    fraud = df["isFraud"].to_numpy() == 1
    a = action.to_numpy()
    blocked, reviewed = a == BLOCK, a == REVIEW

    saved = (np.where(fraud & blocked, amount, 0.0)
             + np.where(fraud & reviewed, amount * costs.review_catch_rate, 0.0))
    review_cost = np.where(reviewed, costs.review_cost, 0.0)
    friction_cost = np.where(~fraud & blocked, costs.friction_cost, 0.0)

    return pd.DataFrame({
        "step": df["step"].to_numpy(),
        "fraud_amount": np.where(fraud, amount, 0.0),
        "fraud_saved": saved,
        "review_cost": review_cost,
        "friction_cost": friction_cost,
        "net_benefit": saved - review_cost - friction_cost,
    }, index=df.index)


def evaluate(df: pd.DataFrame, action: pd.Series, costs: CostConfig) -> dict:
    """Counts, detection quality and money totals for a set of actions."""
    out = row_outcomes(df, action, costs)
    fraud = df["isFraud"].to_numpy() == 1
    a = action.to_numpy()
    flagged = a != APPROVE

    fraud_total = int(fraud.sum())
    fraud_caught = int((fraud & flagged).sum())
    n_flagged = int(flagged.sum())

    fraud_amount = float(out["fraud_amount"].sum())
    fraud_saved = float(out["fraud_saved"].sum())
    review_total = float(out["review_cost"].sum())
    friction_total = float(out["friction_cost"].sum())
    fraud_lost = fraud_amount - fraud_saved

    return {
        "transactions": int(len(df)),
        "approved": int((a == APPROVE).sum()),
        "reviewed": int((a == REVIEW).sum()),
        "blocked": int((a == BLOCK).sum()),
        "fraud_total": fraud_total,
        "fraud_caught": fraud_caught,
        "fraud_missed": fraud_total - fraud_caught,
        "false_alarms": n_flagged - fraud_caught,
        "recall": fraud_caught / fraud_total if fraud_total else 0.0,
        "precision": fraud_caught / n_flagged if n_flagged else 0.0,
        "fraud_amount_total": fraud_amount,
        "fraud_amount_saved": fraud_saved,
        "fraud_amount_lost": fraud_lost,
        "review_cost_total": review_total,
        "friction_cost_total": friction_total,
        "total_cost": fraud_lost + review_total + friction_total,
        "baseline_cost": fraud_amount,            # cost with no system at all
        "net_benefit": fraud_saved - review_total - friction_total,
    }


def cumulative_by_step(df: pd.DataFrame, action: pd.Series, costs: CostConfig) -> pd.DataFrame:
    """Running totals per simulated hour, for the replay chart."""
    out = row_outcomes(df, action, costs)
    out["fraud_lost"] = out["fraud_amount"] - out["fraud_saved"]
    cols = ["fraud_saved", "fraud_lost", "review_cost", "friction_cost", "net_benefit"]
    return out.groupby("step")[cols].sum().cumsum()


def sweep_block_threshold(df: pd.DataFrame, scores, costs: CostConfig,
                          policy: Policy, thresholds) -> pd.DataFrame:
    """Evaluate the policy for many block thresholds (the 'cost curve').

    Lets the user see which threshold gives the best net benefit.
    """
    rows = []
    for t in thresholds:
        pol = Policy(block_threshold=float(t),
                     review_min_score=policy.review_min_score,
                     daily_capacity=policy.daily_capacity,
                     capacity_mode=policy.capacity_mode)
        metrics = evaluate(df, decide(df, scores, pol), costs)
        metrics["block_threshold"] = float(t)
        rows.append(metrics)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Presets and policy advice
# --------------------------------------------------------------------------- #
# Ready-made policies in risk-score units (the same values as the dashboard sliders).
PRESETS = {
    "Balanced": Policy(block_threshold=99.5, review_min_score=90.0, daily_capacity=150),
    "Light touch": Policy(block_threshold=99.9, review_min_score=95.0, daily_capacity=50),
    "Strict": Policy(block_threshold=99.0, review_min_score=85.0, daily_capacity=300),
}

# Settings the advisor tries (risk 100 means "never block automatically").
BLOCK_CANDIDATES = (90.0, 95.0, 97.0, 98.0, 98.5, 99.0, 99.3, 99.5, 99.7, 99.9, 100.0)
CAPACITY_CANDIDATES = (0, 50, 100, 150, 200, 300, 400, 500, 750, 1000)


@dataclass(frozen=True)
class Recommendation:
    """Result of trying many policies on a replay (a backtest)."""
    block_candidates: tuple[float, ...]
    block_net: tuple[float, ...]            # net benefit for each auto-block setting
    best_block: float
    capacity_candidates: tuple[int, ...]
    capacity_net: tuple[float, ...]         # net benefit for each capacity (at the best block setting)
    best_capacity: int
    current_net: float                      # net benefit of the policy in use
    best_net: float                         # net benefit of the recommended policy


def _with(policy: Policy, block: float, capacity: int) -> Policy:
    return Policy(block_threshold=float("inf") if block >= 100 else float(block),
                  review_min_score=policy.review_min_score, daily_capacity=int(capacity),
                  capacity_mode=policy.capacity_mode)


def recommend_policy(df: pd.DataFrame, scores, costs: CostConfig, policy: Policy,
                     block_candidates=BLOCK_CANDIDATES, capacity_candidates=CAPACITY_CANDIDATES) -> Recommendation:
    """Find the auto-block threshold and review capacity with the best net benefit.

    A backtest: it uses the true labels of the replayed transactions (hindsight),
    so it is guidance for choosing settings, not a prediction of the future.
    Ties go to the less aggressive setting (higher threshold, lower capacity).
    """
    scores = np.asarray(scores, dtype=float)

    def net(pol: Policy) -> float:
        return evaluate(df, decide(df, scores, pol), costs)["net_benefit"]

    block_net = [net(_with(policy, b, policy.daily_capacity)) for b in block_candidates]
    best_i = len(block_net) - 1 - int(np.argmax(block_net[::-1]))     # last maximum
    best_block = block_candidates[best_i]

    capacity_net = [net(_with(policy, best_block, c)) for c in capacity_candidates]
    best_c = int(np.argmax(capacity_net))                             # first maximum
    return Recommendation(
        block_candidates=tuple(block_candidates), block_net=tuple(block_net), best_block=best_block,
        capacity_candidates=tuple(capacity_candidates), capacity_net=tuple(capacity_net),
        best_capacity=int(capacity_candidates[best_c]),
        current_net=net(policy), best_net=capacity_net[best_c],
    )