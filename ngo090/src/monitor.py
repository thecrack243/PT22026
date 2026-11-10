"""Turns a replay state into the numbers the dashboard shows.

Pure Python (no Streamlit), so it can be tested on its own. The UI in
app/ui.py only draws what compute_view() returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision import (
    APPROVE, BLOCK, REVIEW, CostConfig, Policy, cumulative_by_step, decide, evaluate,
)
from src.drift import MIN_SAMPLES, drift_status, psi_vs_uniform

WINDOW = 6            # hours used for "vs previous 6 hours" comparisons
DRIFT_WINDOW = 24     # hours of recent traffic used for the drift check


def hour_label(step: int) -> str:
    """Simulated clock: step 1 = day 1, 00:00."""
    return f"D{(step - 1) // 24 + 1} {(step - 1) % 24:02d}:00"


def _pct_change(x: np.ndarray, w: int = WINDOW):
    if len(x) < 2 * w:
        return None
    prev = x[-2 * w:-w].sum()
    return None if prev == 0 else float((x[-w:].sum() - prev) / prev * 100.0)


def compute_view(replay: pd.DataFrame, risk: np.ndarray, reason: np.ndarray,
                 steps: np.ndarray, pos: int, policy: Policy, costs: CostConfig) -> dict:
    """Everything the dashboard needs after `pos` simulated hours.

    pos == 0 gives the "nothing replayed yet" view: the same structure with all
    counts at zero, so the dashboard layout is complete before the replay starts.
    """
    last_step = int(steps[pos - 1]) if pos > 0 else int(steps[0]) - 1
    hours = steps[:pos]
    n_seen = int(np.searchsorted(replay["step"].to_numpy(), last_step, side="right"))
    df, r, why = replay.iloc[:n_seen], risk[:n_seen], reason[:n_seen]

    action = decide(df, r, policy)
    view = build_view(df, r, why, action, hours, last_step, policy.daily_capacity, costs)
    view.update(pos=pos, total=len(steps), first_step=int(steps[0]), end_step=int(steps[-1]))
    return view


def build_view(df: pd.DataFrame, r: np.ndarray, why: np.ndarray, action: pd.Series, hours: np.ndarray,
               last_step: int, capacity: int, costs: CostConfig) -> dict:
    """Dashboard numbers for transactions whose actions are already decided.

    Used by the replay (actions come from decide()) and by the live simulation
    (actions were decided when each transaction arrived).
    `df` needs the columns step, type, amount and isFraud; `hours` are the simulated
    hours (steps) to show on the time axis.
    """
    m = evaluate(df, action, costs)
    a = action.to_numpy()
    flagged = a != APPROVE
    fraud = (df["isFraud"] == 1).to_numpy()
    per_hour = (pd.DataFrame({
        "step": df["step"].to_numpy(), "tx": 1,
        "alerts": flagged.astype(int), "fraud": fraud.astype(int),
        "caught": (flagged & fraud).astype(int),
    }).groupby("step").sum().reindex(hours, fill_value=0))

    cum = cumulative_by_step(df, action, costs).reindex(hours).ffill().fillna(0.0)
    cum_fraud = per_hour["fraud"].cumsum().to_numpy()
    cum_caught = per_hour["caught"].cumsum().to_numpy()
    recall_series = np.divide(cum_caught, cum_fraud, out=np.zeros(len(hours)), where=cum_fraud > 0)
    net = cum["net_benefit"].to_numpy()

    # Main reason behind each alert (from src/explain.py), as a share of all alerts.
    shares = []
    labelled = flagged & (why != "")
    if labelled.any():
        counts = pd.Series(why[labelled]).value_counts(normalize=True)
        shares = [(str(k), float(v)) for k, v in counts.head(6).items()]

    # Latest alerts, newest first.
    last = df.assign(risk=r, reason=why, action=a)
    last = last[flagged].tail(8).iloc[::-1]
    recent = [{
        "hour": hour_label(int(row.step)), "type": row.type, "amount": float(row.amount),
        "risk": float(row.risk), "reason": row.reason, "action": row.action,
        "fraud": bool(row.isFraud == 1),
    } for row in last.itertuples()]

    # Review queue usage today, and drift on the most recent traffic.
    today = (last_step - 1) // 24
    steps_df = df["step"].to_numpy()
    reviews_today = int(((a == REVIEW) & ((steps_df - 1) // 24 == today)).sum())
    recent_mask = steps_df > last_step - DRIFT_WINDOW
    psi = psi_vs_uniform(r[recent_mask]) if recent_mask.sum() >= MIN_SAMPLES else None

    return {
        "metrics": m, "pos": len(hours), "total": len(hours), "last_step": last_step,
        "first_step": int(hours[0]) if len(hours) else last_step, "end_step": last_step,
        "hours": [hour_label(int(s)) for s in hours],
        "tx_per_hour": per_hour["tx"].to_numpy(), "alerts_per_hour": per_hour["alerts"].to_numpy(),
        "cum_saved": cum["fraud_saved"].to_numpy(), "cum_lost": cum["fraud_lost"].to_numpy(),
        "cum_net": net, "recall_series": recall_series,
        "tx_change": _pct_change(per_hour["tx"].to_numpy()),
        "alerts_change": _pct_change(per_hour["alerts"].to_numpy()),
        "recall_change": (float(recall_series[-1] - recall_series[-1 - WINDOW]) * 100.0
                         if len(hours) > WINDOW else None),
        "net_change": float(net[-1] - net[-1 - WINDOW]) if len(hours) > WINDOW else None,
        "reason_shares": shares, "recent": recent,
        "reviews_today": reviews_today, "capacity": capacity,
        "psi": psi, "drift": drift_status(psi),
    }