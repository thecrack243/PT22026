"""Drift monitoring on the engine's risk scores.

Because the risk score is a percentile of the reference traffic, healthy
traffic spreads evenly across the ten risk bands (0-10, 10-20, ... 90-100).
The Population Stability Index (PSI) measures how far recent traffic has moved
away from that even spread. It needs no labels, so it works in production
where fraud labels arrive late.

Common rule of thumb: PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 drift.
"""
from __future__ import annotations

import numpy as np

MIN_SAMPLES = 200


def psi_vs_uniform(risk, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI of risk scores against the even spread expected from the reference."""
    risk = np.clip(np.asarray(risk, dtype=float), 0.0, 100.0)
    counts, _ = np.histogram(risk, bins=bins, range=(0.0, 100.0))
    p = np.clip(counts / max(counts.sum(), 1), eps, None)
    q = np.full(bins, 1.0 / bins)
    return float(np.sum((p - q) * np.log(p / q)))


def drift_status(psi: float | None) -> str:
    """'warming' / 'stable' / 'watch' / 'drift'."""
    if psi is None:
        return "warming"
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "watch"
    return "drift"


def psi_vs_baseline(baseline, current, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI of `current` risk scores against the distribution of a `baseline` sample.

    Bins are the baseline's deciles, so a baseline that looks like the current
    traffic gives a value near zero whatever the shape of the distribution.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(baseline, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3 or len(current) == 0:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    q = np.histogram(baseline, bins=edges)[0] / len(baseline)
    p = np.histogram(current, bins=edges)[0] / len(current)
    q, p = np.clip(q, eps, None), np.clip(p, eps, None)
    return float(np.sum((p - q) * np.log(p / q)))