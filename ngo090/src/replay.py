"""Choosing and sampling what gets replayed (pure Python, no Streamlit).

The replay file covers the whole test period. The app lets the user pick which
simulated days to replay, and by default replays a different random sample of
the transactions every run, so results vary the way real traffic would.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STEPS_PER_DAY = 24
SAMPLE_FRACTION = 0.8       # share of transactions kept in a random replay sample


def day_of_step(step: int) -> int:
    """Simulated day (1-based) that an hour belongs to."""
    return (int(step) - 1) // STEPS_PER_DAY + 1


def default_days(first_day: int, last_day: int) -> tuple[int, int]:
    """Default replay window: three days, starting one day into the pool.

    The first day of the pool is usually partial, so skipping it gives a window
    of full days.
    """
    return min(first_day + 1, last_day), min(first_day + 3, last_day)


def window_bounds(pool_first_step: int, pool_last_step: int,
                  first_day: int, last_day: int) -> tuple[int, int]:
    """First and last hour (step) of the chosen days, limited to the pool."""
    if first_day > last_day:
        raise ValueError("The first replay day must not be after the last one.")
    start = max(pool_first_step, (first_day - 1) * STEPS_PER_DAY + 1)
    end = min(pool_last_step, last_day * STEPS_PER_DAY)
    if start > end:
        raise ValueError("The chosen days contain no transactions.")
    return start, end


def select_window(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Rows of the replay pool between two hours (inclusive), order preserved."""
    mask = (df["step"] >= start) & (df["step"] <= end)
    return df[mask].reset_index(drop=True)


def sample_replay(df: pd.DataFrame, risk: np.ndarray, signal: np.ndarray,
                  fraction: float, seed: int):
    """Random subset of the replay rows (and their scores) for one run.

    The same seed always gives the same subset, so a run can be reproduced.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    keep = np.random.default_rng(seed).random(len(df)) < fraction
    if not keep.any():
        keep[0] = True
    return df[keep].reset_index(drop=True), risk[keep], signal[keep]