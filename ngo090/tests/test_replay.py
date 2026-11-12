"""Unit tests for src/replay.py. Run from the project root: python -m pytest"""
import numpy as np
import pandas as pd
import pytest

from src.replay import day_of_step, default_days, sample_replay, select_window, window_bounds


def test_day_of_step():
    assert [day_of_step(s) for s in (1, 24, 25, 501, 743)] == [1, 1, 2, 21, 31]


def test_window_bounds_clips_to_the_pool_and_validates():
    assert window_bounds(501, 743, 22, 24) == (505, 576)       # three full days
    assert window_bounds(501, 743, 21, 21) == (501, 504)       # partial first day
    assert window_bounds(501, 743, 30, 40) == (697, 743)       # clipped at the end
    with pytest.raises(ValueError):
        window_bounds(501, 743, 24, 22)
    with pytest.raises(ValueError):
        window_bounds(501, 743, 40, 41)


def test_default_days_skips_the_partial_first_day():
    assert default_days(21, 31) == (22, 24)
    assert default_days(21, 22) == (22, 22)


def test_select_window_keeps_order_and_range():
    df = pd.DataFrame({"step": [1, 2, 2, 3, 4], "v": range(5)})
    out = select_window(df, 2, 3)
    assert list(out["v"]) == [1, 2, 3]


def test_sample_replay_is_reproducible_and_aligned():
    n = 1000
    df = pd.DataFrame({"step": np.arange(n), "id": np.arange(n)})
    risk, signal = np.arange(n, dtype=float), np.array([str(i) for i in range(n)], dtype=object)
    a = sample_replay(df, risk, signal, 0.8, seed=7)
    b = sample_replay(df, risk, signal, 0.8, seed=7)
    c = sample_replay(df, risk, signal, 0.8, seed=8)
    assert list(a[0]["id"]) == list(b[0]["id"])                 # same seed, same sample
    assert list(a[0]["id"]) != list(c[0]["id"])                 # new seed, new sample
    assert 700 < len(a[0]) < 900                                # about 80% kept
    assert list(a[1]) == [float(i) for i in a[0]["id"]]         # scores stay aligned with rows
    assert list(a[2]) == [str(i) for i in a[0]["id"]]
    with pytest.raises(ValueError):
        sample_replay(df, risk, signal, 0.0, seed=1)