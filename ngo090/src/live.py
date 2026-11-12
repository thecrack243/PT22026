"""The live simulation: a stream of generated transactions scored as they arrive.

Behaviour (pure Python, no Streamlit):
  - A simulated clock runs faster than the real one (sim_minutes_per_second).
  - Each time advance() is called, every simulated minute that has passed gets its
    transactions generated, scored, decided and stored.
  - A decision is made when the transaction arrives and is never changed afterwards.
    Changing the policy only affects transactions that arrive later, as in a real system.
  - Review capacity is a budget that grows every simulated minute (daily capacity / 1440),
    carries over, and is capped at one day's worth. Each minute the highest-risk candidates
    use it first.
  - The true label stays in the table for judging results; the engine never receives it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.decision import APPROVE, BLOCK, REVIEW, CostConfig, Policy
from src.engine import RiskEngine
from src.generator import GeneratorConfig, TransactionGenerator
from src.drift import MIN_SAMPLES, drift_status, psi_binary, psi_vs_baseline
from src.monitor import build_view
from src.processing import score_clean

SIM_START_MINUTE = 8 * 60          # the simulation starts at day 1, 08:00
MINUTES_PER_DAY = 24 * 60
MAX_CATCHUP_MINUTES = MINUTES_PER_DAY     # a long absence is capped at one simulated day
MAX_ROWS = 100_000                 # oldest rows are dropped beyond this
VIEW_HOURS = 36                    # hours shown on the charts
DRIFT_HOURS = 3                    # recent hours compared with the three hours before them
NO_TIME = 0.0


def minute_label(minute: int) -> str:
    """Compact simulated time, e.g. 'D2 14:05'."""
    return f"D{minute // MINUTES_PER_DAY + 1} {(minute // 60) % 24:02d}:{minute % 60:02d}"


def clock_label(minute: int) -> str:
    return f"Day {minute // MINUTES_PER_DAY + 1} · {(minute // 60) % 24:02d}:{minute % 60:02d}"


@dataclass(frozen=True)
class LiveSettings:
    volume_per_hour: int = 300
    fraud_rate: float = 0.03
    new_pattern: bool = False
    sim_minutes_per_second: float = 2.0
    market_shift: bool = False

    def generator_config(self) -> GeneratorConfig:
        return GeneratorConfig(self.volume_per_hour, self.fraud_rate, self.new_pattern, self.market_shift)


class LiveStream:
    """Generates, scores and stores transactions as simulated time passes."""

    def __init__(self, engine: RiskEngine, seed: int | None = None, settings: LiveSettings | None = None,
                 now: float | None = None):
        self.engine = engine
        self.settings = settings or LiveSettings()
        self.generator = TransactionGenerator(seed)
        self.rows = pd.DataFrame()
        self.minute = SIM_START_MINUTE          # next simulated minute to generate
        self._clock = float(SIM_START_MINUTE)   # simulated time as a fraction of a minute
        self._last_wall = time.time() if now is None else now
        self.budget = 0.0
        self.proc_ms_sum = 0.0
        self.proc_count = 0
        self.processed = 0               # all-time total; rows are trimmed, this is not
        self.recent_throughput = 0.0

    # ------------------------------------------------------------------ #
    def advance(self, policy: Policy, now: float | None = None, settings: LiveSettings | None = None) -> dict:
        """Process everything that arrived since the last call. Returns what happened."""
        now = time.time() if now is None else now
        if settings is not None:
            self.settings = settings
        dt = max(now - self._last_wall, 0.0)
        self._last_wall = now
        # The clock is capped like the generation: if the laptop slept for hours, an
        # uncapped clock jumps millions of simulated days forward and every timestamp
        # in the dashboard becomes nonsense. At most one simulated day passes while
        # nobody is watching; the session simply resumes from there.
        self._clock += min(dt * self.settings.sim_minutes_per_second, float(MAX_CATCHUP_MINUTES))
        target = int(self._clock)

        start_minute, caught_up = self.minute, target - self.minute
        if caught_up <= 0:
            return {"new": 0, "new_alerts": 0, "caught_up_minutes": 0}

        new = self.generator.generate(start_minute, target, self.settings.generator_config())
        self.minute = target
        new_alerts = 0
        if len(new):
            t0 = time.perf_counter()
            risk, reason, applicable = score_clean(self.engine, new)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            new["risk"], new["reason"] = risk, reason
            new["applicable"] = applicable
            new["decision"] = self._decide(new, start_minute, target, policy)
            new["proc_ms"] = elapsed_ms / len(new)
            self.proc_ms_sum += elapsed_ms
            self.proc_count += len(new)
            self.processed += len(new)
            new_alerts = int((new["decision"] != APPROVE).sum())
            self.rows = new if self.rows.empty else pd.concat([self.rows, new], ignore_index=True)
            if len(self.rows) > MAX_ROWS:
                self.rows = self.rows.iloc[-int(MAX_ROWS * 0.8):].reset_index(drop=True)
            self.recent_throughput = len(new) / max(dt, 1.0)   # at least a 1 s window: no absurd spikes
        else:
            self._accrue_only(start_minute, target, policy)
        return {"new": len(new), "new_alerts": new_alerts, "caught_up_minutes": caught_up}

    def _mix_psi(self, current_step: int) -> float | None:
        """How far the shape of arriving traffic moved away from the session's start.

        The engine's risk score is a percentile of the traffic it was validated on,
        so its distribution follows the time of day: hour_of_day is a model input and
        fraud is far more common at night. Comparing score distributions of two short
        windows therefore reports drift on completely healthy traffic. The transaction
        mix does not have that problem — a market shift or a new fraud pattern changes
        amounts and receiver balances, and ordinary traffic does not.
        """
        rows = self.rows
        first_step = SIM_START_MINUTE // 60 + 1
        reference = rows[rows["step"] <= first_step + DRIFT_HOURS - 1]
        # Nights are quiet, so three night hours can hold too few transactions to
        # compare. Widen the recent window (never past the session start) until the
        # sample is usable instead of dropping the monitor to "warming up" overnight.
        recent = rows[rows["step"] > current_step - DRIFT_HOURS]
        for wider in (2 * DRIFT_HOURS, 4 * DRIFT_HOURS):
            if len(recent) >= MIN_SAMPLES or recent.empty:
                break
            recent = rows[rows["step"] > current_step - wider]
        if len(recent) < MIN_SAMPLES or len(reference) < MIN_SAMPLES:
            return None
        amounts = psi_vs_baseline(np.log10(reference["amount"].clip(lower=1.0)),
                                  np.log10(recent["amount"].clip(lower=1.0)))
        empty = psi_binary(reference["oldbalanceDest"] == 0, recent["oldbalanceDest"] == 0)
        return float(max(amounts, empty))

    def _accrue_only(self, start: int, end: int, policy: Policy) -> None:
        per_minute = policy.daily_capacity / MINUTES_PER_DAY
        self.budget = min(self.budget + per_minute * (end - start), float(policy.daily_capacity))

    def _decide(self, new: pd.DataFrame, start: int, end: int, policy: Policy) -> np.ndarray:
        """Decisions minute by minute, using the review budget as it grows."""
        decisions = np.full(len(new), APPROVE, dtype=object)
        risk = new["risk"].to_numpy()
        usable = new["applicable"].to_numpy()
        block = usable & (risk >= policy.block_threshold)
        candidate = usable & ~block & (risk >= policy.review_min_score)
        decisions[block] = BLOCK

        per_minute = policy.daily_capacity / MINUTES_PER_DAY
        positions = pd.Series(np.arange(len(new))).groupby(new["minute"].to_numpy()).indices
        for m in range(start, end):
            self.budget = min(self.budget + per_minute, float(policy.daily_capacity))
            idx = positions.get(m)
            if idx is None:
                continue
            idx = idx[candidate[idx]]
            n_reviews = min(int(self.budget), len(idx))
            if n_reviews > 0:
                top = idx[np.argsort(-risk[idx])[:n_reviews]]
                decisions[top] = REVIEW
                self.budget -= n_reviews
        return decisions

    # ------------------------------------------------------------------ #
    @property
    def clock_minute(self) -> int:
        return self.minute

    @property
    def avg_ms(self) -> float:
        """Average processing time per transaction so far, in milliseconds."""
        return self.proc_ms_sum / self.proc_count if self.proc_count else 0.0

    def view(self, policy: Policy, costs: CostConfig) -> dict:
        """Dashboard numbers (same structure as the replay's view)."""
        current_step = max(self.minute - 1, 0) // 60 + 1
        hours = np.arange(max(current_step - VIEW_HOURS + 1, SIM_START_MINUTE // 60 + 1), current_step + 1)
        if self.rows.empty:
            empty = pd.DataFrame({"step": [], "type": [], "amount": [], "isFraud": [], "hour": []})
            r, why, action = np.array([]), np.array([], dtype=object), pd.Series([], dtype=object)
            return build_view(empty, r, why, action, hours[:0], current_step - 1, policy.daily_capacity, costs)
        df = self.rows
        action = pd.Series(df["decision"].to_numpy(), index=df.index)
        view = build_view(df, df["risk"].to_numpy(), df["reason"].to_numpy(), action, hours, current_step,
                          policy.daily_capacity, costs)
        # Drift: the transaction mix of the last hours against the hours before them,
        # not against the training data (see _mix_psi).
        psi = self._mix_psi(current_step)
        view["psi"], view["drift"] = psi, drift_status(psi)
        return view

    def latest(self, n: int = 14) -> pd.DataFrame:
        """The most recent transactions, newest first."""
        return self.rows.tail(n).iloc[::-1] if len(self.rows) else self.rows

    def alerts(self, n: int = 12) -> pd.DataFrame:
        """The most recent alerts (review or block), newest first."""
        if self.rows.empty:
            return self.rows
        return self.rows[self.rows["decision"] != APPROVE].tail(n).iloc[::-1]

    def get(self, txn_id: int) -> pd.Series | None:
        match = self.rows[self.rows["id"] == txn_id] if len(self.rows) else self.rows
        return None if match.empty else match.iloc[0]