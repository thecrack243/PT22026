"""Mode 1, Historical replay: a past period replayed hour by hour with the risk engine."""
from __future__ import annotations

import random
import time
from datetime import datetime

import numpy as np
import streamlit as st

import ui
from common import (
    MIN_EXPLAIN_RISK, apply_recommendation, get_engine, get_run, popover, render_costs_popover,
    render_policy_popover,
)
from src.artifacts import score_rows
from src.decision import recommend_policy
from src.explain import explain_matrix
from src.monitor import compute_view
from src.replay import (
    SAMPLE_FRACTION, day_of_step, default_days, sample_replay, select_window, window_bounds,
)

# Settings of this mode that must survive a visit to another mode.
PREF_KEYS = ("days", "random_sample", "fixed_seed", "shift", "speed")


def _restore_prefs() -> None:
    for key, value in st.session_state.get("_replay_prefs", {}).items():
        st.session_state.setdefault(key, value)


def _save_prefs() -> None:
    st.session_state["_replay_prefs"] = {k: st.session_state[k] for k in PREF_KEYS if k in st.session_state}


@st.cache_data(show_spinner="Scoring transactions...")
def get_window_output(tag: str, start: int, end: int, shift: bool):
    """Risk score and main reason of every pool transaction in the chosen hours."""
    engine = get_engine(tag)
    df = select_window(get_run(tag).replay, start, end)
    shift_from = start + (end - start + 1) // 2 if shift else None
    out = score_rows(engine, df, shift_from)
    # Only transactions the policy can flag (risk >= MIN_EXPLAIN_RISK) need a reason.
    reasons, _ = explain_matrix(engine, out.features, out.risk, min_risk=MIN_EXPLAIN_RISK)
    return out.risk, reasons


# --------------------------------------------------------------------------- #
# Replay state
# --------------------------------------------------------------------------- #
def _new_seed() -> int:
    return random.randrange(1, 1_000_000)           # 0 is reserved for "random sample"


def init_state() -> None:
    st.session_state.setdefault("pos", 0)           # simulated hours replayed
    st.session_state.setdefault("playing", False)
    st.session_state.setdefault("total", 0)         # hours in the chosen window
    st.session_state.setdefault("run_seed", _new_seed())


def _toggle_play() -> None:
    total = st.session_state.total
    if not st.session_state.playing and st.session_state.pos >= total:
        st.session_state.pos = 0                    # replay from the start
    if not st.session_state.playing and st.session_state.pos == 0:
        st.session_state.run_seed = _new_seed()     # a new run draws a new random sample
    st.session_state.playing = not st.session_state.playing


def _advance(hours: int) -> None:
    if st.session_state.pos == 0:
        st.session_state.run_seed = _new_seed()
    st.session_state.playing = False
    st.session_state.pos = min(st.session_state.pos + hours, st.session_state.total)


def _reset_run() -> None:
    """Back to the start (also used when the replay days or sampling change)."""
    st.session_state.playing = False
    st.session_state.pos = 0
    st.session_state.run_seed = _new_seed()


def tick(total: int) -> bool:
    """Move one simulated hour forward if playing. Returns True when it just finished."""
    if not st.session_state.playing:
        return False
    st.session_state.pos = min(st.session_state.pos + 1, total)
    if st.session_state.pos >= total:
        st.session_state.playing = False
        return True
    return False


# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
def render_controls(day_lo: int, day_hi: int) -> dict:
    """Buttons and settings. Returns the policy, costs and scenario settings in use."""
    cols = st.columns([1, 1, 1, 1, 1.5, 1.2, 1.5])

    pos, total = st.session_state.pos, st.session_state.total
    if st.session_state.playing:
        label = "Stop"
    elif pos == 0 or pos >= total:
        label = "Start"
    else:
        label = "Continue"
    cols[0].button(label, on_click=_toggle_play, type="primary")
    cols[1].button("+1 hour", on_click=_advance, args=(1,))
    cols[2].button("+1 day", on_click=_advance, args=(24,))
    cols[3].button("Reset", on_click=_reset_run)

    with cols[4]:
        policy = render_policy_popover()
    with cols[5]:
        costs = render_costs_popover()
    with cols[6]:
        with popover()("Scenario"):
            if day_lo < day_hi:
                st.slider("Replay days", day_lo, day_hi, default_days(day_lo, day_hi), key="days",
                          on_change=_reset_run,
                          help="Which simulated days to replay. Changing this restarts the replay.")
            random_on = st.toggle("Random sample each run", value=True, key="random_sample", on_change=_reset_run,
                                  help=f"On: each new run replays a different random {SAMPLE_FRACTION:.0%} of the "
                                       "transactions, so results vary like real traffic. "
                                       "Off: you choose the sample yourself (next field).")
            st.number_input("Sample number (0 = all transactions)", min_value=0, max_value=999_999,
                            value=0, step=1, key="fixed_seed", on_change=_reset_run, disabled=random_on,
                            help="Available when 'Random sample each run' is off. Type the number shown in the header "
                                 "(for example 203674) to replay exactly the same transactions again. "
                                 "0 replays every transaction.")
            st.toggle("Simulate a market shift", value=False, key="shift",
                      help="From the middle of the replay, amounts become 10x larger and receivers are new accounts. "
                           "Watch the drift monitor react.")
            speed = st.slider("Seconds per simulated hour", 0.1, 2.0, 0.4, 0.1, key="speed")

    return {"policy": policy, "costs": costs, "speed": float(speed)}


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
def draw_dashboard(replay, risk, reason, steps, cfg) -> None:
    pos = min(st.session_state.pos, len(steps))
    st.session_state.pos = pos
    view = compute_view(replay, risk, reason, steps, pos, cfg["policy"], cfg["costs"])
    st.markdown(ui.dashboard_html(view, datetime.now().strftime("%H:%M:%S")), unsafe_allow_html=True)


def render(header, tag: str, run) -> None:
    """Draw the whole replay mode. `header` is the placeholder at the top of the page."""
    init_state()
    _restore_prefs()
    pool_first, pool_last = int(run.replay["step"].min()), int(run.replay["step"].max())
    day_lo, day_hi = day_of_step(pool_first), day_of_step(pool_last)

    # Settings chosen in the Scenario popover (widget values live in session_state).
    first_day, last_day = st.session_state.get("days", default_days(day_lo, day_hi))
    random_on = st.session_state.get("random_sample", True)
    shift_on = st.session_state.get("shift", False)

    start, end = window_bounds(pool_first, pool_last, first_day, last_day)
    steps = np.arange(start, end + 1)               # every simulated hour of the window
    st.session_state.total = len(steps)

    fixed_seed = int(st.session_state.get("fixed_seed", 0))
    if random_on:
        seed, sample_text = st.session_state.run_seed, f"random sample #{st.session_state.run_seed:06d}"
    elif fixed_seed > 0:
        seed, sample_text = fixed_seed, f"fixed sample #{fixed_seed:06d}"
    else:
        seed, sample_text = None, "all transactions"
    pos = st.session_state.pos
    state = ("playing" if st.session_state.playing else "finished" if pos >= len(steps)
             else "ready" if pos == 0 else "stopped")
    header.markdown(ui.header_html(state, f"PaySim replay · days {first_day}–{last_day} · {sample_text}"),
                    unsafe_allow_html=True)

    cfg = render_controls(day_lo, day_hi)
    _save_prefs()

    window_df = select_window(run.replay, start, end)
    risk, reason = get_window_output(tag, start, end, shift_on)
    if seed is None:
        replay = window_df                          # every transaction, no sampling
    else:
        replay, risk, reason = sample_replay(window_df, risk, reason, SAMPLE_FRACTION, seed)

    if hasattr(st, "fragment"):
        interval = cfg["speed"] if st.session_state.playing else None

        @st.fragment(run_every=interval)
        def live() -> None:
            finished = tick(len(steps))
            draw_dashboard(replay, risk, reason, steps, cfg)
            if finished:
                st.rerun()                          # refresh the buttons and header

        live()
        render_advisor(replay, risk, steps, cfg)
    else:                                           # older Streamlit: rerun the whole page
        finished = tick(len(steps))
        draw_dashboard(replay, risk, reason, steps, cfg)
        render_advisor(replay, risk, steps, cfg)
        if st.session_state.playing or finished:
            time.sleep(cfg["speed"])
            st.rerun()
    render_help()


def render_advisor(replay, risk, steps, cfg) -> None:
    """Optional cost curves and a recommended policy for the replay on screen."""
    if not st.toggle("Show the policy advisor", value=False, key="advisor_on",
                     help="Tries many auto-block and review-capacity settings on the hours replayed so far "
                          "and shows which one earns the most with your costs."):
        return
    # Only the hours already replayed: advising from hours the analyst has not
    # reached yet would leak the future into the recommendation.
    pos = min(st.session_state.pos, len(steps))
    if pos == 0:
        st.caption("Start the replay first: the advisor needs at least one replayed hour.")
        return
    n_seen = int(np.searchsorted(replay["step"].to_numpy(), int(steps[pos - 1]), side="right"))
    with st.spinner("Trying policies..."):
        rec = recommend_policy(replay.iloc[:n_seen].reset_index(drop=True), risk[:n_seen],
                               cfg["costs"], cfg["policy"])
    st.session_state.advisor_rec = (rec.best_block, rec.best_capacity)
    current_block = 100.0 if cfg["policy"].block_threshold == float("inf") else cfg["policy"].block_threshold
    st.markdown(ui.advisor_html(rec, current_block, cfg["policy"].daily_capacity), unsafe_allow_html=True)
    st.button("Apply the recommended settings", on_click=apply_recommendation)


HELP_TEXT = """
**What you are looking at.** A past period of transactions is replayed hour by hour. For each transaction the risk
engine gives a risk score from 0 to 100, your decision rules turn it into an action (approve, send to review, or
block), and the dashboard compares those actions with what really happened.

**Numbers with letters.** K means thousand, M million and B billion, in the currency units of the dataset.
For example, a net benefit of 478.37M means about 478,370,000 units.

**Top tiles** (the small text compares the last 6 simulated hours with the 6 hours before):
- *Transactions*: how many transactions have been replayed so far.
- *Alerts raised*: transactions that were blocked or sent to review. More alerts means more work for the team.
- *Fraud caught*: share of all real fraud so far that was blocked or sent to review.
- *Net benefit*: fraud money stopped, minus the cost of reviews and of blocking honest customers, compared with
  having no system at all. A positive value means the system pays for itself.

**Real-time trends.** Blue is transactions per hour and red is alerts per hour. The clock reads day and hour of the
simulated period (D22 10:00 is day 22 at 10:00). A k on the axis means thousand.

**Decisions.** Share of transactions that were approved, sent to review or blocked. The centre is the alert rate.

**Money impact.** Green is the value of fraud that was stopped, red is the value of fraud that got through.
You want green far above red.

**Main reasons for alerts.** For every alert the engine checks which fact about the transaction pushed its score up
most (for example an unusually large amount or an empty receiving account). The bars show how often each reason
was the main one. Reasons describe what the engine reacted to; they are not proof of fraud.

**Latest alerts.** Risk is the 0 to 100 score (99.3 means riskier than 99.3% of normal traffic). Main reason is the fact that raised
the score most. Decision is what
the system did. Outcome says whether the alert was real fraud or a false alarm; this is known only because the
replay has the true labels. In real life that answer arrives days later.

**System status.** The drift monitor compares the last 24 hours of risk scores with normal traffic: a shift index
below 0.10 is stable, 0.10 to 0.25 means watch, above 0.25 means the traffic has changed and the model may be less
reliable. The review queue shows how many of the day's review slots are used.

**Policy advisor.** Switch it on below the dashboard to see how the net benefit changes with the auto-block score
and with the review capacity, and which settings earn the most with your costs. It is a backtest that uses the true
labels, so treat it as guidance. "Apply the recommended settings" copies them into the Decision policy.

**Controls.** Decision policy sets when to block and how many reviews per day are possible; the presets are
ready-made combinations. Costs sets what a
review and a wrongly blocked customer cost. Scenario chooses the replay days, whether every run uses a new random
sample or one specific sample (turn the random option off and type its number), and can simulate a market shift
to trigger the drift monitor.
"""


def render_help() -> None:
    with st.expander("How to read this dashboard"):
        st.markdown(HELP_TEXT)