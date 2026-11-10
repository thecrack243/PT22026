"""Fraud Audit Console: risk monitoring, simulation and testing around one risk engine.

Run from the project root (the folder that contains app/, src/ and models/):

    streamlit run app/app.py

Requires the files produced by train/train.py (see the README, section 4).
The user never picks a model: the risk engine combines three detectors
behind the scenes and the app only shows its decisions and results.
"""
from __future__ import annotations

import common                      # sets up the import paths; keep this import first
import streamlit as st

import mode_live
import mode_replay
import mode_testing
import ui
from src.artifacts import discover_runs

MODES = ("Historical replay", "Live simulation", "Transaction testing")

st.set_page_config(page_title="Fraud Audit Console", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")


def render_nav() -> str:
    """Mode switch under the header. Returns the mode to show."""
    st.session_state.setdefault("mode", MODES[0])
    if hasattr(st, "segmented_control"):
        choice = st.segmented_control("Mode", MODES, default=st.session_state.mode, key="mode_choice",
                                      label_visibility="collapsed")
    else:
        choice = st.radio("Mode", MODES, horizontal=True, index=MODES.index(st.session_state.mode),
                          key="mode_choice", label_visibility="collapsed")
    if choice:                                     # a segmented control can be deselected; keep the last mode
        st.session_state.mode = choice
    return st.session_state.mode


def main() -> None:
    st.markdown(ui.STYLE, unsafe_allow_html=True)

    runs = discover_runs(common.ROOT)
    if not runs:
        st.error(
            "No trained risk engine found. Run the training command from the project root first:\n\n"
            "`python train/train.py --data-path data/<PaySim csv> --drop-features amount_to_balance oldbalanceOrg`"
        )
        st.stop()

    tag = runs[0]                                   # the recommended run comes first
    try:
        run = common.get_run(tag)
        engine = common.get_engine(tag)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Could not load the risk engine: {exc}")
        st.stop()

    header = st.empty()                             # each mode fills in its own header
    mode = render_nav()
    if mode != MODES[0]:
        st.session_state["playing"] = False         # a replay does not keep running in the background

    if mode == MODES[0]:
        mode_replay.render(header, tag, run)
    elif mode == MODES[1]:
        mode_live.render(header, tag, run, engine)
    else:
        mode_testing.render(header, tag, run, engine)


main()