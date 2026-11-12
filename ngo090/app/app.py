"""Fraud Audit Console: risk monitoring, simulation and testing around one risk engine.

Run from the project root (the folder that contains app/, src/ and models/):

    streamlit run app/app.py

Requires the files produced by train/train.py (see the README, section 4).
The user never picks a model: the risk engine combines three detectors
behind the scenes and the app only shows its decisions and results.
"""
from __future__ import annotations

import sys
from pathlib import Path

# This file is also the entry point, so the import paths must exist before the
# first import of a sibling module (common does it again for its own imports).
# Without this, the app only starts when the working directory happens to be
# the project root.
_APP_DIR = Path(__file__).resolve().parent
for _path in (str(_APP_DIR.parent), str(_APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import common                      # noqa: E402  (sets up the import paths)
import streamlit as st             # noqa: E402

import mode_live                   # noqa: E402
import mode_replay                 # noqa: E402
import mode_testing                # noqa: E402
import ui                          # noqa: E402
from src.artifacts import discover_runs  # noqa: E402

REPLAY, LIVE, TESTING = "Historical replay", "Live simulation", "Transaction testing"
MODES = (LIVE, REPLAY, TESTING)    # the live feed is what greets the analyst

st.set_page_config(page_title="Fraud Audit Console", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")


def render_nav() -> str:
    """Mode switch under the header. Returns the mode to show.

    The widget has no key on purpose: its default is the current mode, so the
    stored state and the widget can never disagree (a keyed widget plus a
    default produces a Streamlit warning and a stale selection).
    """
    st.session_state.setdefault("mode", MODES[0])
    if hasattr(st, "segmented_control"):
        choice = st.segmented_control("Mode", MODES, default=st.session_state.mode,
                                      label_visibility="collapsed")
    else:
        choice = st.radio("Mode", MODES, horizontal=True, index=MODES.index(st.session_state.mode),
                          label_visibility="collapsed")
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
    if mode != REPLAY:
        st.session_state["playing"] = False         # a replay does not keep running in the background

    if mode == REPLAY:
        mode_replay.render(header, tag, run)
    elif mode == LIVE:
        mode_live.render(header, tag, run, engine)
    else:
        mode_testing.render(header, tag, run, engine)


main()