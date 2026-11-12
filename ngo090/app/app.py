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

import mode_cases                  # noqa: E402
import mode_live                   # noqa: E402
import mode_replay                 # noqa: E402
import mode_testing                # noqa: E402
import ui                          # noqa: E402
from src.artifacts import discover_runs  # noqa: E402

REPLAY, LIVE, TESTING = "Historical replay", "Live simulation", "Transaction testing"
CASES = "Cases"                     # the workspace page, not a simulation mode
MODES = (LIVE, REPLAY, TESTING)    # the live feed is what greets the analyst
SLUG = {LIVE: "live", REPLAY: "replay", TESTING: "testing", CASES: "cases"}
ICON = {LIVE: ":material/sensors:", REPLAY: ":material/history:", TESTING: ":material/science:",
        CASES: ":material/folder_shared:"}

st.set_page_config(page_title="Fraud Audit Console", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")


def _switch_to(mode: str) -> None:
    """Runs before the page is drawn, so the sidebar and the page agree in one pass.

    Pop-ups belong to the page they were opened on: leaving the page dismisses
    them, so they do not reappear when the analyst comes back later.
    """
    st.session_state.mode = mode
    st.session_state["case_to_open"] = None      # investigation-case pop-up
    st.session_state["live_alert_open"] = None   # live alert pop-up


def render_sidebar(store) -> str:
    """The three modes live in the sidebar; the page itself stays one canvas."""
    st.session_state.setdefault("mode", MODES[0])
    with st.sidebar:
        st.markdown(ui.sidebar_brand(), unsafe_allow_html=True)
        st.markdown('<div class="fa-root"><div class="fa-side-label">Modes</div></div>',
                    unsafe_allow_html=True)
        for mode in MODES:
            st.button(mode, key=f"nav_{SLUG[mode]}", icon=ICON[mode], width="stretch",
                      type="primary" if st.session_state.mode == mode else "secondary",
                      on_click=_switch_to, args=(mode,))

        st.markdown('<div class="fa-root"><div class="fa-side-label" style="margin-top:16px">Workspace</div></div>',
                    unsafe_allow_html=True)
        active = store.counts()["active"]           # cases still being worked on
        label = f"Cases · {active}" if active else CASES
        st.button(label, key=f"nav_{SLUG[CASES]}", icon=ICON[CASES], width="stretch",
                  type="primary" if st.session_state.mode == CASES else "secondary",
                  on_click=_switch_to, args=(CASES,),
                  help="Alerts an analyst kept, stored in a local SQLite database.")
    return st.session_state.mode


def page_entry(slug: str) -> tuple[int, bool]:
    """A counter that only moves when the visible page changes.

    It becomes part of the keyframe name, so a mode switch replays the fade-in while a
    live page that repaints every second keeps the same name and stays still.
    """
    state = st.session_state
    changed = state.get("_page_slug") != slug
    if changed:
        state["_page_slug"] = slug
        state["_page_token"] = int(state.get("_page_token", 0)) + 1
    return state["_page_token"], changed


def scroll_to_top() -> None:
    """A mode switch starts at the top of the new page instead of at the old offset."""
    try:
        import streamlit.components.v1 as components

        components.html(
            "<script>try{const w=window.parent;w.scrollTo(0,0);"
            "const m=w.document.querySelector('[data-testid=\"stMain\"]');"
            "if(m){m.scrollTop=0}}catch(e){}</script>",
            height=0,
        )
    except Exception:                               # never let a cosmetic touch break the page
        pass


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
    store = common.get_case_store(str(common.cases_path()))
    mode = render_sidebar(store)
    if mode != REPLAY:
        st.session_state["playing"] = False         # a replay does not keep running in the background

    slug = SLUG[mode]
    token, changed = page_entry(slug)
    if changed:
        scroll_to_top()
    st.markdown(ui.page_animation(token, slug), unsafe_allow_html=True)

    with st.container(key=f"page-{slug}"):
        if mode == REPLAY:
            mode_replay.render(header, tag, run)
        elif mode == LIVE:
            mode_live.render(header, tag, run, engine)
        elif mode == CASES:
            mode_cases.render(header, store)
        else:
            mode_testing.render(header, tag, run, engine)


main()
