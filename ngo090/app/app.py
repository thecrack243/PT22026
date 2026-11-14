"""FRAUDINTEL: risk monitoring, simulation and testing around one risk engine.

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

import mode_agent                   # noqa: E402
import mode_audit                  # noqa: E402
import mode_case                   # noqa: E402
import mode_cases                  # noqa: E402
import mode_live                   # noqa: E402
import mode_replay                 # noqa: E402
import mode_testing                # noqa: E402
import ui                          # noqa: E402
from src.artifacts import discover_runs  # noqa: E402

REPLAY, LIVE, TESTING = "Historical replay", "Live simulation", "Transaction testing"
CASES = "Cases"                     # the list of cases: a workspace page, not a mode
CASE = "Case"                       # one case, opened from the list or from an alert
AUDIT = "Audit"                     # the decision log: workspace too
APPROVALS = "Approvals"             # the assistant's queue: workspace too
MODES = (LIVE, REPLAY, TESTING)    # the live feed is what greets the analyst
SLUG = common.PAGE_SLUGS           # the same slug the URL uses, so the two cannot drift
ICON = {LIVE: ":material/sensors:", REPLAY: ":material/history:", TESTING: ":material/science:",
        CASES: ":material/folder_shared:", CASE: ":material/description:",
        AUDIT: ":material/receipt_long:", APPROVALS: ":material/fact_check:"}

st.set_page_config(page_title="FRAUDINTEL", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")


def _switch_to(mode: str) -> None:
    """Runs before the page is drawn, so the sidebar and the page agree in one pass.

    Pop-ups belong to the page they were opened on: leaving the page dismisses
    them, so they do not reappear when the analyst comes back later. The same
    goes for the open case: leaving the workspace forgets which case it was.
    """
    st.session_state.mode = mode
    st.session_state["alert_to_open"] = None     # live alert pop-up (and its pending trigger)
    if mode != CASE:
        st.session_state["selected_case"] = None


def render_sidebar(store) -> str:
    """The three modes live in the sidebar; the page itself stays one canvas."""
    if "mode" not in st.session_state:              # a fresh session: first visit or a reload
        mode, case = common.restore_page()          # the URL remembers where the analyst was
        st.session_state.mode = mode
        if case is not None:
            st.session_state["selected_case"] = case
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
        open_n = store.counts()["open"]             # not-closed cases: what can still be worked on
        label = f"Cases · {open_n}" if open_n else CASES
        # The case page is a subpage of Cases, so Cases stays highlighted there.
        st.button(label, key=f"nav_{SLUG[CASES]}", icon=ICON[CASES], width="stretch",
                  type="primary" if st.session_state.mode in (CASES, CASE) else "secondary",
                  on_click=_switch_to, args=(CASES,),
                  help="Alerts an analyst kept, stored in a local SQLite database.")
        st.button(AUDIT, key=f"nav_{SLUG[AUDIT]}", icon=ICON[AUDIT], width="stretch",
                  type="primary" if st.session_state.mode == AUDIT else "secondary",
                  on_click=_switch_to, args=(AUDIT,),
                  help="Every decision ever made, with the model and policy version behind it.")
        waiting = common.proposals().counts()["pending"]
        st.button(f"{APPROVALS} · {waiting}" if waiting else APPROVALS,
                  key=f"nav_{SLUG[APPROVALS]}", icon=ICON[APPROVALS], width="stretch",
                  type="primary" if st.session_state.mode == APPROVALS else "secondary",
                  on_click=_switch_to, args=(APPROVALS,),
                  help="What the investigation assistant recommends. Nothing changes until a "
                       "person approves it.")
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
    number = st.session_state.get("selected_case")
    if mode == CASE and (number is None or store.get(number) is None):
        mode = st.session_state.mode = CASES        # a stale case link lands on the list
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
            mode_cases.render(header, store, tag)
        elif mode == CASE:
            mode_case.render(header, store, tag)
        elif mode == AUDIT:
            mode_audit.render(header, store)
        elif mode == APPROVALS:
            mode_agent.render(header, store)
        else:
            mode_testing.render(header, tag, run, engine)

    # The page just drawn becomes the one the URL names, so a reload -- or a
    # bookmark, or a fresh tab on the same link -- opens it again.
    common.remember_page(mode, st.session_state.get("selected_case"))


main()
