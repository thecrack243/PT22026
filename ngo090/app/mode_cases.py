"""Investigation cases: the analyst's durable record of an alert worth keeping.

This page is the list. A case itself has its own page (`app/mode_case.py`),
which is where the work happens: reading the evidence, writing notes, asking
RiskPilot and approving a change. Clicking a card here goes there.

Both this page and the case page read from SQLite rather than session state, so
they survive restarts. Creating a case happens where the alert is, in the live
feed's pop-up.
"""
from __future__ import annotations

import streamlit as st

import ui
from src.cases import PRIORITIES, PRIORITY_LABELS, STATUSES, STATUS_LABELS, CaseStore

ALL = "All"
STATES = ("Open", "Closed")


def _open(number: int) -> None:
    """A card was clicked: the case page opens on the next run."""
    st.session_state["selected_case"] = int(number)
    st.session_state.mode = "Case"


def _clear_filters() -> None:
    st.session_state.case_status = ALL
    st.session_state.case_state = ALL
    st.session_state.case_priority = ALL
    st.session_state.case_search = ""


def render_filters() -> tuple[str, str, str, str]:
    cols = st.columns([1.05, 0.75, 0.95, 2.0, 0.75], gap="medium", vertical_alignment="bottom")
    with cols[0]:
        status = st.selectbox("Status", (ALL, *STATUSES),
                              format_func=lambda s: s if s == ALL else STATUS_LABELS[s],
                              key="case_status")
    with cols[1]:
        state = st.selectbox("State", (ALL, *STATES), key="case_state")
    with cols[2]:
        priority = st.selectbox("Priority", (ALL, *PRIORITIES),
                                format_func=lambda p: p if p == ALL else PRIORITY_LABELS[p],
                                key="case_priority")
    with cols[3]:
        search = st.text_input("Search", key="case_search",
                               placeholder="Case id (C-0007) or transaction id (1234)")
    with cols[4]:
        # The reset has to happen in a callback: callbacks run before the widgets
        # are instantiated on the next pass, whereas writing to a widget's session
        # state key after that widget was created raises StreamlitAPIException.
        st.button("Clear", key="case_clear", width="stretch", on_click=_clear_filters)
    return status, state, priority, search


def render(header, store: CaseStore, tag: str = "") -> None:
    """Draw the cases page. `store` is the shared SQLite store."""
    counts = store.counts()
    header.markdown(
        ui.header_html("cases", "Every alert an analyst kept, with its status, priority and notes",
                       "Investigation cases"),
        unsafe_allow_html=True)

    st.markdown(ui.cases_overview_html(counts), unsafe_allow_html=True)

    status, state, priority, search = render_filters()
    cases = store.list(status=None if status == ALL else status,
                       state=None if state == ALL else state,
                       priority=None if priority == ALL else priority,
                       search=search)

    if counts["all"] == 0:
        st.markdown(ui.cases_empty_html(), unsafe_allow_html=True)
        return
    if not cases:
        st.markdown('<div class="fa-note">No case matches these filters.</div>', unsafe_allow_html=True)
        return

    with st.container(border=True, key="case_list"):
        for case in cases:
            # The card is the button: an invisible native button (keyed
            # case_row_N) first, then the HTML card, which the CSS slides up
            # to exactly cover it. The whole card is clickable, the click
            # stays a normal session click -- no reload, no lost filters.
            st.button("Open", key=f"case_row_{case.number}", on_click=_open, args=(case.number,))
            st.markdown(ui.case_card_html(case), unsafe_allow_html=True)
