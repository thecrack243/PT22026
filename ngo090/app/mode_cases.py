"""Investigation cases: the analyst's durable record of an alert worth keeping.

This page is not a simulation mode: it works on what the analyst saved earlier
and survives restarts, so it reads from SQLite rather than from session state.
Creating a case happens where the alert is, in the live feed's pop-up.
"""
from __future__ import annotations

import streamlit as st

import ui
from src.cases import PRIORITIES, PRIORITY_LABELS, STATUSES, STATUS_LABELS, CaseStore, case_id

ALL = "All"


def _open(number: int) -> None:
    """Remember which case the analyst clicked; the pop-up stays up until it is closed."""
    st.session_state.case_to_open = int(number)


def _close_case() -> None:
    st.session_state.case_to_open = None


def _clear_filters() -> None:
    st.session_state.case_status = ALL
    st.session_state.case_priority = ALL
    st.session_state.case_search = ""


def render_filters() -> tuple[str, str, str]:
    cols = st.columns([1.1, 1.1, 2.4, 0.8], vertical_alignment="bottom")
    with cols[0]:
        status = st.selectbox("Status", (ALL, *STATUSES),
                              format_func=lambda s: s if s == ALL else STATUS_LABELS[s], key="case_status")
    with cols[1]:
        priority = st.selectbox("Priority", (ALL, *PRIORITIES),
                                format_func=lambda p: p if p == ALL else PRIORITY_LABELS[p],
                                key="case_priority")
    with cols[2]:
        search = st.text_input("Search", key="case_search",
                               placeholder="Case id (C-0007) or transaction id (1234)")
    with cols[3]:
        # The reset has to happen in a callback: callbacks run before the widgets
        # are instantiated on the next pass, whereas writing to a widget's session
        # state key after that widget was created raises StreamlitAPIException.
        st.button("Clear", key="case_clear", width="stretch", on_click=_clear_filters)
    return status, priority, search


def open_case(number: int, store: CaseStore) -> None:
    """Pop-up with the case, its status, priority and notes."""

    @st.dialog(f"Case {case_id(number)}", width="large")
    def _show() -> None:
        case = store.get(number)
        if case is None:                            # deleted in another tab
            st.info("This case no longer exists.")
            return
        st.markdown(ui.case_detail_html(case), unsafe_allow_html=True)

        with st.form(f"case_form_{case.number}"):
            cols = st.columns(2)
            status = cols[0].selectbox("Status", STATUSES, index=STATUSES.index(case.status),
                                       format_func=lambda s: STATUS_LABELS[s],
                                       help="Open, investigating, then one of the three closing states.")
            priority = cols[1].selectbox("Priority", PRIORITIES, index=PRIORITIES.index(case.priority),
                                         format_func=lambda p: PRIORITY_LABELS[p])
            notes = st.text_area("Analyst notes", value=case.notes, height=170,
                                 placeholder="What did you check, what did you find, what did you decide?")
            submitted = st.form_submit_button("Save case", type="primary")

        if submitted:
            updated = store.update(case.case_id, status=status, notes=notes, priority=priority)
            st.session_state["case_to_open"] = updated.number   # the pop-up shows the saved state
            st.toast(f"{updated.case_id} saved · {updated.status_label}")
            st.rerun()

        st.button("Close case", key=f"close_case_{case.number}", on_click=_close_case, width="stretch")

    _show()


def render(header, store: CaseStore) -> None:
    """Draw the cases page. `store` is the shared SQLite store."""
    counts = store.counts()
    header.markdown(
        ui.header_html("cases", "Every alert an analyst kept, with its status, priority and notes",
                       "Investigation cases"),
        unsafe_allow_html=True)

    st.markdown(ui.cases_overview_html(counts), unsafe_allow_html=True)

    status, priority, search = render_filters()
    cases = store.list(status=None if status == ALL else status,
                       priority=None if priority == ALL else priority,
                       search=search)

    if counts["all"] == 0:
        st.markdown(ui.cases_empty_html(), unsafe_allow_html=True)
        return
    if not cases:
        st.markdown('<div class="fa-note">No case matches these filters.</div>', unsafe_allow_html=True)
        return

    st.markdown(f'<div class="fa-note" style="margin:2px 2px 8px">{len(cases):,} case(s), '
                f'most recently updated first</div>', unsafe_allow_html=True)
    with st.container(border=True):
        for case in cases:
            st.button(ui.case_label(case), key=f"case_{case.number}", on_click=_open, args=(case.number,))

    # The pop-up has to be re-created on every run while it is open, otherwise
    # editing the form inside it (a full rerun) would throw it away.
    open_id = st.session_state.get("case_to_open")
    if open_id is not None and hasattr(st, "dialog"):
        open_case(int(open_id), store)
    elif open_id is not None:                       # older Streamlit: no pop-up support
        st.info(f"Case {case_id(int(open_id))} is saved. Open the pop-up in a Streamlit version that supports it.")
