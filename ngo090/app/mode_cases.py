"""Investigation cases: the analyst's durable record of an alert worth keeping.

This page is not a simulation mode: it works on what the analyst saved earlier
and survives restarts, so it reads from SQLite rather than from session state.
Creating a case happens where the alert is, in the live feed's pop-up.
"""
from __future__ import annotations

import streamlit as st

import ui
from common import cases_path, get_case_store
from src.cases import PRIORITIES, PRIORITY_LABELS, STATUSES, STATUS_LABELS, CaseStore, case_id
from src.tools import Toolkit, ToolError

ALL = "All"
STATES = ("Open", "Closed")


def _open(number: int) -> None:
    """Remember which case the analyst clicked; the pop-up stays up until dismissed."""
    st.session_state.case_to_open = int(number)


def _popup_dismissed() -> None:
    """The X, a filter change or a mode change all put the pop-up away."""
    st.session_state.case_to_open = None


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
                              key="case_status", on_change=_popup_dismissed)
    with cols[1]:
        state = st.selectbox("State", (ALL, *STATES), key="case_state", on_change=_popup_dismissed)
    with cols[2]:
        priority = st.selectbox("Priority", (ALL, *PRIORITIES),
                                format_func=lambda p: p if p == ALL else PRIORITY_LABELS[p],
                                key="case_priority", on_change=_popup_dismissed)
    with cols[3]:
        search = st.text_input("Search", key="case_search", on_change=_popup_dismissed,
                               placeholder="Case id (C-0007) or transaction id (1234)")
    with cols[4]:
        # The reset has to happen in a callback: callbacks run before the widgets
        # are instantiated on the next pass, whereas writing to a widget's session
        # state key after that widget was created raises StreamlitAPIException.
        st.button("Clear", key="case_clear", width="stretch", on_click=_clear_filters)
    return status, state, priority, search


def open_case(number: int, store: CaseStore) -> None:
    """Pop-up with the case, its resolution, priority and notes."""
    case = store.get(number)
    title = f"Case {case_id(number)}" + (" · Closed" if case is not None and case.closed else "")
    tools = Toolkit(cases=store)                    # the same update_case an agent calls

    @st.dialog(title, width="large")
    def _show() -> None:
        case = store.get(number)
        if case is None:                            # deleted in another tab
            st.info("This case no longer exists.")
            return

        # Two columns: the case and its editable fields on the left, the audit
        # trail on the right -- side by side, so the history is never something
        # the analyst has to scroll down to.
        left, right = st.columns([1.55, 1], gap="medium")

        with left:
            st.markdown(ui.case_detail_html(case), unsafe_allow_html=True)

            # Status lives OUTSIDE the form on purpose: a form only hands its
            # widgets' values over when it is submitted, so "Save and close" would
            # never learn that the analyst picked a resolution and stay disabled.
            # Out here the pop-up (a fragment) reruns on every change instead.
            status = st.selectbox("Status", STATUSES, index=STATUSES.index(case.status),
                                  format_func=lambda s: STATUS_LABELS[s],
                                  key=f"case_popup_status_{number}",
                                  help="Investigating while you work; Confirmed fraud or "
                                       "False positive once you decided.")

            with st.form(f"case_form_{case.number}"):
                priority = st.selectbox("Priority", PRIORITIES, index=PRIORITIES.index(case.priority),
                                        format_func=lambda p: PRIORITY_LABELS[p])
                notes = st.text_area("Analyst notes", value=case.notes, height=170,
                                     placeholder="What did you check, what did you find, what did you decide?")
                saved = False
                saved_closed = False
                if case.closed:
                    saved = st.form_submit_button("Save case")   # a closed case is only edited, not re-closed
                else:
                    btn = st.columns(2)
                    saved = btn[0].form_submit_button("Save case")
                    # Closing records the decision: while the resolution is still
                    # "Investigating" there is nothing to close the case with.
                    saved_closed = btn[1].form_submit_button("Save and close", type="primary",
                                                             disabled=status == "investigating")

            if case.closed:
                # A plain on_click callback would only rerun this fragment, leaving
                # the pop-up title and the sidebar badge stale; handling the click
                # here (like the save buttons) lets the st.rerun below refresh all.
                if st.button("Reopen case", key=f"reopen_case_{case.number}",
                             help="Puts the case back to work: open again, with the same resolution."):
                    tools.update_case(case.case_id, state="open")
                    st.rerun()

        if saved or saved_closed:
            try:
                updated = tools.update_case(
                    case.case_id, status=status, notes=notes, priority=priority,
                    state="closed" if saved_closed and not case.closed else None).case
            except ToolError as exc:                 # the store's vocabulary changed under us
                st.error(str(exc))
                return
            st.session_state["case_to_open"] = updated.number   # the pop-up shows the saved state
            st.toast(f"{updated.case_id} saved · {updated.state_label}")
            st.rerun()

        with right:
            # The audit trail: everything that ever happened to this case,
            # newest first. It only grows -- edits overwrite the case, never
            # the history -- and it lives beside the case, not under it.
            events = store.events(case.number)
            st.markdown('<div class="fa-sub2" style="margin-top:2px">Investigation history</div>',
                        unsafe_allow_html=True)
            if events:
                st.markdown(ui.case_history_html(events), unsafe_allow_html=True)
            else:
                st.markdown('<div class="fa-note">Nothing recorded yet.</div>',
                            unsafe_allow_html=True)

    _show()


def render(header, store: CaseStore) -> None:
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

    # The pop-up has to be re-created on every run while it is open, otherwise
    # editing the form inside it (a full rerun) would throw it away.
    open_id = st.session_state.get("case_to_open")
    if open_id is not None and hasattr(st, "dialog"):
        open_case(int(open_id), store)
    elif open_id is not None:                       # older Streamlit: no pop-up support
        st.info(f"Case {case_id(int(open_id))} is saved. Open the pop-up in a Streamlit version that supports it.")
