"""The case page: one investigation, with RiskPilot beside it.

This used to be a pop-up. A pop-up is the right shape for "show me this alert",
but an investigation is a place you stay: you read the evidence, write notes,
ask questions, and send a change for approval. So the case has its own page, the
alert's "Open investigation case" button lands here, and the assistant sits in
the right-hand column instead of under the form.

The page keeps the same rule as everything else in this app:

    RiskPilot may read anything and prepare anything; a person approves

Two ways to ask for a change therefore exist and they meet in the same queue:
the panel's "Run investigation -> Send for approval" (the rules recommend), and
the chat ("mark it false positive") where the analyst asks. In both cases a
proposal is filed and the Approve button -- here, beside the message -- is the
only thing that touches the case.
"""
from __future__ import annotations

import re

import streamlit as st
import streamlit.components.v1 as components

import ui
from common import agent_for, chats, pilot_for, proposals, toolkit
from src.agent import approve, reject
from src.cases import PRIORITIES, PRIORITY_LABELS, STATUSES, STATUS_LABELS
from src.tools import Toolkit, ToolError

CHAT_ROWS = 60                                  # messages on screen; the store keeps all
CHAT_HEIGHT = 520                               # px of transcript, its own scroll window
CHIPS = ("Why was it flagged?", "What is related?", "What do you recommend?")


def _back() -> None:
    """Back to the list, with the selection forgotten."""
    st.session_state.mode = "Cases"
    st.session_state["selected_case"] = None


def _token(case) -> str:
    """A key suffix that changes whenever the case does.

    The editor widgets are keyed with it, so a change made elsewhere -- the chat
    approving a proposal, another tab -- rebuilds them from the stored values
    instead of leaving a stale selection on screen.
    """
    return re.sub(r"\W", "", case.updated_at or "")


# --------------------------------------------------------------------------- #
# Editing the case
# --------------------------------------------------------------------------- #
def _editor(case, tools: Toolkit) -> None:
    """Resolution, priority and notes, with the same rules as everywhere else."""
    token = _token(case)
    status = st.selectbox("Status", STATUSES, index=STATUSES.index(case.status),
                          format_func=lambda s: STATUS_LABELS[s],
                          key=f"case_page_status_{case.number}_{token}",
                          help="Investigating while you work; Confirmed fraud or False "
                               "positive once you decided.")

    with st.form(f"case_page_form_{case.number}_{token}"):
        priority = st.selectbox("Priority", PRIORITIES, index=PRIORITIES.index(case.priority),
                                format_func=lambda p: PRIORITY_LABELS[p])
        notes = st.text_area("Analyst notes", value=case.notes, height=170,
                             placeholder="What did you check, what did you find, what did you decide?")
        saved = saved_closed = False
        if case.closed:
            saved = st.form_submit_button("Save case")       # a closed case is edited, not re-closed
        else:
            buttons = st.columns(2)
            saved = buttons[0].form_submit_button("Save case")
            # Closing records the decision: with the resolution still
            # "Investigating" there is nothing to close the case with.
            saved_closed = buttons[1].form_submit_button("Save and close", type="primary",
                                                         disabled=status == "investigating")

    if saved or saved_closed:
        try:
            updated = tools.update_case(
                case.case_id, status=status, notes=notes, priority=priority,
                state="closed" if saved_closed and not case.closed else None).case
        except ToolError as exc:                          # the store's vocabulary changed
            st.error(str(exc))
            return
        st.session_state.pop(f"investigation_{case.number}", None)   # that reading is stale now
        st.toast(f"{updated.case_id} saved · {updated.state_label}")
        st.rerun()

    if case.closed:
        if st.button("Reopen case", key=f"case_page_reopen_{case.number}_{token}",
                     help="Puts the case back to work: open again, with the same resolution."):
            tools.update_case(case.case_id, state="open")
            st.rerun()


# --------------------------------------------------------------------------- #
# The investigation panel (deterministic rules -> proposal)
# --------------------------------------------------------------------------- #
def _investigation_panel(case, tag: str) -> None:
    """Run the rules, read the evidence, send the recommendation for approval.

    Nothing this section does can change the case: "Run" only reads, and "Send
    for approval" writes a proposal. The case moves on the Approvals page or in
    the chat, when a person says so.
    """
    store = proposals()
    pending = store.pending_for(case.number)
    st.markdown('<div class="fa-sub2" style="margin-top:4px">Investigation assistant</div>',
                unsafe_allow_html=True)
    if pending is not None:
        st.markdown(ui.proposal_card_html(pending, show_case=False), unsafe_allow_html=True)
        st.caption("Waiting for a decision. The case has not been changed.")
        return

    key = f"investigation_{case.number}"
    if st.button("Run investigation", key=f"run_agent_{case.number}",
                 icon=":material/travel_explore:", help="Reads the case, compares it with the "
                 "live feed, asks the Risk Manager and recommends an action. It cannot edit the case."):
        try:
            st.session_state[key] = agent_for(toolkit(tag)).investigate(case.number)
        except ToolError as exc:
            st.error(str(exc))
            return

    result = st.session_state.get(key)
    if result is None:
        st.caption("RiskPilot gathers the evidence an analyst would want, then recommends an "
                   "action. You approve it. Nothing is changed until you do.")
        return

    st.markdown(ui.investigation_html(result), unsafe_allow_html=True)
    left, right = st.columns([1, 1])
    if result.recommendation.changes_anything:
        if left.button("Send for approval", key=f"file_agent_{case.number}", type="primary"):
            try:
                filed = agent_for(toolkit(tag)).file(result)
            except ToolError as exc:
                st.error(str(exc))
                return
            st.session_state.pop(key, None)
            st.toast(f"Proposal {filed.number} sent for approval")
            st.rerun()
    else:
        # A recommendation to do what is already true needs no approver; the
        # reading above is still useful, so it stays on screen.
        left.caption("No decision needed: the case already matches this recommendation.")
    if right.button("Discard", key=f"discard_agent_{case.number}"):
        st.session_state.pop(key, None)
        st.rerun()


# --------------------------------------------------------------------------- #
# The conversation (RiskPilot)
# --------------------------------------------------------------------------- #
def _inline_proposal(proposal, queue, tools: Toolkit, transcript, case) -> None:
    """A proposal the analyst asked for in chat, with its own Approve/Reject."""
    st.markdown(ui.proposal_card_html(proposal, show_case=False), unsafe_allow_html=True)
    if not proposal.pending:
        st.markdown(f'<div class="fa-note">Proposal {proposal.number} &middot; '
                    f'{ui.escape(proposal.status_label.lower())}'
                    + (f' by {ui.escape(proposal.decided_by)}' if proposal.decided_by else '')
                    + '.</div>', unsafe_allow_html=True)
        return

    override = False
    if proposal.requires_override:
        override = st.checkbox("I have read this and want to override the policy",
                               key=f"chat_override_{proposal.number}")
    left, right = st.columns(2)
    if left.button("Approve", key=f"chat_approve_{proposal.number}", type="primary",
                   width="stretch", disabled=proposal.requires_override and not override):
        try:
            result = approve(queue, tools, proposal, by="analyst", override=override)
        except ToolError as exc:
            st.error(str(exc))
            return
        transcript.append(case.number, "system",
                          f"Approved · {result.case.case_id} is now "
                          f"{result.case.state_label.lower()}, resolution "
                          f"{result.case.status_label.lower()}, "
                          f"{result.case.priority_label.lower()} priority.")
        st.rerun()
    if right.button("Reject", key=f"chat_reject_{proposal.number}", width="stretch"):
        try:
            reject(queue, proposal, by="analyst")
        except ToolError as exc:
            st.error(str(exc))
            return
        transcript.append(case.number, "system",
                          f"Rejected · proposal {proposal.number} dropped, "
                          f"{case.case_id} unchanged.")
        st.rerun()


def _scroll_chat() -> None:
    """Keep the newest turn in view without the analyst having to scroll."""
    try:
        components.html(
            "<script>try{const d=window.parent.document;"
            "const el=d.querySelector('.st-key-chat_scroll');if(el){"
            "[el,...el.querySelectorAll('div')].forEach(function(n){"
            "if(n.scrollHeight>n.clientHeight+4){n.scrollTop=n.scrollHeight;}});}"
            "}catch(e){}</script>", height=0)
    except Exception:                               # a cosmetic touch must never break the page
        pass


def _chat(case, tools: Toolkit, pilot) -> None:
    """Header, transcript, quick prompts and the input box."""
    queue, transcript = proposals(), chats()
    st.markdown(ui.chat_head_html(pilot.name), unsafe_allow_html=True)

    messages = transcript.history(case.number, limit=CHAT_ROWS)
    with st.container(height=CHAT_HEIGHT, key="chat_scroll", border=False):
        if not messages:
            st.markdown(ui.chat_intro_html(case.case_id), unsafe_allow_html=True)
        for message in messages:
            st.markdown(ui.chat_message_html(message, pilot.name), unsafe_allow_html=True)
            action = message.action or {}
            if message.role == "assistant" and action.get("kind") == "proposal":
                found = queue.get(int(action["proposal"])) if action.get("proposal") else None
                if found is not None:
                    _inline_proposal(found, queue, tools, transcript, case)

    if not messages:                                # first visit: make the usefulness obvious
        columns = st.columns(len(CHIPS))
        for column, text in zip(columns, CHIPS):
            if column.button(text, key=f"chat_chip_{abs(hash(text)) % 10**8}"):
                pilot.converse(case.number, text)
                st.rerun()

    message = st.chat_input(f"Ask {pilot.name} about {case.case_id}...",
                            key=f"chat_input_{case.number}")
    if message:
        pilot.converse(case.number, message)
        st.rerun()
    _scroll_chat()


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #
def render(header, store, tag: str = "") -> None:
    """Draw the case page. `store` is the shared SQLite case store."""
    number = st.session_state.get("selected_case")
    case = store.get(number) if number is not None else None
    if case is None:
        st.markdown('<div class="fa-root"><div class="fa-card fa-empty"><h3>No case selected</h3>'
                    '<p>Pick one from the cases list.</p></div></div>', unsafe_allow_html=True)
        if st.button("All cases", key="case_none_back", icon=":material/arrow_back:"):
            _back()
            st.rerun()
        return

    tools = toolkit(tag) if tag else Toolkit(cases=store)
    pilot = pilot_for(tools)

    nav = st.columns([0.22, 0.78])
    if nav[0].button("All cases", key="case_back", icon=":material/arrow_back:"):
        _back()
        st.rerun()
    nav[1].markdown(ui.header_html(
        "case", f"transaction #{int(case.txn_id)} · {case.tx_type.replace('_', ' ').lower() or 'unknown'} · "
                f"{case.priority_label.lower()} priority", f"Case {case.case_id}"),
        unsafe_allow_html=True)

    left, right = st.columns([1.55, 1], gap="medium")
    with left:
        st.markdown(ui.case_detail_html(case, head=False), unsafe_allow_html=True)
        _editor(case, tools)
    with right:
        _chat(case, tools, pilot)

    explain, trail = st.columns([1.55, 1], gap="medium")
    with explain:
        if tag:
            _investigation_panel(case, tag)
    with trail:
        events = store.events(case.number)
        st.markdown('<div class="fa-sub2" style="margin-top:4px">Investigation history</div>',
                    unsafe_allow_html=True)
        if events:
            st.markdown(ui.case_history_html(events), unsafe_allow_html=True)
        else:
            st.markdown('<div class="fa-note">Nothing recorded yet.</div>',
                        unsafe_allow_html=True)
