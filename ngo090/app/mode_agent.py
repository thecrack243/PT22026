"""Workspace: the assistant's recommendations, and the person who decides them.

This page is the other half of `src/agent.py`. The assistant investigates a case
and files a proposal; nothing happens until someone sits here and reads it. That
separation is the point: the same screen shows what the agent recommended, what
the Risk Manager said, and what approving would actually change -- so saying yes
is a decision rather than a formality.

Rejecting is a first-class outcome. A rejected proposal keeps its row, so the
record shows that a person looked and disagreed, which is exactly what an audit
is for.
"""
from __future__ import annotations

import streamlit as st

import ui
from common import get_case_store, cases_path, proposals
from src.agent import approve, reject
from src.tools import Toolkit, ToolError

TAB_PENDING, TAB_DECIDED = "Awaiting approval", "Decided"


def _counts_html(counts: dict) -> str:
    pills = [
        ("bell", "Awaiting approval", counts.get("pending", 0), "#fbbf24"),
        ("check", "Approved", counts.get("approved", 0), "#4ade80"),
        ("alert", "Rejected", counts.get("rejected", 0), "#94a3b8"),
        ("clock", "Superseded", counts.get("superseded", 0), "#60a5fa"),
    ]
    body = "".join(
        f'<div class="fa-pill">{ui.icon(icon_name, color, 17)}<span>{label}</span>'
        f'<b>{int(value):,}</b></div>' for icon_name, label, value, color in pills)
    return f'<div class="fa-root"><div class="fa-pills2">{body}</div></div>'


def _decide(kind: str, number: int, override: bool, note: str) -> None:
    """Approve or reject one proposal, then reload the page so counts agree."""
    queue = proposals()
    tools = Toolkit(cases=get_case_store(str(cases_path())))
    item = queue.get(number)
    if item is None:
        st.error("That proposal no longer exists.")
        return
    try:
        if kind == "approve":
            result = approve(queue, tools, item, by="analyst", note=note, override=override)
            st.toast(f"{result.case.case_id} updated · {result.case.status_label} · "
                     f"{result.case.state_label}")
        else:
            reject(queue, item, by="analyst", note=note)
            st.toast(f"Proposal {number} rejected · the case is unchanged")
    except ToolError as exc:
        st.error(str(exc))
        return
    st.rerun()


def _proposal_block(proposal) -> None:
    """One card, its approve/reject controls, and the rest of its reasoning."""
    with st.container(border=True):
        st.markdown(ui.proposal_card_html(proposal), unsafe_allow_html=True)

        note = st.text_input("Why (optional)", key=f"prop_note_{proposal.number}",
                             placeholder="Added to the case record when you approve it.")
        override = False
        if proposal.requires_override:
            override = st.checkbox(
                "I have read this and want to override the policy",
                key=f"prop_override_{proposal.number}",
                help="Without this tick the approval is refused, so an override cannot happen "
                     "by muscle memory.")

        left, middle, right = st.columns([1, 1, 2.2])
        approve_key = f"prop_approve_{proposal.number}"
        if left.button("Approve", key=approve_key, type="primary", width="stretch",
                       disabled=proposal.requires_override and not override):
            _decide("approve", proposal.number, override, note)
        if middle.button("Reject", key=f"prop_reject_{proposal.number}", width="stretch"):
            _decide("reject", proposal.number, False, note)
        right.caption(f"Filed {ui._stamp(proposal.created_at)}"
                      + (f" · {proposal.agent}" if proposal.agent else ""))

        with st.expander("What the assistant found", expanded=False):
            if proposal.narrative:
                st.markdown(f'<div class="fa-root"><div class="fa-narr">'
                            f'{ui.escape(proposal.narrative)}</div></div>', unsafe_allow_html=True)
            evidence = (proposal.evidence or {}).get("evidence", [])
            if evidence:
                rows = "".join(
                    f'<div class="fa-ev {ui.escape(e.get("kind", "fact"))}"><i></i>'
                    f'<div>{ui.escape(e.get("text", ""))}'
                    f'<small>{ui.escape(e.get("source", ""))}</small></div></div>'
                    for e in evidence)
                st.markdown(f'<div class="fa-root"><div class="fa-evlist">{rows}</div></div>',
                            unsafe_allow_html=True)
            for gap in (proposal.evidence or {}).get("gaps", []):
                st.markdown(f'<div class="fa-root"><div class="fa-gaps">{ui.escape(gap)}</div></div>',
                            unsafe_allow_html=True)
            steps = (proposal.evidence or {}).get("steps", [])
            if steps:
                trace = "".join(
                    f'<div class="fa-trow"><span class="n">{n}</span>'
                    f'<span class="k">{ui.escape(s.get("label", ""))}</span>'
                    f'<span class="d">{ui.escape(s.get("detail", ""))}</span>'
                    f'<span class="ms"></span></div>' for n, s in enumerate(steps, 1))
                st.markdown(f'<div class="fa-root"><div class="fa-trace">{trace}</div></div>',
                            unsafe_allow_html=True)


def _decided_list(items) -> None:
    """A plain table of what was decided, so approvals are auditable at a glance."""
    if not items:
        st.markdown('<div class="fa-note">Nothing has been decided yet.</div>',
                    unsafe_allow_html=True)
        return
    rows = "".join(
        f'<tr><td>{ui.escape(p.case_id)}</td>'
        f'<td><span class="fa-tag {p.kind}">{ui.escape(p.kind_label)}</span></td>'
        f'<td><span class="fa-tag {p.status}">{ui.escape(p.status_label)}</span></td>'
        f'<td>{p.risk:.1f}</td>'
        f'<td>{ui._stamp(p.decided_at or "")}</td>'
        f'<td>{ui.escape(p.decided_by or "")}</td>'
        f'<td>{ui.escape(p.decision_note or "")}</td></tr>' for p in items)
    st.markdown(
        '<div class="fa-root"><div class="fa-tablewrap"><table class="fa-table">'
        '<thead><tr><th>Case</th><th>Recommendation</th><th>Decision</th><th>Risk</th>'
        '<th>When</th><th>By</th><th>Note</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div></div>', unsafe_allow_html=True)


def render(header, store=None) -> None:
    """Draw the approvals page."""
    header.markdown(ui.header_html(
        "approvals", "What the assistant recommends, and whether a person agreed",
        "Approvals"), unsafe_allow_html=True)

    queue = proposals()
    counts = queue.counts()
    st.markdown(_counts_html(counts), unsafe_allow_html=True)

    pending = queue.pending()
    decided = [p for p in queue.list(limit=200) if not p.pending]

    if counts["all"] == 0:
        st.markdown('<div class="fa-root"><div class="fa-card fa-empty">'
                    '<h3>No recommendations yet</h3><p>Open a case in the Cases page and run the '
                    'investigation assistant. Its recommendations arrive here, and nothing is '
                    'changed until you approve them.</p></div></div>', unsafe_allow_html=True)
        return

    if pending:
        st.markdown(f'<div class="fa-root"><div class="fa-note" style="margin:6px 2px 12px">'
                    f'{len(pending)} proposal(s) waiting · oldest first. The case is untouched '
                    f'until you approve one.</div></div>', unsafe_allow_html=True)
        for proposal in pending:
            _proposal_block(proposal)
    else:
        st.markdown('<div class="fa-root"><div class="fa-note" style="margin:6px 2px 16px">'
                    'Nothing is waiting for approval.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="fa-sub2" style="margin-top:22px">Decided</div>', unsafe_allow_html=True)
    _decided_list(decided[:25])
