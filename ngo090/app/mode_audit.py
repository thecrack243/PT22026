"""Workspace: the audit log -- every decision, and the policy that produced it.

The live page shows what is happening now, and forgets. This page shows what
happened, permanently: one row per transaction with its score, its decision,
the model version and the policy version in force at that second, so any
decision can be explained long after the feed has moved on.

Nothing here is editable. That is the point.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import ui
from common import audits, cases_path, get_case_store
from src.audit import DECISION_LABELS

ALL = "All"
DECISIONS = ("block", "review", "approve")
LATEST = 500            # rows shown in the table; the log itself keeps everything


def _clear_filters() -> None:
    st.session_state.audit_decision = ALL
    st.session_state.audit_policy = ALL
    st.session_state.audit_search = ""


def _sim_time(minute: int) -> str:
    return f"D{minute // 1440 + 1} {(minute // 60) % 24:02d}:{minute % 60:02d}"


def render_filters(policies: list[dict]) -> tuple[str, str, str]:
    """Decision, policy version and a transaction search, with a reset."""
    cols = st.columns([0.85, 1.05, 1.7, 0.7], gap="medium", vertical_alignment="bottom")
    with cols[0]:
        decision = st.selectbox("Decision", (ALL, *DECISIONS), key="audit_decision",
                                format_func=lambda d: d if d == ALL else DECISION_LABELS[d],
                                help="Blocked, sent to review, or approved without action.")
    with cols[1]:
        options = (ALL, *[p["hash"] for p in policies])
        policy = st.selectbox("Policy version", options, key="audit_policy",
                              format_func=lambda h: h if h == ALL else f"policy {h}",
                              help="Every distinct set of thresholds gets its own version.")
    with cols[2]:
        search = st.text_input("Transaction", key="audit_search",
                               placeholder="Transaction number (e.g. 1600) or a word from the reason")
    with cols[3]:
        st.button("Clear", key="audit_clear", width="stretch", on_click=_clear_filters)
    return decision, policy, search


def _table(records) -> None:
    """The log as a dense table: one row per decision, newest first."""
    frame = pd.DataFrame([{
        "Transaction": r.txn_id,
        "Recorded": ui._stamp(r.recorded_at),
        "Simulated time": _sim_time(r.sim_minute),
        "Type": (r.tx_type or "-").replace("_", " ").title(),
        "Amount": round(r.amount, 2),
        "Risk": round(r.score, 1),
        "Decision": r.decision_label,
        "Policy": r.policy_hash,
    } for r in records])

    event = st.dataframe(
        frame, hide_index=True, width="stretch", height=420,
        on_select="rerun", selection_mode="single-row", key="audit_table",
        column_config={
            "Transaction": st.column_config.NumberColumn("Transaction", format="#%d", width="small"),
            "Recorded": st.column_config.TextColumn("Recorded (local)", width="medium"),
            "Simulated time": st.column_config.TextColumn("Simulated", width="small"),
            "Amount": st.column_config.NumberColumn("Amount", format="%.2f"),
            "Risk": st.column_config.NumberColumn("Risk", format="%.1f", width="small"),
            "Decision": st.column_config.TextColumn("Decision", width="small"),
            "Policy": st.column_config.TextColumn("Policy", width="small"),
        })
    picked = getattr(event, "selection", None)
    rows = list(getattr(picked, "rows", []) or [])
    return rows[0] if rows else None


def open_record(record, log, store) -> None:
    """One decision, end to end: what arrived, what was decided, what came of it."""
    @st.dialog(f"Decision record · transaction #{record.txn_id}", width="large")
    def _show() -> None:
        chain = log.chain(record.txn_id, cases=store, session=record.session)
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown(ui.decision_record_html(record, chain["policy"], chain["cases"]),
                        unsafe_allow_html=True)
            if record.reason:
                st.markdown(f'<div class="fa-root"><div class="fa-sub2">What drove the score</div>'
                            f'<div class="fa-case-text">{_safe(record.reason)}</div></div>',
                            unsafe_allow_html=True)
        with right:
            if chain["cases"]:
                for item in chain["cases"]:
                    case = item["case"]
                    st.markdown(f'<div class="fa-root"><div class="fa-sub2">{case.case_id} · '
                                f'{_safe(case.state_label)} · {_safe(case.status_label)}</div></div>',
                                unsafe_allow_html=True)
                    st.markdown(ui.case_history_html(item["events"]) if item["events"]
                                else '<div class="fa-note">No history recorded.</div>',
                                unsafe_allow_html=True)
            else:
                st.markdown('<div class="fa-root"><div class="fa-note">No investigation case was opened '
                            'for this transaction, so there is no analyst outcome to report.</div></div>',
                            unsafe_allow_html=True)

    _show()


def _safe(text: str) -> str:
    from html import escape
    return escape(str(text))


def render(header, store=None) -> None:
    """Draw the audit page."""
    header.markdown(ui.header_html("audit", "Every score, decision and the policy behind it — kept on disk",
                                   "Audit log"), unsafe_allow_html=True)

    log = audits()
    counts = log.counts()
    policies = log.policies()
    st.markdown(ui.audit_counts_html(counts), unsafe_allow_html=True)

    if counts["all"] == 0:
        st.markdown('<div class="fa-root"><div class="fa-card fa-empty">'
                    '<h3>No decisions logged yet</h3><p>Open the live simulation: every transaction it '
                    'decides is written here, with the model and policy version that produced it.</p>'
                    '</div></div>', unsafe_allow_html=True)
        return

    decision, policy, search = render_filters(policies)
    records = log.list(decision=None if decision == ALL else decision,
                       policy_hash=None if policy == ALL else policy,
                       search=search, limit=LATEST)

    if not records:
        st.markdown('<div class="fa-note" style="margin:20px 2px">No decision matches these filters.</div>',
                    unsafe_allow_html=True)
        return

    st.markdown('<div class="fa-root"><div class="fa-note" style="margin:18px 2px 8px">'
                f'{len(records):,} decision(s), newest first · select a row to see how it was decided</div></div>',
                unsafe_allow_html=True)
    index = _table(records)
    if index is not None and index < len(records):
        open_record(records[index], log, store if store is not None else
                    get_case_store(str(cases_path())))

    with st.expander(f"Policy versions used ({len(policies)})", expanded=False):
        st.markdown('<div class="fa-root"><div class="fa-note" style="margin:0 2px 10px">'
                    'Changing a threshold creates a new version. Decisions keep the version '
                    'they were made under, so history is never rewritten.</div></div>',
                    unsafe_allow_html=True)
        for version in policies:
            st.markdown(ui.policy_card_html(version), unsafe_allow_html=True)
