"""Mode 3, Transaction testing: score one transaction by hand, or a whole file."""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

import ui
from common import render_costs_popover, render_policy_popover
from src.decision import Policy
from src.processing import MAX_ROWS, ProcessedBatch, process_frame, summarize_batch, template_csv
from src.tools import Toolkit

TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
EXAMPLES = {
    "normal": {"tx_type": "TRANSFER", "tx_amount": 8_000.0, "tx_sender": 120_000.0, "tx_receiver": 60_000.0, "tx_hour": 14},
    "empty": {"tx_type": "TRANSFER", "tx_amount": 250_000.0, "tx_sender": 250_000.0, "tx_receiver": 0.0, "tx_hour": 3},
}


def _load_example(name: str) -> None:
    for key, value in EXAMPLES[name].items():
        st.session_state[key] = value


@st.cache_data(show_spinner="Scoring the file...")
def process_upload(tag: str, content: bytes, block: float, review_min: float, capacity: int) -> ProcessedBatch:
    """Score an uploaded file (cached, so changing a cost slider does not rescore it)."""
    from common import get_engine
    df = pd.read_csv(io.BytesIO(content))
    policy = Policy(block_threshold=block, review_min_score=review_min, daily_capacity=capacity)
    return process_frame(get_engine(tag), df, policy)


def _render_single(engine, policy: Policy) -> None:
    st.markdown(ui.mode_head("Single transaction", "Type in a transaction and see its risk, the decision and the reasons."),
                unsafe_allow_html=True)
    st.session_state.setdefault("tx_type", "TRANSFER")
    st.session_state.setdefault("tx_amount", 250_000.0)
    st.session_state.setdefault("tx_sender", 250_000.0)
    st.session_state.setdefault("tx_receiver", 0.0)
    st.session_state.setdefault("tx_hour", 3)

    c = st.columns(5)
    c[0].selectbox("Type", TYPES, key="tx_type")
    c[1].number_input("Amount", min_value=0.0, step=1_000.0, format="%.2f", key="tx_amount")
    c[2].number_input("Sender balance before", min_value=0.0, step=1_000.0, format="%.2f", key="tx_sender")
    c[3].number_input("Receiver balance before", min_value=0.0, step=1_000.0, format="%.2f", key="tx_receiver")
    c[4].slider("Hour of day", 0, 23, key="tx_hour")

    b = st.columns([1, 1.6, 3.4])
    b[0].button("Example: normal transfer", on_click=_load_example, args=("normal",))
    b[1].button("Example: large transfer to an empty account", on_click=_load_example, args=("empty",))
    if "oldbalanceOrg" in engine.drop:
        st.caption("This engine does not use the sender balance (it is removed in the recommended setting), "
                   "so changing it has no effect on the score.")

    tx = {"type": st.session_state.tx_type, "amount": st.session_state.tx_amount,
          "oldbalanceOrg": st.session_state.tx_sender, "oldbalanceDest": st.session_state.tx_receiver,
          "hour": st.session_state.tx_hour}
    try:
        # The same tool an agent or a script calls, not a UI-only shortcut.
        prediction = Toolkit(engine, policy=policy).explain_transaction(tx)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.markdown(ui.prediction_html(prediction, policy), unsafe_allow_html=True)


def _render_file(tag: str, policy: Policy, costs) -> None:
    st.markdown(ui.mode_head("Upload a file", f"Score up to {MAX_ROWS:,} transactions at once from a CSV file."),
                unsafe_allow_html=True)
    st.markdown(
        "Required columns: `type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`. Optional: `hour` (0-23) or `step`, "
        "and `isFraud` (0 or 1) if you know the true labels, which adds detection results and the money impact. "
        "Header names are not case sensitive and other columns are ignored.")
    st.download_button("Download a template file", data=template_csv(), file_name="transactions_template.csv",
                       mime="text/csv")
    uploaded = st.file_uploader("CSV file", type=["csv"], key="upload", label_visibility="collapsed")
    if uploaded is None:
        st.info("Choose a file to score it. Review capacity is not applied to uploaded files, "
                "because it needs transactions spread over time.")
        return

    try:
        batch = process_upload(tag, uploaded.getvalue(), float(policy.block_threshold),
                               float(policy.review_min_score), int(policy.daily_capacity))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        st.error("This file could not be read as a CSV. Check that it has a header row and comma separators.")
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    if batch.rows.empty:
        st.error("No valid rows were found in the file.")
    else:
        st.markdown(ui.batch_summary_html(summarize_batch(batch, costs)), unsafe_allow_html=True)
        st.markdown(ui.scored_table_html(batch.rows), unsafe_allow_html=True)
        out = batch.rows.rename(columns={"reason": "main_reason"}).drop(columns=["applicable"])
        st.download_button("Download the scored file", data=out.to_csv(index=False), file_name="scored_transactions.csv",
                           mime="text/csv")
    if not batch.rejected.empty:
        with st.expander(f"{len(batch.rejected):,} rows were rejected"):
            st.dataframe(batch.rejected.head(200), hide_index=True)


def render(header, tag: str, run, engine) -> None:
    """Draw the whole testing mode. `header` is the placeholder at the top of the page."""
    header.markdown(ui.header_html("testing", "Same engine as the replay, one transaction at a time",
                                  "Transaction testing"),
                    unsafe_allow_html=True)
    cols = st.columns([1.5, 1.2, 4])
    with cols[0]:
        policy = render_policy_popover()
    with cols[1]:
        costs = render_costs_popover()

    tab_one, tab_file = st.tabs(["Single transaction", "Upload a file"])
    with tab_one:
        _render_single(engine, policy)
    with tab_file:
        _render_file(tag, policy, costs)