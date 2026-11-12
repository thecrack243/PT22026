"""Mode 2, Live simulation: new transactions arrive continuously and are scored as they come.

The analyst can only watch and work the alerts: there is no start, stop or reset, just like
a real live feed. The simulator settings are for the person running the demo. Changing the
decision policy affects only transactions that arrive afterwards.
"""
from __future__ import annotations

import time

import streamlit as st

import ui
from common import get_engine, popover, render_costs_popover, render_policy_popover
from src.decision import Policy
from src.live import LiveSettings, LiveStream, clock_label
from src.predict import predict_one

TICK_SECONDS = 1.0
AWAY_MINUTES = 20           # a catch-up longer than this is reported as "while you were away"
PREF_KEYS = ("sim_volume", "sim_fraud", "sim_speed", "sim_new_pattern", "sim_shift", "sim_show_truth")


def _restore_prefs() -> None:
    for key, value in st.session_state.get("_live_prefs", {}).items():
        st.session_state.setdefault(key, value)


def _save_prefs() -> None:
    st.session_state["_live_prefs"] = {k: st.session_state[k] for k in PREF_KEYS if k in st.session_state}


def _select_alert(txn_id: int) -> None:
    """An alert was clicked. The pop-up is opened by render() on the next full run."""
    st.session_state.alert_to_open = int(txn_id)
    st.session_state.live_selected = int(txn_id)     # used by the inline fallback


def _clear_alert() -> None:
    st.session_state.live_selected = None


def get_stream(engine) -> LiveStream:
    """The running simulation of this browser session (created on first use)."""
    if "live_stream" not in st.session_state:
        st.session_state.live_stream = LiveStream(engine)
    stream = st.session_state.live_stream
    # A stream created by an older version of the app can still be alive in this
    # browser session after a code reload. Give it anything newer versions added,
    # instead of crashing the page.
    if not hasattr(stream, "processed"):
        stream.processed = int(getattr(stream, "proc_count", 0))
    return stream


def render_simulator_popover() -> tuple[LiveSettings, bool]:
    """Controls for the demo operator, not for the analyst."""
    with popover()("Simulator"):
        volume = st.slider("Transactions per simulated hour", 50, 1000, 300, 50, key="sim_volume")
        fraud = st.slider("Fraud rate (%)", 0.5, 10.0, 3.0, 0.5, key="sim_fraud",
                          help="Share of generated transactions that are fraud. Real fraud is far rarer; "
                               "a higher rate makes the demo easier to follow.")
        if fraud > 1.0:
            st.caption(":orange[Real PaySim fraud is about **0.13%**. At {:.1f}% the fraud caught, recall and "
                       "money-saved numbers on this page look far better than they would in production.]".format(fraud))
        speed = st.slider("Simulated minutes per real second", 1.0, 10.0, 2.0, 0.5, key="sim_speed")
        new_pattern = st.toggle("Inject a new fraud pattern", value=False, key="sim_new_pattern",
                                help="Adds small transfers between accounts that already hold money. The engine "
                                     "learned 'the sender is emptied into an empty receiver', so watch how many "
                                     "of these it misses.")
        shift = st.toggle("Simulate a market shift", value=False, key="sim_shift",
                          help="Amounts become 10x larger and receivers are new accounts. The drift monitor compares "
                               "the last 3 simulated hours with the start of the session and should react.")
        show_truth = st.toggle("Show the true label", value=True, key="sim_show_truth",
                               help="Adds a true-label column to the live feed. In real life the label is not "
                                    "known when a transaction arrives - it is only available here because the "
                                    "traffic is simulated.")
    return LiveSettings(int(volume), fraud / 100.0, bool(new_pattern), float(speed), bool(shift)), bool(show_truth)


def _prediction(engine, row, policy):
    """Reasons for one alert (cached per transaction)."""
    cache = st.session_state.setdefault("live_pred_cache", {})
    if int(row.id) not in cache:
        if len(cache) > 300:
            cache.clear()
        cache[int(row.id)] = predict_one(engine, {
            "type": row.type, "amount": row.amount, "oldbalanceOrg": row.oldbalanceOrg,
            "oldbalanceDest": row.oldbalanceDest, "hour": int(row.hour)}, policy)
    return cache[int(row.id)]


def open_alert_popup(txn_id: int, policy: Policy, show_truth: bool) -> None:
    """Show one alert in a centered pop-up window with a close button (Escape or a click outside also close it)."""
    @st.dialog(f"Alert #{txn_id}", width="large")
    def _show() -> None:
        stream = st.session_state.get("live_stream")
        row = stream.get(txn_id) if stream is not None else None
        if row is None:
            st.info("This alert is no longer available (it may have been trimmed from the feed).")
            return
        tag = st.session_state.get("live_tag")
        if tag is None:
            st.info("The session was reset. Close this window and open the alert again.")
            return
        engine = get_engine(tag)
        st.markdown(ui.alert_detail_html(row, _prediction(engine, row, policy), policy, show_truth, in_dialog=True),
                    unsafe_allow_html=True)

    _show()


def draw(stream: LiveStream, engine, policy, costs, settings: LiveSettings, show_truth: bool) -> None:
    """Process what arrived since the last tick and draw the whole live screen."""
    if st.session_state.get("alert_to_open") is not None and hasattr(st, "dialog"):
        st.rerun()                                  # a full run opens the pop-up (see render)
    try:
        info = stream.advance(policy, settings=settings)
    except Exception as exc:                        # keep the page alive and say what broke
        st.error(f"The live simulator stopped while processing new transactions: {exc}")
        return
    if info["caught_up_minutes"] >= AWAY_MINUTES and hasattr(st, "toast"):
        st.toast(f"While you were away, {info['new']:,} simulated transactions arrived "
                 f"and {info['new_alerts']:,} raised an alert.")

    view = stream.view(policy, costs)
    strip = ui.live_status_strip(view, clock_label(stream.minute), stream.avg_ms, stream.recent_throughput,
                                 stream.processed)
    st.markdown(ui.live_dashboard_html(view, strip), unsafe_allow_html=True)

    col_feed, col_alerts = st.columns([1.55, 1])
    with col_feed:
        st.markdown(ui.feed_html(stream.latest(14), show_truth), unsafe_allow_html=True)
    with col_alerts:
        with st.container(border=True):
            m = view["metrics"]
            st.markdown(ui.alerts_title_html(m["reviewed"] + m["blocked"]), unsafe_allow_html=True)
            alerts = stream.alerts(10)
            if alerts.empty:
                st.markdown(ui.alert_hint_html(), unsafe_allow_html=True)
            for row in alerts.itertuples():
                st.button(ui.alert_label(row), key=f"alert_{row.id}", on_click=_select_alert, args=(row.id,))
        selected = st.session_state.get("live_selected")
        if selected is not None and not hasattr(st, "dialog"):       # older Streamlit: show the card inline
            row = stream.get(selected)
            if row is None:                         # trimmed away
                _clear_alert()
            else:
                st.markdown(ui.alert_detail_html(row, _prediction(engine, row, policy), policy, show_truth),
                            unsafe_allow_html=True)
                st.button("Close details", key="close_alert", on_click=_clear_alert)


def render(header, tag: str, run, engine) -> None:
    """Draw the whole live mode. `header` is the placeholder at the top of the page."""
    _restore_prefs()
    header.markdown(ui.header_html("live", "Live simulation · PaySim format"), unsafe_allow_html=True)
    cols = st.columns([1.5, 1.2, 1.4, 3])
    with cols[0]:
        policy = render_policy_popover()
    with cols[1]:
        costs = render_costs_popover()
    with cols[2]:
        settings, show_truth = render_simulator_popover()
    _save_prefs()

    stream = get_stream(engine)
    st.session_state.live_tag = tag
    pending = st.session_state.pop("alert_to_open", None)
    if pending is not None and hasattr(st, "dialog"):
        open_alert_popup(pending, policy, show_truth)

    if hasattr(st, "fragment"):
        @st.fragment(run_every=TICK_SECONDS)
        def live() -> None:
            draw(stream, engine, policy, costs, settings, show_truth)

        live()
    else:                                           # older Streamlit: rerun the whole page
        draw(stream, engine, policy, costs, settings, show_truth)
        time.sleep(TICK_SECONDS)
        st.rerun()