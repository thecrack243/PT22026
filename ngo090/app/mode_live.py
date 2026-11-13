"""Mode 2, Live simulation: new transactions arrive continuously and are scored as they come.

The analyst can only watch and work the alerts: there is no start, stop or reset, just like
a real live feed. The simulator settings are for the person running the demo. Changing the
decision policy affects only transactions that arrive afterwards.
"""
from __future__ import annotations

import time

import streamlit as st

import ui
from common import (cases_path, get_case_store, get_engine, popover, render_costs_popover,
                    render_policy_popover)
from src.cases import auto_open
from src.decision import APPROVE, Policy
from src.live import LiveSettings, LiveStream, clock_label
from src.predict import predict_one

TICK_SECONDS = 1.0
AWAY_MINUTES = 20           # a catch-up longer than this is reported as "while you were away"
PREF_KEYS = ("sim_volume", "sim_fraud", "sim_speed", "sim_new_pattern", "sim_shift", "sim_show_truth",
             "auto_case_on", "auto_case_min_risk", "auto_case_limit")
AUTO_DEFAULTS = {"auto_case_on": False, "auto_case_min_risk": 99.0, "auto_case_limit": 50}


def _restore_prefs() -> None:
    # Saved settings first, then the defaults — both with setdefault. Anything
    # already in the session (above all: a value the analyst just changed in a
    # widget) wins; what was saved is only needed when a widget unmounted (a
    # trip to another page drops its session value), and the defaults fill the
    # gaps of a brand-new session.
    for key, value in st.session_state.get("_live_prefs", {}).items():
        st.session_state.setdefault(key, value)
    for key, value in AUTO_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("auto_case_made", 0)
    # The switch of the current generation starts from the server's truth, so a
    # mid-batch visit to another page does not come back as a phantom "off".
    gen = int(st.session_state.get("auto_case_gen", 0))
    st.session_state.setdefault(f"auto_case_switch_{gen}", bool(st.session_state.get("auto_case_on", False)))


def _save_prefs() -> None:
    st.session_state["_live_prefs"] = {k: st.session_state[k] for k in PREF_KEYS if k in st.session_state}


def _sync_switch(gen: int) -> None:
    """The analyst moved the switch: honour it, arming resets the batch.

    A stop retires the widget's generation; if the browser then replays its
    stale value for the retired key, the generation check drops it instead of
    silently re-arming a batch the analyst never asked for.
    """
    if gen != int(st.session_state.get("auto_case_gen", 0)):
        return                                       # a retired key's stale replay
    on = bool(st.session_state[f"auto_case_switch_{gen}"])
    st.session_state["auto_case_on"] = on
    if on:
        st.session_state["auto_case_made"] = 0


def render_alert_popover() -> tuple[bool, float, int]:
    """Automatic case creation: on/off, the risk cut-off, and the batch size."""
    on = bool(st.session_state.get("auto_case_on", False))
    made = int(st.session_state.get("auto_case_made", 0))
    limit = int(st.session_state.get("auto_case_limit", 50) or 0)
    # The generation in the key matters: when the batch fills, the switch is
    # turned off by the app itself. Writing to a widget's key does not stop the
    # browser from replaying its stale "on" for the old key, which silently
    # re-armed the batch. A fresh key per stop leaves the old value homeless.
    gen = int(st.session_state.get("auto_case_gen", 0))
    with popover()("Alert"):
        st.toggle("Create cases automatically", key=f"auto_case_switch_{gen}",
                  on_change=_sync_switch, args=(gen,),
                  help="Every alert at or above the risk score below is copied into an investigation "
                       "case the moment it arrives, so a high-risk alert cannot slip through the feed "
                       "unnoticed. The analyst still decides how each case ends.")
        min_risk = st.slider("Only above risk score", 90.0, 100.0, 99.0, 0.1, key="auto_case_min_risk",
                             help="Applies to new alerts only; cases already stored are never changed.")
        limit = st.number_input("Stop after this many cases", 0, 1000, 50, 10, key="auto_case_limit",
                                help="The batch. When it is full the switch turns itself off and nothing "
                                     "more is created until you switch it back on. 0 = no limit.")
        if limit and made >= limit:
            st.caption(":orange[Batch full — switch it on again for a new batch.]")
        elif limit:
            st.caption(f"{'On' if on else 'Off'} · {made} of {int(limit)} automatic cases in this batch.")
        elif on:
            st.caption(f"On · {made} automatic cases so far, no limit.")
        else:
            st.caption("Off — alerts are only kept when you open a case yourself.")
    return on, float(min_risk), int(limit)


def _stop_autocase() -> None:
    """Pause automatic creation: server truth off, the widget's generation retired.

    There is deliberately no st.rerun here. A rerun from inside the fragment
    breaks the browser session, and the reconnecting tab replays its stale
    "switch on" — silently re-arming the batch. Instead the switch is off for
    the app from this tick on, the retired key keeps the browser's stale value
    harmless, the toast tells the analyst what happened, and the live status
    line in the alerts panel turns to "paused" within a second. The switch
    itself redraws in its new generation at the next interaction.
    """
    st.session_state["auto_case_on"] = False
    st.session_state["auto_case_gen"] = int(st.session_state.get("auto_case_gen", 0)) + 1
    if hasattr(st, "toast"):
        st.toast("Automatic case creation is paused: the batch is full. Open the Alert settings "
                 "to start a new one.")


def _select_alert(txn_id: int) -> None:
    """An alert was clicked: the pop-up is drawn on the next full run.

    The pop-up is one-shot: it is closed with the X (or Escape) and does not
    come back on the next rerun. Its own button asks for the pop-up again,
    because a dialog that is not re-created throws its widgets away.
    """
    st.session_state.live_selected = int(txn_id)     # also used by the inline fallback
    st.session_state.alert_to_open = int(txn_id)     # ask the next full run to draw it


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


def _create_case(txn_id: int) -> None:
    """Copy this alert into a durable case and jump straight to it.

    Runs in the pop-up body after the button click (not in an on_click
    callback: a callback triggered inside a dialog only reruns the fragment,
    so the jump to the Cases page would never happen in a real browser).
    The st.rerun() right after turns this into a full run, where the Cases
    page opens the new pop-up and the sidebar highlight follows.
    """
    store = get_case_store(str(cases_path()))
    case = store.open_for(txn_id)                    # clicked twice before the page redrew
    if case is None:
        stream = st.session_state.get("live_stream")
        row = stream.get(txn_id) if stream is not None else None
        if row is None:                              # trimmed away between click and rerun
            return
        case = store.create(txn_id, float(row.risk), str(row.decision), str(row.reason),
                            source="live", amount=float(row.amount), tx_type=str(row.type))
    st.session_state.mode = "Cases"                  # the sidebar highlight follows on this run
    st.session_state["case_to_open"] = case.number   # the case pop-up opens on the Cases page
    st.session_state["alert_to_open"] = None         # do not re-open this alert pop-up later
    st.rerun()


def case_action(row) -> None:
    """Offer to keep this alert as an investigation case (stored on disk).

    The alert itself is ephemeral: rows are trimmed and a restart forgets the
    feed. A case is the analyst's durable copy, so it is written to SQLite here.
    """
    existing = get_case_store(str(cases_path())).open_for(int(row.id))

    st.markdown('<div class="fa-note" style="margin:14px 0 6px">Investigation</div>', unsafe_allow_html=True)
    if existing is not None:
        st.caption(f"Case {existing.case_id} already exists · {existing.state_label} · "
                   f"{existing.priority_label} priority. Open it from Cases in the sidebar.")
        return

    if st.button("Open investigation case", key=f"open_case_{int(row.id)}", width="stretch",
                 icon=":material/add_task:"):
        _create_case(int(row.id))


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
        case_action(row)

    _show()


def _autocase_status() -> None:
    """Live one-liner about automatic case creation, fresh every tick.

    This is what the analyst actually watches: the switch in the Alert settings
    only redraws on a full run, while this line is part of the ticking panel.
    """
    limit = int(st.session_state.get("auto_case_limit", 50) or 0)
    made = int(st.session_state.get("auto_case_made", 0))
    on = bool(st.session_state.get("auto_case_on", False))
    full = bool(limit) and made >= limit
    if full:
        st.caption(":orange[Auto-case paused — batch full. Switch it on again for a new batch.]")
    elif on:
        st.caption(f"Auto-case on · {made} of {limit} opened" if limit else f"Auto-case on · {made} opened")
    # Off stays silent: the panel is about the alerts, not about a disabled feature.


def _autocase(rows) -> None:
    """Turn the alerts that just arrived into cases, while the batch has room."""
    on = bool(st.session_state.get("auto_case_on", False))
    if not on:
        return
    limit = int(st.session_state.get("auto_case_limit", 50) or 0)
    made = int(st.session_state.get("auto_case_made", 0))
    room = 0 if limit == 0 else max(limit - made, 0)
    if limit and room == 0:                          # the batch was already full
        _stop_autocase()
        return
    alerts = rows[rows["decision"] != APPROVE]
    if alerts.empty:
        return
    store = get_case_store(str(cases_path()))
    created = auto_open(store, alerts.itertuples(),
                        float(st.session_state.get("auto_case_min_risk", 99.0)), room)
    if created:
        st.session_state["auto_case_made"] = made + created
    if limit and made + created >= limit:
        _stop_autocase()


def draw(stream: LiveStream, engine, policy, costs, settings: LiveSettings, show_truth: bool) -> None:
    """Process what arrived since the last tick and draw the whole live screen."""
    if st.session_state.get("alert_to_open") is not None and hasattr(st, "dialog"):
        st.rerun()                                  # a full run opens the pop-up (see render)
    try:
        info = stream.advance(policy, settings=settings)
    except Exception as exc:                        # keep the page alive and say what broke
        st.error(f"The live simulator stopped while processing new transactions: {exc}")
        return
    if info["new"]:
        # tail() rather than a column slice: the stream trims old rows when it grows.
        _autocase(stream.rows.tail(int(info["new"])))
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
            _autocase_status()
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
    header.markdown(ui.header_html("live", "PaySim format · every transaction scored as it arrives",
                                   "Live simulation"), unsafe_allow_html=True)
    cols = st.columns([1.5, 1.0, 1.3, 1.15, 3])
    with cols[0]:
        policy = render_policy_popover()
    with cols[1]:
        costs = render_costs_popover()
    with cols[2]:
        settings, show_truth = render_simulator_popover()
    with cols[3]:
        render_alert_popover()
    _save_prefs()

    stream = get_stream(engine)
    st.session_state.live_tag = tag
    pending = st.session_state.pop("alert_to_open", None)   # one-shot: the click that asked for the pop-up
    if pending is not None and hasattr(st, "dialog"):
        open_alert_popup(int(pending), policy, show_truth)

    if hasattr(st, "fragment"):
        @st.fragment(run_every=TICK_SECONDS)
        def live() -> None:
            draw(stream, engine, policy, costs, settings, show_truth)

        live()
    else:                                           # older Streamlit: rerun the whole page
        draw(stream, engine, policy, costs, settings, show_truth)
        time.sleep(TICK_SECONDS)
        st.rerun()