"""Integration tests that drive the real Streamlit app headlessly (streamlit.testing.AppTest).

These run the app end to end, including loading the trained risk engine from
results/ and models/, so they are slow and skipped when no trained run exists.
Marked `slow`: run `pytest -m "not slow"` for the fast suite only.
"""
import re
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.artifacts import discover_runs, load_engine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))           # ui is an app/ module, importable here
import ui  # noqa: E402
RUNS = discover_runs(ROOT)
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not RUNS, reason="no trained risk engine: run train/train.py first"),
]
TAG = RUNS[0]


def text(html) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(html)))


def page(markdown_html=True, **state):
    """Start the app with the given session state."""
    at = AppTest.from_file(ROOT / "app" / "app.py", default_timeout=240)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    return at


def all_text(at) -> str:
    return " | ".join(text(m.value) for m in at.markdown)


def test_the_app_lands_on_live_simulation_without_errors():
    at = page()
    assert not at.exception
    assert at.session_state["mode"] == "Live simulation"
    assert "Simulated time" in all_text(at)


def test_the_sidebar_navigates_between_modes():
    """The three modes plus the two workspace pages are sidebar buttons: clicking one redraws the page."""
    at = page()
    nav = [b for b in at.sidebar.button if b.key and b.key.startswith("nav_")]
    assert {b.key for b in nav} == {"nav_live", "nav_replay", "nav_testing", "nav_cases",
                                    "nav_audit", "nav_approvals"}
    assert not at.exception

    at.sidebar.button(key="nav_replay").click().run()
    assert not at.exception
    assert at.session_state["mode"] == "Historical replay"
    assert "PaySim replay" in all_text(at)
    at.sidebar.button(key="nav_testing").click().run()
    assert not at.exception
    assert at.session_state["mode"] == "Transaction testing"
    assert "Single transaction" in all_text(at)
    at.sidebar.button(key="nav_cases").click().run()
    assert not at.exception
    assert at.session_state["mode"] == "Cases"
    assert "Investigation cases" in all_text(at)
    at.sidebar.button(key="nav_audit").click().run()
    assert not at.exception
    assert at.session_state["mode"] == "Audit"


def test_the_audit_page_lists_decisions_with_the_policy_behind_them(tmp_path, monkeypatch):
    """The decision log is readable on its own page, filters included."""
    import pandas as pd

    from src.audit import AuditLog
    from src.decision import Policy

    db = tmp_path / "audit.db"
    monkeypatch.setenv("FRAUD_AUDIT_DB", str(db))
    log = AuditLog(db)
    policy = Policy(block_threshold=96.0, review_min_score=90.0, daily_capacity=240)
    log.record(pd.DataFrame([
        {"id": 1600, "minute": 520, "amount": 215_000.0, "type": "CASH_OUT",
         "risk": 99.7, "decision": "block", "reason": "sender emptied into a new account"},
        {"id": 1601, "minute": 521, "amount": 900.0, "type": "TRANSFER",
         "risk": 12.0, "decision": "approve", "reason": ""},
    ]), policy, model_tag="v3", session="s1")

    at = page(mode="Audit")
    assert not at.exception
    body = all_text(at)
    assert "Decisions logged" in body and "Policy versions" in body

    table = at.dataframe[0].value
    assert list(table["Transaction"]) == [1601, 1600]          # newest first
    assert table.iloc[1]["Decision"] == "Blocked"
    assert table.iloc[1]["Policy"] == log.entry(1600).policy_hash

    at.selectbox(key="audit_decision").select("review").run()   # nothing was reviewed
    assert not at.exception
    assert "No decision matches these filters" in all_text(at)
    log.close()


def test_the_empty_live_charts_do_not_talk_about_the_replay():
    """Live mode has no Start button, so its empty charts must not say 'Press Start'."""
    at = page()
    assert "Press Start" not in all_text(at)
    assert "Waiting for the first simulated transactions" in all_text(at)


def test_switching_modes_preserves_the_live_stream():
    from src.live import LiveStream, LiveSettings
    from src.decision import Policy

    stream = LiveStream(_engine(), seed=7, now=0.0, settings=LiveSettings(300, 0.03, False, 2.0, False))
    stream.advance(Policy(), now=20.0)
    at = page(live_stream=stream)
    rows_before, processed_before = len(stream.rows), stream.processed

    for mode in ("Historical replay", "Transaction testing", "Live simulation"):
        at.session_state["mode"] = mode
        at.run()
        assert not at.exception, mode
    assert at.session_state["live_stream"] is stream           # same object, not recreated
    assert len(stream.rows) >= rows_before and stream.processed >= processed_before
    assert "Live transactions" in all_text(at)


def test_alert_selection_opens_the_detail_pop_up():
    from src.live import LiveStream, LiveSettings
    from src.decision import Policy

    stream = LiveStream(_engine(), seed=7, now=0.0, settings=LiveSettings(300, 0.03, False, 2.0, False))
    stream.advance(Policy(), now=30.0)
    at = page(live_stream=stream)
    alert_buttons = [b for b in at.button if b.key and b.key.startswith("alert_")]
    assert alert_buttons, "a populated stream should raise alerts"
    alert_buttons[0].click().run()
    assert not at.exception
    assert len(at.get("dialog")) == 1
    dialog_text = " ".join(text(m.value) for m in at.get("dialog")[0].markdown)
    assert "Risk score" in dialog_text and "Transaction #" in dialog_text
    assert "What raised the risk" in dialog_text


def test_a_stream_from_an_older_version_is_upgraded_not_crashed():
    from src.live import LiveStream, LiveSettings
    from src.decision import Policy

    stream = LiveStream(_engine(), seed=3, now=0.0, settings=LiveSettings(300, 0.03, False, 2.0, False))
    stream.advance(Policy(), now=15.0)
    del stream.processed                                       # pretend it predates the field
    at = page(live_stream=stream)
    assert not at.exception
    assert stream.processed == stream.proc_count
    assert "processed" in all_text(at)


def test_the_cases_page_clear_button_resets_the_filters(tmp_path, monkeypatch):
    """Regression: Clear used to write to widget keys after their widgets were
    instantiated, raising StreamlitAPIException. It now resets in a callback."""
    from src.cases import CaseStore

    monkeypatch.setenv("FRAUD_CASES_DB", str(tmp_path / "cases.db"))
    CaseStore(tmp_path / "cases.db").create(txn_id=42, risk=91.0, decision="block",
                                            amount=1234.5, tx_type="TRANSFER", source="test",
                                            explanation="unit test")
    at = page(mode="Cases")
    assert not at.exception

    def listed(at):
        """The case id shows in a card markdown, and its Open button exists."""
        shown = any("C-0001" in m.value for m in at.markdown)
        return shown and at.button(key="case_row_1") is not None

    assert listed(at)

    at.selectbox(key="case_status").select("investigating").run()
    at.text_input(key="case_search").set_value("zzz").run()
    assert not any("C-0001" in m.value for m in at.markdown)   # the filters actually filtered

    at.button(key="case_clear").click().run()
    assert not at.exception
    assert at.selectbox(key="case_status").value == "All"
    assert at.selectbox(key="case_priority").value == "All"
    assert at.text_input(key="case_search").value == ""
    assert listed(at)                                          # the case is back


def _by_prefix(widgets, prefix):
    """The one widget whose key starts with a prefix (keys carry a freshness token)."""
    for widget in widgets:
        key = getattr(widget, "key", None)
        if key and key.startswith(prefix):
            return widget
    return None


def test_opening_a_case_from_the_list_goes_to_its_page(tmp_path, monkeypatch):
    """A card is a link now: the case has a page, not a pop-up."""
    from src.cases import CaseStore

    monkeypatch.setenv("FRAUD_CASES_DB", str(tmp_path / "cases.db"))
    store = CaseStore(tmp_path / "cases.db")
    case = store.create(txn_id=88, risk=93.0, decision="review", amount=1500.0,
                        tx_type="CASH_OUT", source="test", explanation="a test case")

    at = page(mode="Cases")
    assert not at.exception
    at.button(key=f"case_row_{case.number}").click().run()
    assert not at.exception
    assert at.session_state["mode"] == "Case"
    assert at.session_state["selected_case"] == case.number
    assert len(at.get("dialog")) == 0                      # no pop-up any more
    assert f"Case {case.case_id}" in all_text(at)
    assert "RiskPilot" in all_text(at)

    at.button(key="case_back").click().run()
    assert not at.exception
    assert at.session_state["mode"] == "Cases"
    assert at.session_state["selected_case"] is None
    assert "Investigation cases" in all_text(at)


def test_the_case_lifecycle_on_the_case_page(tmp_path, monkeypatch):
    """Open -> Save and close -> Reopen, driven through the real widgets.

    Regression for the form gotcha: a selectbox inside an st.form only hands
    its value over on submit, so the Status picker lives outside the form —
    otherwise 'Save and close' could never become clickable in a browser.
    """
    from src.cases import CaseStore

    db = tmp_path / "cases.db"
    monkeypatch.setenv("FRAUD_CASES_DB", str(db))
    store = CaseStore(db)
    case = store.create(txn_id=77, risk=95.0, decision="block", amount=900.0,
                        tx_type="TRANSFER", source="test", explanation="lifecycle test")

    at = page(mode="Case", selected_case=case.number)
    assert not at.exception
    status = _by_prefix(at.selectbox, f"case_page_status_{case.number}")
    assert status.value == "investigating"

    save_close = [b for b in at.button if b.label == "Save and close"][0]
    assert save_close.disabled            # nothing to close it with yet

    # Picking a resolution arms the close button; the status picker must be
    # outside the form for this to work (values land before the submit rerun).
    status.select("confirmed_fraud").run()
    assert not at.exception
    save_close = [b for b in at.button if b.label == "Save and close"][0]
    assert not save_close.disabled

    save_close.click().run()              # Save and close
    assert not at.exception
    saved = store.get(case.case_id)
    assert saved.closed and saved.status == "confirmed_fraud"

    # The page now shows the closed state and offers Reopen case.
    reopen = [b for b in at.button if b.label == "Reopen case"]
    assert reopen, "a closed case offers Reopen case"
    reopen[0].click().run()
    assert not at.exception
    reopened = store.get(case.case_id)
    assert not reopened.closed and reopened.status == "confirmed_fraud"

    # The audit trail is one click away beside the status pill, in a popover:
    # open -> status -> close -> reopen. AppTest cannot see popovers, so the
    # trail content is asserted through the renderer the popover feeds.
    trail = ui.case_history_html(store.events(case.case_id))
    assert "Opened from alert #77" in trail
    assert "Status: Investigating -&gt; Confirmed fraud" in trail
    assert "Case closed" in trail and "Case reopened" in trail


def _engine():
    return load_engine(ROOT, TAG)


# --------------------------------------------------------------------------- #
# The investigation assistant and its approval queue (src/agent.py)
# --------------------------------------------------------------------------- #
def _find_button(at, key):
    """A button by key, whether it sits on the page or inside a pop-up."""
    for widget in at.button:
        if getattr(widget, "key", None) == key:
            return widget
    for dialog in at.get("dialog"):
        for widget in dialog.button:
            if getattr(widget, "key", None) == key:
                return widget
    return None


def test_the_approvals_page_starts_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUD_PROPOSALS_DB", str(tmp_path / "proposals.db"))
    at = page(mode="Approvals")
    assert not at.exception
    assert "No recommendations yet" in all_text(at)
    assert "investigation assistant" in all_text(at)


def test_the_assistant_runs_from_the_case_page_and_waits_for_approval(tmp_path, monkeypatch):
    """The whole loop through the real widgets: run, file, approve, case moves.

    The point of the test is the last assertion: the case is unchanged by the
    assistant and changes only after the approval click.
    """
    from src.cases import CaseStore
    from src.decision import Policy
    from src.live import LiveSettings, LiveStream
    from src.proposals import ProposalStore

    cases_db = tmp_path / "cases.db"
    proposals_db = tmp_path / "proposals.db"
    monkeypatch.setenv("FRAUD_CASES_DB", str(cases_db))
    monkeypatch.setenv("FRAUD_PROPOSALS_DB", str(proposals_db))
    monkeypatch.setenv("FRAUD_CHAT_DB", str(tmp_path / "chat.db"))

    stream = LiveStream(_engine(), seed=7, now=0.0, settings=LiveSettings(300, 0.03, False, 2.0, False))
    stream.advance(Policy(), now=30.0)
    flagged = stream.rows[stream.rows["decision"] != "approve"]
    assert not flagged.empty, "the stream should raise alerts"
    # Highest risk first: on this engine's scale that is always >= 95, so the
    # assistant's rule R3 fires (escalate to urgent) and the approval below has
    # a visible effect on a case seeded at "low".
    row = flagged.sort_values("risk", ascending=False).iloc[0]

    store = CaseStore(cases_db)
    case = store.create(txn_id=int(row.id), risk=float(row.risk), decision=str(row.decision),
                        amount=float(row.amount), tx_type=str(row.type), source="live",
                        priority="low", explanation="assistant flow")

    at = page(mode="Case", selected_case=case.number, live_stream=stream)
    assert not at.exception

    run = _find_button(at, f"run_agent_{case.number}")
    assert run is not None, "the page offers the assistant"
    run.click().run()
    assert not at.exception

    panel = all_text(at)
    assert "Evidence (" in panel                  # the sourced facts
    assert "get_case" not in panel                # tool names stay in the data, not on screen
    assert "rule R" in panel
    unchanged = store.get(case.case_id)
    assert unchanged.priority == "low" and unchanged.status == "investigating"

    _find_button(at, f"file_agent_{case.number}").click().run()
    assert not at.exception
    queue = ProposalStore(proposals_db)
    pending = queue.pending()
    assert len(pending) == 1
    assert pending[0].case_number == case.number
    assert store.get(case.case_id) == unchanged      # filing is not approving

    queue.close()

    at.session_state["mode"] = "Approvals"
    at.run()
    assert not at.exception
    assert "Awaiting approval" in all_text(at)

    item = ProposalStore(proposals_db).pending()[0]
    if item.requires_override:
        at.checkbox(key=f"prop_override_{item.number}").check().run()
    _find_button(at, f"prop_approve_{item.number}").click().run()
    assert not at.exception

    moved = store.get(case.case_id)
    assert moved.priority != "low"                  # the approved change landed
    assert "Assistant (rules:v2" in moved.notes
    assert ProposalStore(proposals_db).get(item.number).status == "approved"


def test_rejecting_a_proposal_leaves_the_case_untouched(tmp_path, monkeypatch):
    from src.cases import CaseStore
    from src.proposals import ProposalStore

    cases_db = tmp_path / "cases.db"
    proposals_db = tmp_path / "proposals.db"
    monkeypatch.setenv("FRAUD_CASES_DB", str(cases_db))
    monkeypatch.setenv("FRAUD_PROPOSALS_DB", str(proposals_db))

    store = CaseStore(cases_db)
    case = store.create(txn_id=501, risk=97.0, decision="block", amount=1000.0,
                        tx_type="TRANSFER", source="test", priority="low")
    queue = ProposalStore(proposals_db)
    proposal = queue.create(case.case_id, "escalate", rationale="because the policy blocks it",
                            risk=97.0, policy_decision="block", proposed_priority="urgent",
                            agent="rules:v1/template")
    queue.close()

    at = page(mode="Approvals")
    assert not at.exception
    _find_button(at, f"prop_reject_{proposal.number}").click().run()
    assert not at.exception

    assert store.get(case.case_id) == case           # nothing happened to the case
    assert ProposalStore(proposals_db).get(proposal.number).status == "rejected"


# --------------------------------------------------------------------------- #
# RiskPilot: the conversation on the case page (src/assistant.py, src/chat.py)
# --------------------------------------------------------------------------- #
def test_the_chat_answers_and_applies_only_after_approval(tmp_path, monkeypatch):
    """One question, one request, one approval -- driven through the chat box."""
    from src.cases import CaseStore
    from src.chat import ChatStore
    from src.proposals import ProposalStore

    cases_db = tmp_path / "cases.db"
    proposals_db = tmp_path / "proposals.db"
    monkeypatch.setenv("FRAUD_CASES_DB", str(cases_db))
    monkeypatch.setenv("FRAUD_PROPOSALS_DB", str(proposals_db))
    monkeypatch.setenv("FRAUD_CHAT_DB", str(tmp_path / "chat.db"))

    store = CaseStore(cases_db)
    case = store.create(txn_id=986, risk=95.1, decision="review", amount=477_066.97,
                        tx_type="CASH_OUT", source="live",
                        explanation="Receiver balance - Receiver balance 0 (typical 558,465)")

    at = page(mode="Case", selected_case=case.number)
    assert not at.exception
    assert "RiskPilot" in all_text(at)
    assert "I have read" in all_text(at)            # the intro, before any turn

    at.chat_input(key=f"chat_input_{case.number}").set_value("Why was it flagged?").run()
    assert not at.exception
    body = all_text(at)
    assert "Why was it flagged?" in body            # the analyst's turn is on screen
    assert "opened at risk 95.1" in body            # grounded in the case, not invented
    assert ChatStore(tmp_path / "chat.db").count(case.case_id) == 2

    at.chat_input(key=f"chat_input_{case.number}").set_value("mark it as false positive").run()
    assert not at.exception
    assert store.get(case.case_id).status == "investigating"      # a request, not a change
    proposal = ProposalStore(proposals_db).pending()[0]
    assert proposal.kind == "close_false_positive"

    _find_button(at, f"chat_approve_{proposal.number}").click().run()
    assert not at.exception
    moved = store.get(case.case_id)
    assert moved.status == "false_positive" and moved.closed

    transcript = ChatStore(tmp_path / "chat.db").history(case.case_id)
    assert transcript[-1].role == "system"
    assert "Approved" in transcript[-1].text and "false positive" in transcript[-1].text
    assert "Proposal" in all_text(at)               # the card now shows the verdict


def test_a_quick_prompt_starts_the_conversation(tmp_path, monkeypatch):
    """The three chips on an empty transcript are real turns, not decoration."""
    from src.cases import CaseStore
    from src.chat import ChatStore

    monkeypatch.setenv("FRAUD_CASES_DB", str(tmp_path / "cases.db"))
    monkeypatch.setenv("FRAUD_PROPOSALS_DB", str(tmp_path / "proposals.db"))
    monkeypatch.setenv("FRAUD_CHAT_DB", str(tmp_path / "chat.db"))
    store = CaseStore(tmp_path / "cases.db")
    case = store.create(txn_id=12, risk=96.0, decision="review", amount=10_000.0,
                        tx_type="TRANSFER", source="test", priority="low")

    at = page(mode="Case", selected_case=case.number)
    chips = [b for b in at.button if b.key and b.key.startswith("chat_chip_")]
    assert len(chips) == 3
    chips[2].click().run()                          # "What do you recommend?"
    assert not at.exception
    assert "recommendation" in all_text(at).lower()
    assert ChatStore(tmp_path / "chat.db").count(case.case_id) == 2
