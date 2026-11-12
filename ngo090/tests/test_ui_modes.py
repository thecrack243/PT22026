"""Integration tests that drive the real Streamlit app headlessly (streamlit.testing.AppTest).

These run the app end to end, including loading the trained risk engine from
results/ and models/, so they are slow and skipped when no trained run exists.
Marked `slow`: run `pytest -m "not slow"` for the fast suite only.
"""
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.artifacts import discover_runs, load_engine

ROOT = Path(__file__).resolve().parent.parent
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
    """The three modes plus Cases are sidebar buttons: clicking one redraws the page."""
    at = page()
    nav = [b for b in at.sidebar.button if b.key and b.key.startswith("nav_")]
    assert {b.key for b in nav} == {"nav_live", "nav_replay", "nav_testing", "nav_cases"}
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
    assert any("C-0001" in b.label for b in at.button)

    at.selectbox(key="case_status").select("investigating").run()
    at.text_input(key="case_search").set_value("zzz").run()
    assert not any("C-0001" in b.label for b in at.button)   # the filters actually filtered

    at.button(key="case_clear").click().run()
    assert not at.exception
    assert at.selectbox(key="case_status").value == "All"
    assert at.selectbox(key="case_priority").value == "All"
    assert at.text_input(key="case_search").value == ""
    assert any("C-0001" in b.label for b in at.button)       # the case is back


def _engine():
    return load_engine(ROOT, TAG)
