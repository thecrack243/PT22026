"""The chat store: an append-only transcript per case (src/chat.py)."""
import pytest

from src.cases import CaseStore
from src.chat import ChatStore


@pytest.fixture
def world(tmp_path):
    return CaseStore(tmp_path / "cases.db"), ChatStore(tmp_path / "chat.db")


def test_messages_are_read_back_oldest_first(world):
    cases, chat = world
    case = cases.create(txn_id=1, risk=90.0, decision="review", amount=10.0)
    chat.append(case.number, "analyst", "why was it flagged?")
    chat.append(case.number, "assistant", "because of the receiver balance")
    chat.append(case.number, "analyst", "thanks")

    history = chat.history(case.case_id)
    assert [m.role for m in history] == ["analyst", "assistant", "analyst"]
    assert history[0].text == "why was it flagged?"
    assert history[0].from_analyst and not history[1].from_analyst
    assert chat.count(case.case_id) == 3


def test_the_transcript_of_one_case_cannot_leak_into_another(world):
    cases, chat = world
    first = cases.create(txn_id=1, risk=90.0, decision="review")
    second = cases.create(txn_id=2, risk=95.0, decision="block")
    chat.append(first.number, "analyst", "belongs to the first")
    chat.append(second.number, "analyst", "belongs to the second")

    assert [m.text for m in chat.history(first.case_id)] == ["belongs to the first"]
    assert [m.text for m in chat.history(second.case_id)] == ["belongs to the second"]


def test_an_action_survives_the_round_trip(world):
    cases, chat = world
    case = cases.create(txn_id=1, risk=90.0, decision="review")
    chat.append(case.number, "assistant", "I prepared proposal 4",
                {"kind": "proposal", "proposal": 4})

    stored = chat.history(case.case_id)[0]
    assert stored.action == {"kind": "proposal", "proposal": 4}
    assert stored.to_dict()["action"]["proposal"] == 4


def test_an_unknown_role_is_refused(world):
    cases, chat = world
    case = cases.create(txn_id=1, risk=90.0, decision="review")
    with pytest.raises(ValueError):
        chat.append(case.number, "system-operator", "hello")


def test_the_store_offers_no_way_to_rewrite_history(world):
    cases, chat = world
    assert not hasattr(chat, "update")
    assert not hasattr(chat, "delete")
    assert not hasattr(chat, "clear")


def test_a_shared_database_file_is_readable_by_a_second_store(tmp_path):
    cases = CaseStore(tmp_path / "cases.db")
    case = cases.create(txn_id=1, risk=90.0, decision="review")
    first = ChatStore(tmp_path / "chat.db")
    first.append(case.number, "analyst", "written by one process")

    second = ChatStore(tmp_path / "chat.db")            # e.g. the next Streamlit run
    assert [m.text for m in second.history(case.case_id)] == ["written by one process"]
