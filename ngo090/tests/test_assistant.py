"""RiskPilot: the grammar, the answers, and the boundary (src/assistant.py).

These tests never touch Streamlit. They build the same toolkit, proposal queue
and transcript the app builds, and drive the assistant exactly as the chat box
does -- which is the point of the tool layer: a conversation is a script.
"""
import pytest

from src.agent import AGENT_TAG, approve, reject
from src.assistant import ASSISTANT_NAME, RiskPilot, parse
from src.cases import CaseStore
from src.chat import ChatStore
from src.proposals import ProposalStore
from src.tools import Toolkit


class World:
    """One case, one queue, one transcript, one assistant."""

    def __init__(self, tmp_path, **case_fields):
        self.root = tmp_path
        self.cases = CaseStore(tmp_path / "cases.db")
        self.queue = ProposalStore(tmp_path / "proposals.db")
        self.chat = ChatStore(tmp_path / "chat.db")
        fields = {"txn_id": 986, "risk": 95.1, "decision": "review", "amount": 477_066.97,
                  "tx_type": "CASH_OUT", "source": "live",
                  "explanation": "Receiver balance - Receiver balance 0 (typical 558,465)"}
        fields.update(case_fields)
        self.case = self.cases.create(**fields)
        self.tools = Toolkit(cases=self.cases)
        self.pilot = RiskPilot(self.tools, proposals=self.queue, chat=self.chat)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


# --------------------------------------------------------------------------- #
# The grammar
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("message, kind, detail", [
    ("mark it as false positive", "close_false_positive", ""),
    ("Mark this case as closed", "close", ""),
    ("close as confirmed fraud", "close_fraud", ""),
    ("Can you mark this as false positive?", "close_false_positive", ""),
    ("it's a false positive", "close_false_positive", ""),
    ("add this in the note: Called the customer", "note", "Called the customer"),
    ("note: the customer confirmed the transfer", "note", "the customer confirmed the transfer"),
    ("add a note saying the customer confirmed", "note", "the customer confirmed"),
    ("add this to the note", "note", ""),
    ("set priority to high", "priority", "high"),
    ("make it urgent", "priority", "urgent"),
    ("escalate", "priority", "urgent"),
    ("reopen the case", "reopen", ""),
    ("help", "help", ""),
    ("what can you do?", "help", ""),
    ("hello", "greeting", ""),
    ("hi", "greeting", ""),
    ("thanks", "ack", ""),
    ("thank you for that", "ack", ""),
])
def test_commands_are_read_as_commands(message, kind, detail):
    intent = parse(message)
    assert (intent.kind, intent.detail) == (kind, detail)


@pytest.mark.parametrize("message", [
    "Why was it flagged?",
    "why did you mark it false positive?",
    "what is the risk?",
    "show me related transactions",
    "what do you recommend?",
    "is it a false positive?",
    "what is the status?",
])
def test_questions_are_never_read_as_instructions(message):
    assert parse(message).kind == "question"


def test_a_note_keeps_the_analysts_capitalisation():
    assert parse("add this to the note: Called the customer, they CONFIRMED") \
        .detail == "Called the customer, they CONFIRMED"


# --------------------------------------------------------------------------- #
# Typos. One edit away from a keyword is corrected; nothing cleverer, because
# a wrong correction files the wrong proposal.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("message, kind, detail", [
    ("mark it as fals positive", "close_false_positive", ""),
    ("Mark it as flase positive", "close_false_positive", ""),
    ("mark it as fraudulant", "close_fraud", ""),
    ("set priority to urgant", "priority", "urgent"),
    ("clsoe it", "close", ""),
    ("mrak it as closed", "close", ""),
    ("escalte", "priority", "urgent"),
    ("add to the nots: Called the customer", "note", "Called the customer"),
])
def test_a_one_edit_typo_still_lands(message, kind, detail):
    intent = parse(message)
    assert (intent.kind, intent.detail) == (kind, detail)


def test_a_misspelled_question_still_finds_its_topic():
    assert parse("is it fradulent?").kind == "question"
    from src.assistant import _topic
    assert _topic("is it fradulent?") == "judge"
    assert _topic("why was it flaged?") == "why"


def test_an_ambiguous_word_is_left_alone():
    from src.assistant import _closest_keyword
    # "closes" sits one edit from both "close" and "closed" -- no correction.
    assert _closest_keyword("closes") is None
    assert parse("mark it as closes").kind == "unknown"


def test_common_words_are_never_corrected_into_commands():
    # "now" is one edit from "low"; correcting it would invent a priority.
    assert parse("set priority now").kind != "priority" or \
        parse("set priority now").detail != "low"


def test_a_note_is_not_spellchecked():
    # The analyst's words go in the case record exactly as typed.
    intent = parse("add to the note: customer says it was fraudulant")
    assert intent.kind == "note"
    assert "fraudulant" in intent.detail


# --------------------------------------------------------------------------- #
# What it says
# --------------------------------------------------------------------------- #
def test_the_assistant_has_a_name_and_is_not_a_decision_maker():
    assert ASSISTANT_NAME == "RiskPilot"


def test_a_greeting_is_a_greeting_not_a_manual(world):
    """The failure this guards against: a leaflet in reply to "hello"."""
    reply = world.pilot.converse(world.case.case_id, "hello")
    assert world.case.case_id in reply.text
    assert len(reply.text) < 220, "a greeting is a line, not a capabilities dump"
    assert world.chat.count(world.case.case_id) == 2       # the question and the answer


def test_help_is_the_one_turn_that_lists_what_it_can_do(world):
    for message in ("help", "what can you do?"):
        text = world.pilot.converse(world.case.case_id, message).text
        assert "RiskPilot" in text
        assert "approval" in text.lower() or "approve" in text.lower()


def test_thanks_does_not_earn_a_capability_list(world):
    text = world.pilot.converse(world.case.case_id, "thanks").text
    assert "Anytime" in text
    assert len(text) < 120


def test_why_is_answered_from_the_case_record_not_invented(world):
    text = world.pilot.converse(world.case.case_id, "why was it flagged?").text
    assert "95.1" in text
    assert "The note that came with it" in text             # the explanation stored on the case
    assert "receiver balance" in text.lower()               # the explanation stored on the case
    assert "can't re-score" in text                         # no live feed here: says so


def test_status_and_notes_come_from_the_case(world):
    world.tools.update_case(world.case.case_id, note="Called the customer, they confirmed")
    status = world.pilot.converse(world.case.case_id, "what is the status?").text
    assert "open" in status.lower() and "investigating" in status
    assert world.case.priority_label.lower() in status.lower()

    notes = world.pilot.converse(world.case.case_id, "what are the notes?").text
    assert "Called the customer" in notes


def test_status_names_the_three_fields_one_word_each(world):
    # "Status" used to stand for whichever of state / resolution / priority the
    # reader had in mind; the answer now names all three.
    text = world.pilot.converse(world.case.case_id, "what is the status?").text
    assert text.startswith("Open")
    assert "no verdict yet" in text
    assert "priority" in text


def test_a_question_about_the_resolution_is_not_read_as_a_recommendation(world):
    # Regression: "should i change it?" matched the "recommend" topic first, so
    # a question about the resolution was answered with a rant about priority.
    text = world.pilot.converse(
        world.case.case_id, "What about the resolution? should i change it ?").text
    assert "resolution" in text.lower()
    assert "escalating" not in text


def test_it_can_explain_its_own_vocabulary(world):
    text = world.pilot.converse(
        world.case.case_id, "what is the difference between status and resolution?").text
    assert "open or closed" in text
    assert "verdict" in text
    assert "low, medium, high or urgent" in text


def test_a_verdict_on_record_is_not_talked_back_to_investigating(tmp_path):
    # Regression: a case already marked confirmed fraud got "investigating is
    # the honest answer" -- it read as if no verdict had been recorded.
    w = World(tmp_path, status="confirmed_fraud")
    text = w.pilot._resolution_text(w.case, FakeInvestigation(kind="keep_open"),
                                    "should i change it?")
    assert "confirmed fraud" in text
    assert "investigating is the honest answer" not in text


def test_history_lists_the_real_events(world):
    text = world.pilot.converse(world.case.case_id, "what is the history?").text
    assert "Opened from alert #986" in text


def test_an_unknown_sentence_gets_a_boundary_not_a_guess(world):
    reply = world.pilot.converse(world.case.case_id, "transfer 500 to bob")
    assert reply.action["kind"] == "clarify"
    assert world.case.case_id in reply.text
    assert world.queue.counts()["all"] == 0
    assert world.cases.get(world.case.case_id) == world.case


def test_it_will_not_talk_about_a_case_it_was_not_given(world):
    with pytest.raises(Exception):
        world.pilot.reply("C-9999", "why was it flagged?")


# --------------------------------------------------------------------------- #
# What it does -- and does not do
# --------------------------------------------------------------------------- #
def test_a_note_is_written_directly_and_attributed(world):
    reply = world.pilot.converse(world.case.case_id,
                                 "add this in the note: Called the customer, they confirmed")
    assert reply.action == {"kind": "note", "note": "Called the customer, they confirmed"}
    stored = world.cases.get(world.case.case_id)
    assert "Called the customer, they confirmed" in stored.notes
    assert f"Note added via {ASSISTANT_NAME}" in stored.notes      # who wrote it
    assert stored.status == "investigating" and not stored.closed   # nothing else moved


def test_a_note_without_content_asks_rather_than_writes(world):
    reply = world.pilot.converse(world.case.case_id, "add this to the note")
    assert reply.action["kind"] == "clarify"
    assert world.cases.get(world.case.case_id).notes == ""


def test_a_requested_close_is_a_proposal_and_the_case_does_not_move(world):
    reply = world.pilot.converse(world.case.case_id, "mark it as false positive")
    assert reply.action["kind"] == "proposal"
    proposal = world.queue.get(reply.proposal.number)
    assert proposal.kind == "close_false_positive"
    assert proposal.proposed_status == "false_positive"
    assert proposal.proposed_state == "closed"
    assert proposal.pending
    assert proposal.agent == "riskpilot/chat"
    assert "asked me in chat" in proposal.rationale

    untouched = world.cases.get(world.case.case_id)
    assert untouched.status == "investigating" and not untouched.closed
    assert untouched.notes == ""


def test_approving_the_chat_proposal_is_what_moves_the_case(world):
    reply = world.pilot.converse(world.case.case_id, "mark it as false positive")
    result = approve(world.queue, world.tools, reply.proposal, by="analyst")

    assert result.case.status == "false_positive" and result.case.closed
    assert "asked me in chat" in result.case.notes
    assert world.queue.get(reply.proposal.number).status == "approved"
    # and the case record says who did it, through the same update_case path
    assert "Assistant (riskpilot/chat)" in result.case.notes


def test_rejecting_leaves_the_case_exactly_as_it_was(world):
    reply = world.pilot.converse(world.case.case_id, "mark it as confirmed fraud")
    reject(world.queue, reply.proposal, by="analyst")

    assert world.cases.get(world.case.case_id) == world.case
    assert world.queue.get(reply.proposal.number).status == "rejected"


def test_closing_without_a_resolution_asks_which_one(world):
    reply = world.pilot.converse(world.case.case_id, "close the case")
    assert reply.action["kind"] == "clarify"
    assert "false positive" in reply.text and "confirmed fraud" in reply.text
    assert world.queue.counts()["all"] == 0


def test_a_change_that_is_already_true_files_nothing(world):
    world.cases.update(world.case.case_id, priority="urgent")
    reply = world.pilot.converse(world.case.case_id, "make it urgent")
    assert reply.action["kind"] == "noop"
    assert world.queue.counts()["all"] == 0

    reopen = world.pilot.converse(world.case.case_id, "reopen the case")
    assert reopen.action["kind"] == "noop"
    assert "already open" in reopen.text


def test_a_second_request_replaces_the_first_and_says_so(world):
    first = world.pilot.converse(world.case.case_id, "mark it as false positive")
    second = world.pilot.converse(world.case.case_id, "set priority to urgent")

    assert second.proposal.number != first.proposal.number
    assert world.queue.get(first.proposal.number).status == "superseded"
    assert "replaces proposal" in second.text
    assert [p.number for p in world.queue.pending()] == [second.proposal.number]


def test_a_close_that_contradicts_the_policy_says_so_but_needs_no_checkbox(tmp_path):
    """The analyst asked; the override tick is for the assistant's own ideas."""
    world = World(tmp_path, risk=99.2, decision="block")
    reply = world.pilot.converse(world.case.case_id, "mark it as false positive")
    assert reply.proposal.requires_override is False
    assert "Risk Manager said" in reply.text


def test_the_transcript_keeps_a_failed_turn(world):
    world.pilot.converse(world.case.case_id, "why was it flagged?")
    world.cases.delete(world.case.case_id)                  # deleted in another tab

    reply = world.pilot.converse(world.case.case_id, "mark it as false positive")
    assert reply.intent == "error"
    assert "couldn't do that" in reply.text
    assert [m.role for m in world.chat.history(world.case.case_id)][-2:] == \
           ["analyst", "assistant"]                          # the question is still on record


def test_one_case_conversation_cannot_change_another(world):
    other = world.cases.create(txn_id=42, risk=99.0, decision="block", amount=5_000.0,
                               tx_type="TRANSFER", source="live")
    world.pilot.converse(world.case.case_id, "mark it as false positive")

    assert world.queue.pending_for(other.number) is None
    assert world.cases.get(other.case_id) == other
    assert world.chat.count(other.case_id) == 0


def test_the_rule_set_is_still_what_recommends(world):
    """The chat understands requests; it does not replace the rules."""
    world.pilot.converse(world.case.case_id, "what do you recommend?")
    assert AGENT_TAG.startswith("rules:")


# --------------------------------------------------------------------------- #
# The voice: an opinion first, prose a person would use
# --------------------------------------------------------------------------- #
class FakeInvestigation:
    """The shape InvestigationAgent.investigate returns, built by hand.

    The point of these tests is the wording, so the evidence is a stub: what
    matters is that the sentence is built from facts and opens with a position.
    """

    def __init__(self, kind="escalate", priority="urgent", risk=99.2, decision="review",
                 override=False):
        from types import SimpleNamespace as N

        self.risk = risk
        self.policy_decision = decision
        self.evidence = [N(source="live feed")]
        self.explanation = N(
            reasons_up=[N(text="Amount 1,258,316 (typical 171,704)"),
                        N(text="Receiver balance 0 (typical 558,465)")],
            reasons_down=[N(text="Receiving account was empty")])
        self.related = None
        self.gaps = ()
        self.assessment = N(rule="risk 99.2 is at or above the review threshold")
        self.recommendation = N(kind=kind, rule="R3", proposed_priority=priority,
                                requires_override=override, label=kind,
                                rationale="risk is high")


def speaking(world, **kwargs):
    """A world whose investigation returns a known recommendation."""
    world.pilot.agent.investigate = lambda case_id: FakeInvestigation(**kwargs)
    return world


def test_the_recommendation_opens_with_the_position(world):
    text = speaking(world).pilot.converse(world.case.case_id, "what do you recommend?").text
    assert text.lower().startswith("i'd make it urgent"), text
    assert "rule r" not in text.lower(), "no rule codes in a recommendation"
    assert "99.2" in text


def test_a_close_question_gets_a_yes_or_no_before_the_reasoning(world):
    text = speaking(world).pilot.converse(world.case.case_id, "should I close it?").text
    assert text.startswith("Not yet."), text
    assert "I'd make it urgent" in text, "then the advice behind the no"


def test_a_false_positive_recommendation_says_yes_to_closing(world):
    text = speaking(world, kind="close_false_positive").pilot.converse(
        world.case.case_id, "should I close this?").text
    assert text.startswith("Yes, I would."), text
    assert "false positive" in text


def test_the_rule_number_is_there_for_whoever_asks_for_it(world):
    """Auditability without putting rule codes in every answer."""
    text = speaking(world).pilot.converse(world.case.case_id, "how did you decide?").text
    assert "R3 fired" in text


def test_figures_are_spoken_not_pasted_from_the_database(world):
    text = speaking(world).pilot.converse(world.case.case_id, "why was it flagged?").text
    assert "1.26M" in text and "1,258,316" not in text
    assert "172k" in text
    assert "Two things pushed it there" in text
    assert "The counterweight" in text


def test_a_question_it_cannot_place_gets_a_short_answer_and_a_next_step(world):
    text = speaking(world).pilot.converse(world.case.case_id, "what does it look like?").text
    assert text.startswith(world.case.case_id)
    assert "Ask me for the details" in text
