"""RiskPilot: the analyst's assistant, in conversation, about one case.

What it is
----------
The same investigation assistant as `src/agent.py`, with a mouth. An analyst can
ask why a case was flagged, what the policy decided, what else looks like it,
what the notes say -- and can also ask for a change ("mark this false positive",
"add this to the note", "make it urgent"). Questions are answered from the
tool layer only. Changes follow exactly the same path as the panel's "Send for
approval": RiskPilot files a proposal, and the analyst approves it. The chat adds
a place to *talk*; it does not add a way to bypass the gate.

Two rules keep the conversation honest
-------------------------------------
    * It only talks about its case. There is no cross-case search here, so a
      question about another case gets a refusal, not a plausible answer.
    * It never guesses. The intent is parsed deterministically (a fixed grammar,
      see `parse`), and anything it does not understand comes back as a
      question with the list of things it can do. No fuzzy matching that might
      close the wrong case.

The grammar is small on purpose. Every command the analyst can give is a
sentence RiskPilot can prove it understood -- and the proposal it files shows
the change in words before it lands, which is the confirmation step.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.agent import InvestigationAgent
from src.cases import Case, STATUS_LABELS, PRIORITY_LABELS
from src.chat import ChatStore
from src.decision import APPROVE, BLOCK, REVIEW
from src.proposals import Proposal, ProposalStore
from src.tools import Toolkit, ToolError

ASSISTANT_NAME = "RiskPilot"

# The verbs that make a sentence an instruction rather than a question.
_COMMAND_VERBS = ("mark", "close", "set", "flag", "resolve", "treat", "classify",
                  "add", "append", "write", "put", "reopen", "escalate", "make",
                  "change", "raise", "lower", "bump", "move")
_POLITE = re.compile(r"^(?:please|kindly|can you|could you|would you|will you|hey|hi|ok|okay)\b[\s,]*")

MAX_NOTE = 2_000          # a note longer than this is a paste, not a note

CAPABILITIES = (
    "Ask me why the case was flagged, what the risk and policy decision are, what "
    "related transactions exist, or what the notes and history say. I can also add "
    "to the notes, mark it false positive or confirmed fraud, close or reopen it, "
    "and change its priority -- but a change only lands after you approve it."
)


# --------------------------------------------------------------------------- #
# Understanding the analyst
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Intent:
    """What the analyst asked for, after parsing. `detail` is the payload."""
    kind: str                 # help / note / close_false_positive / close_fraud / close
                              # priority / reopen / question / unknown
    detail: str = ""
    raw: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "raw": self.raw}


def _core(message: str) -> str:
    """The sentence without politeness, punctuation and extra spaces."""
    text = " ".join(str(message or "").split()).strip().lower()
    while True:
        stripped = _POLITE.sub("", text)
        if stripped == text:
            break
        text = stripped
    return text.strip(" .!")


def _is_question(text: str) -> bool:
    if text.endswith("?"):
        return True
    return bool(re.match(r"^(why|what|how|when|where|who|which|whose|does|did|do|is|are|"
                         r"was|were|can|could|should|would|tell|show|explain|summari[sz]e|"
                         r"any|list|give)\b", text))


def _note_text(raw: str) -> str:
    """The part of an 'add this to the note ...' sentence that is the note.

    Read from the original message, not the lowercased copy: a note keeps the
    analyst's capitalisation ("Called the customer, they confirmed the transfer").
    """
    found = re.search(r"\bnotes?\b\s*[:\-\u2014]\s*(?P<text>.*)$", raw, re.IGNORECASE)
    if found:
        return found.group("text").strip()
    found = re.search(r"\bnotes?\b\s*(?P<text>.*)$", raw, re.IGNORECASE)
    if not found:
        return ""
    rest = found.group("text").strip(" :-,.")
    for lead in ("saying that ", "saying ", "that ", "with ", "as "):
        if rest.lower().startswith(lead):
            rest = rest[len(lead):].strip()
    return rest


def parse(message: str) -> Intent:
    """The fixed grammar: one sentence in, one intent out.

    The order is deliberate. Commands are recognised before questions, because
    "can you mark this false positive?" is a request; and questions are
    recognised before the fallback, because "why is it marked false positive?"
    must never be read as an instruction.
    """
    raw = " ".join(str(message or "").split()).strip()
    text = _core(raw)
    if not text:
        return Intent("unknown", raw=raw)

    # 1 -- a greeting or a request for help ----------------------------
    if re.match(r"^(help|commands?|what can you do|how can you help|what do you do)\b", text) \
            or text in ("hi", "hello", "hey", "?"):
        return Intent("help", raw=raw)

    # 2 -- a note: "add this to the note: ...", "note: ..." --------------
    if re.match(r"^notes?\s*[:\-]", text) or (
            re.match(r"^(?:add|append|write|put)\b", text) and re.search(r"\bnotes?\b", text)):
        return Intent("note", _note_text(raw), raw)

    command = bool(re.match(r"^(?:" + "|".join(_COMMAND_VERBS) + r")\b", text))

    # 3 -- a resolution. Both are stated before the generic "close". -----
    if command and re.search(r"\bfalse[\s-]?positive\b", text):
        return Intent("close_false_positive", raw=raw)
    if command and re.search(r"\b(?:confirmed\s+)?(?:fraud|fraudulent)\b", text):
        return Intent("close_fraud", raw=raw)
    if re.match(r"^(?:it'?s|this is|that'?s)\s+(?:a\s+|clearly\s+|just\s+)?false[\s-]?positive\b", text):
        return Intent("close_false_positive", raw=raw)
    if re.match(r"^(?:it'?s|this is|that'?s)\s+(?:clearly\s+|confirmed\s+)?fraud", text):
        return Intent("close_fraud", raw=raw)

    # 4 -- "close it", "mark it closed": which resolution? asked later.
    if re.match(r"^close\b", text) or re.match(r"^mark\b.*\bas closed\b", text):
        return Intent("close", raw=raw)

    # 5 -- priority ------------------------------------------------------
    priority = re.search(r"\b(urgent|high|medium|low)\b", text)
    if priority and (command or "priority" in text):
        return Intent("priority", priority.group(1), raw)
    if re.match(r"^escalate\b", text):
        return Intent("priority", "urgent", raw)

    # 6 -- reopen --------------------------------------------------------
    if re.match(r"^reopen\b", text) or re.search(r"\breopen (?:the |this )?case\b", text):
        return Intent("reopen", raw=raw)

    # 7 -- a question, or something outside the grammar ------------------
    if _is_question(text):
        return Intent("question", raw=raw)
    return Intent("unknown", raw=raw)


# --------------------------------------------------------------------------- #
# What the assistant says back
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Reply:
    """One answer, and -- when it did or asked for something -- the record of it."""
    text: str
    intent: str
    action: dict | None = None
    proposal: Proposal | None = None
    case: Case | None = None

    @property
    def changed_case(self) -> bool:
        return bool((self.action or {}).get("kind") == "note")


def _verb(decision: str) -> str:
    return {BLOCK: "blocks it", REVIEW: "sends it to review",
            APPROVE: "approves it"}.get(decision, f'decides "{decision}"')


def _money(value: float) -> str:
    return f"{float(value):,.0f}"


def _stamp(value: str) -> str:
    """A stored UTC timestamp, in the reader's local time."""
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(value)).astimezone().strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return "an unknown time"


_TOPICS = (
    ("recommend", ("recommend", "should i", "what should", "next step", "advice", "advise",
                   "suggest", "what do you think", "your call", "worth escalating")),
    ("why", ("why", "reason", "flagged", "suspicious", "explain")),
    ("related", ("related", "similar", "cluster", "look-alike", "lookalike",
                 "other transaction", "same account", "linked")),
    ("policy", ("policy", "threshold", "rule", "decision", "auto-block", "block", "approve",
                "review")),
    ("risk", ("risk", "score", "probability", "confidence")),
    ("notes", ("note", "comment")),
    ("history", ("history", "timeline", "events", "who worked", "updated", "changed")),
    ("transaction", ("transaction", "amount", "details", "hour", "balance", "sender",
                     "receiver")),
    ("status", ("status", "state", "priority", "open", "closed", "resolution")),
)


def _contradicts(decision: str, kind: str) -> bool:
    """Would this change go against what the Risk Manager said?

    Not an override in the blocking sense -- the analyst asked for it, so there
    is no checkbox -- but the reply says so, because a person reading the case
    later deserves to know the two disagreed.
    """
    if kind == "close_fraud":
        return decision == APPROVE
    if kind == "close_false_positive":
        return decision == BLOCK
    return False


def _topic(question: str) -> str:
    text = _core(question)
    for name, words in _TOPICS:
        if any(word in text for word in words):
            return name
    return ""


class RiskPilot:
    """One assistant, one case at a time, reading through the tool layer.

    `proposals` is where a requested change goes; `chat` is where the
    conversation is kept. Both are optional so a test or a script can drive the
    conversation without a database.
    """

    name = ASSISTANT_NAME

    def __init__(self, tools: Toolkit, *, proposals: ProposalStore | None = None,
                 chat: ChatStore | None = None, agent: InvestigationAgent | None = None,
                 name: str = ASSISTANT_NAME):
        self.tools = tools
        self.proposals = proposals
        self.chat = chat
        self.agent = agent or InvestigationAgent(tools, proposals=proposals)
        self.name = name

    # -- the public entry points ---------------------------------------- #
    def converse(self, case_id: str | int, message: str) -> Reply:
        """One turn of the conversation, with both halves written down.

        The analyst's message is stored before anything else happens, so a
        failure to answer still leaves the question on the record.
        """
        text = " ".join(str(message or "").split()).strip()
        if not text:
            raise ToolError("The message is empty.")
        if self.chat is not None:
            self.chat.append(case_id, "analyst", text)
        try:
            reply = self.reply(case_id, text)
        except ToolError as exc:
            reply = Reply(f"I couldn't do that: {exc}", "error",
                          action={"kind": "error"}, case=None)
        if self.chat is not None:
            self.chat.append(case_id, "assistant", reply.text, reply.action)
        return reply

    def reply(self, case_id: str | int, message: str) -> Reply:
        """Answer without touching the transcript. The work happens here."""
        found = self.tools.get_case(case_id)
        if found is None:
            raise ToolError(f"Case {case_id!r} does not exist.")
        case = found.case
        intent = parse(message)

        if intent.kind == "help":
            return self._say(case, f"I'm {self.name}. I work on {case.case_id} only. "
                                   f"{CAPABILITIES}", intent)
        if intent.kind == "note":
            return self._note(case, intent)
        if intent.kind == "close_false_positive":
            return self._close(case, intent, "false_positive")
        if intent.kind == "close_fraud":
            return self._close(case, intent, "confirmed_fraud")
        if intent.kind == "close":
            return self._close(case, intent, None)
        if intent.kind == "priority":
            return self._priority(case, intent)
        if intent.kind == "reopen":
            return self._reopen(case, intent)
        if intent.kind == "question":
            return self._answer(case, message)
        return self._say(case, self._boundary(case), intent, action={"kind": "clarify"})

    # -- actions: notes are direct, everything else is a proposal -------- #
    def _note(self, case: Case, intent: Intent) -> Reply:
        text = intent.detail.strip()
        if not text:
            return self._say(
                case,
                "What should I add? Write it after the word note, for example: "
                "\"add this to the note: called the customer, they confirmed the transfer\".",
                intent, action={"kind": "clarify"})
        if len(text) > MAX_NOTE:
            return self._say(case, f"That note is {len(text):,} characters long. Keep it under "
                                   f"{MAX_NOTE:,} and I'll add it.", intent,
                             action={"kind": "clarify"})
        # The one write the assistant does on its own: appending a note is
        # additive, attributed, and cannot change a status or close a case.
        self.tools.update_case(case.case_id, note=f"Note added via {self.name}: {text}")
        return self._say(case, f"Added to the notes on {case.case_id}: \"{text}\"", intent,
                         action={"kind": "note", "note": text})

    def _close(self, case: Case, intent: Intent, status: str | None) -> Reply:
        if status is None:
            if case.closed:
                return self._noop(case, f"{case.case_id} is already closed as "
                                        f"{case.status_label.lower()}.", intent)
            if case.status == "investigating":
                return self._say(
                    case,
                    "Which resolution? I won't pick one for you: say \"mark it false positive\" "
                    "or \"mark it confirmed fraud\", and I'll prepare that for your approval.",
                    intent, action={"kind": "clarify"})
            status = case.status                      # the resolution already on the case
        closing_now = not case.closed
        wanted = {"false_positive": "false positive", "confirmed_fraud": "confirmed fraud"}[status]
        summary = (f"mark {case.case_id} as {wanted} and close it" if closing_now
                   else f"change the resolution of {case.case_id} to {wanted}")
        return self._propose(case, intent,
                             kind="close_false_positive" if status == "false_positive"
                             else "close_fraud",
                             summary=summary, status=status,
                             state="closed" if closing_now else None)

    def _priority(self, case: Case, intent: Intent) -> Reply:
        level = intent.detail
        if level not in PRIORITY_LABELS:
            return self._say(case, "Which priority: low, medium, high or urgent?", intent,
                             action={"kind": "clarify"})
        return self._propose(case, intent, kind="priority",
                             summary=f"set the priority of {case.case_id} to {PRIORITY_LABELS[level]}",
                             priority=level)

    def _reopen(self, case: Case, intent: Intent) -> Reply:
        if not case.closed:
            return self._noop(case, f"{case.case_id} is already open "
                                    f"({case.status_label.lower()}).", intent)
        return self._propose(case, intent, kind="reopen",
                             summary=f"reopen {case.case_id}", state="open")

    def _propose(self, case: Case, intent: Intent, *, kind: str, summary: str,
                 status: str | None = None, state: str | None = None,
                 priority: str | None = None) -> Reply:
        """File what the analyst asked for, and show it before it lands."""
        if self.proposals is None:
            raise ToolError("This assistant has no proposal queue, so it cannot change a case.")
        current_state = "closed" if case.closed else "open"
        changes = [
            (status, case.status),
            (state, current_state),
            (priority, case.priority),
        ]
        if not any(value and value != now for value, now in changes):
            return self._noop(case, f"{case.case_id} already matches that: "
                                    f"{case.state_label.lower()}, "
                                    f"{case.status_label.lower()}, "
                                    f"{case.priority_label.lower()} priority.", intent)

        replaced = self.proposals.pending_for(case.number)
        proposal = self.proposals.create(
            case.case_id, kind, rationale=f"The analyst asked me in chat: {summary}.",
            risk=float(case.risk), policy_decision=case.decision, proposed_status=status,
            proposed_state=state, proposed_priority=priority, agent=f"{self.name.lower()}/chat")

        text = (f"Understood: {summary}. I prepared proposal {proposal.number} -- approve it "
                f"here and I'll apply it through the same path an analyst's own edit uses.")
        if case.closed and state is None:
            text += f" The case is closed at the moment; this changes its resolution only."
        if replaced is not None and replaced.number != proposal.number:
            text += f" It replaces proposal {replaced.number}, which was still waiting."
        if _contradicts(case.decision, kind):
            text += (f" Note that the Risk Manager said \"{case.decision}\" for this "
                     f"transaction, so this goes against its decision -- your call, and the "
                     f"proposal records it.")
        return Reply(text, intent.kind, action={"kind": "proposal", "proposal": proposal.number},
                     proposal=proposal, case=case)

    def _noop(self, case: Case, text: str, intent: Intent) -> Reply:
        return Reply(text, intent.kind, action={"kind": "noop"}, case=case)

    def _say(self, case: Case, text: str, intent: Intent,
             action: dict | None = None) -> Reply:
        return Reply(text, intent.kind, action=action or {"kind": "answer"}, case=case)

    # -- questions ------------------------------------------------------ #
    def _answer(self, case: Case, question: str) -> Reply:
        topic = _topic(question)
        needs_investigation = topic in ("recommend", "why", "related", "policy", "risk",
                                        "transaction", "status")
        investigation = None
        if needs_investigation:
            try:
                investigation = self.agent.investigate(case.number)
            except ToolError:
                investigation = None

        if topic == "recommend":
            text = self._recommend_text(case, investigation)
        elif topic == "why":
            text = self._why_text(case, investigation)
        elif topic == "related":
            text = self._related_text(case, investigation)
        elif topic == "policy":
            text = self._policy_text(case, investigation)
        elif topic == "risk":
            text = self._risk_text(case, investigation)
        elif topic == "notes":
            text = (f"The notes read: \"{case.notes}\"" if case.notes
                    else f"There are no notes on {case.case_id} yet. Say \"add this to the "
                         f"note: ...\" and I'll add one.")
        elif topic == "history":
            text = self._history_text(case)
        elif topic == "transaction":
            text = self._transaction_text(case, investigation)
        elif topic == "status":
            text = self._status_text(case)
        else:
            text = (f"Here is where {case.case_id} stands: {self._status_text(case)} "
                    f"Ask me why it was flagged, what is related, or what I recommend.")
        return Reply(text, "question", action={"kind": "answer", "topic": topic}, case=case)

    @staticmethod
    def _exact(investigation) -> bool:
        return investigation is not None and any(e.source == "live feed" for e in investigation.evidence)

    def _why_text(self, case: Case, investigation) -> str:
        decision = investigation.policy_decision if investigation else case.decision
        parts = [f"{case.case_id} was opened at risk {case.risk:.1f}, and the policy "
                 f"{_verb(decision)}."]
        reasons = (investigation.explanation.reasons_up[:3]
                   if investigation and investigation.explanation else ())
        if reasons:
            parts.append("What drove the score: "
                         + "; ".join(f"{r.label} -- {r.text}" for r in reasons) + ".")
        elif case.explanation:
            parts.append(f"The explanation recorded with the case: {case.explanation}.")
        if investigation and investigation.assessment is not None:
            parts.append(f"Why the policy decided that: {investigation.assessment.rule}.")
        if investigation and investigation.gaps and not self._exact(investigation):
            parts.append("I could not re-score it in this session, so those are the numbers "
                         "recorded when the case was opened.")
        return " ".join(parts)

    def _risk_text(self, case: Case, investigation) -> str:
        if investigation and investigation.assessment is not None:
            return (f"Risk {investigation.risk:.1f} out of 100. The policy "
                    f"{_verb(investigation.policy_decision)}: {investigation.assessment.rule}.")
        return (f"The engine could not score the transaction again in this session, so the "
                f"score recorded when the case was opened stands: {case.risk:.1f} out of 100, "
                f"decision \"{case.decision}\".")

    def _policy_text(self, case: Case, investigation) -> str:
        if investigation and investigation.assessment is not None:
            policy = investigation.assessment.policy
            return (f"The policy says \"{investigation.policy_decision}\" for this transaction: "
                    f"{investigation.assessment.rule} (auto-block at "
                    f"{policy.get('block_threshold', '-')}, review from "
                    f"{policy.get('review_min_score', '-')}, capacity "
                    f"{policy.get('daily_capacity', '-')} a day). The decision belongs to the "
                    f"policy, not to me.")
        return (f"The case records the decision \"{case.decision}\" at risk {case.risk:.1f}. I "
                f"cannot re-run the policy without the transaction's balances, so I will not "
                f"invent a newer answer.")

    def _related_text(self, case: Case, investigation) -> str:
        if investigation is None or investigation.related is None:
            gap = (investigation.gaps[0] if investigation and investigation.gaps
                   else "the live row for this transaction is not available in this session")
            return f"I could not search related activity: {gap.rstrip('.')}."
        found = investigation.related
        if not found.items:
            return (f"I compared it with {found.examined:,} transactions and none shared enough "
                    f"evidence to count as related.")
        lines = []
        for item in found.items[:3]:
            risk = f", risk {item.risk:.1f}" if item.risk is not None else ""
            why = ", ".join(item.relation) if item.relation else "shared evidence"
            lines.append(f"#{item.transaction['id']} "
                         f"({item.transaction['type'].replace('_', ' ').lower()} of "
                         f"{_money(item.transaction['amount'])}{risk}: {why})")
        return (f"{len(found.items)} related transaction(s) out of {found.examined:,} examined -- "
                + "; ".join(lines) + ".")

    def _recommend_text(self, case: Case, investigation) -> str:
        if investigation is None:
            return (f"I could not build a recommendation without the transaction's values. The "
                    f"case as recorded: {case.state_label.lower()}, "
                    f"{case.status_label.lower()}, {case.priority_label.lower()} priority, "
                    f"risk {case.risk:.1f}.")
        rec = investigation.recommendation
        text = f"My recommendation is {rec.label.lower()} (rule {rec.rule}): {rec.rationale}."
        if rec.changes_anything:
            text += " Nothing changes until you approve it."
        if rec.requires_override:
            text += " It also contradicts the policy's own decision, so approving it takes an " \
                    "explicit override."
        return text

    def _transaction_text(self, case: Case, investigation) -> str:
        if investigation is not None and self._exact(investigation):
            tx = investigation.transaction
            return (f"Transaction #{tx['id']}: {tx['type'].replace('_', ' ').title()} of "
                    f"{_money(tx['amount'])} at simulated hour {tx['hour']:02d}:00. Sender balance "
                    f"{_money(tx['oldbalanceOrg'])}, receiver balance {_money(tx['oldbalanceDest'])}.")
        return (f"The case records transaction #{case.txn_id}: "
                f"{(case.tx_type or 'unknown type').replace('_', ' ').title()} of "
                f"{_money(case.amount)}. The full row is not in this session's feed, so the "
                f"balances are not something I can vouch for.")

    def _history_text(self, case: Case) -> str:
        found = self.tools.get_case(case.number)
        events = found.events[:3] if found is not None else ()
        if not events:
            return f"Nothing has been recorded on {case.case_id} yet."
        return (f"The last events on {case.case_id}: "
                + "; ".join(f"{e.detail} ({_stamp(e.created_at)})" for e in events) + ".")

    def _status_text(self, case: Case) -> str:
        return (f"{case.case_id} is {case.state_label.lower()} with resolution "
                f"{case.status_label.lower()} and {case.priority_label.lower()} priority. "
                f"Opened {_stamp(case.created_at)}, last updated {_stamp(case.updated_at)}.")

    @staticmethod
    def _boundary(case: Case) -> str:
        return (f"I only work on {case.case_id}, and I would rather ask than guess. "
                f"{CAPABILITIES}")
