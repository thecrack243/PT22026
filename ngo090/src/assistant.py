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

from src.agent import CLUSTER_MIN, CLUSTER_RISK, InvestigationAgent
from src.cases import Case, PRIORITY_LABELS
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
    "I can tell you why it was flagged, what the system decided, what else looks like "
    "it, and what I would do about it. I can also change the case -- false positive, "
    "confirmed fraud, close, reopen, priority -- though those wait for your approval. "
    "Notes I just add."
)

# Social turns. They are here because a wall of capabilities in reply to "thanks"
# is the fastest way to make an assistant feel like a form.
_GREETINGS = ("hi", "hello", "hey", "yo", "good morning", "good afternoon",
              "good evening", "morning")
_ACKS = ("thanks", "thank you", "thx", "cheers", "ok", "okay", "got it",
         "sounds good", "perfect", "great", "nice", "cool", "understood")


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


# The words the grammar keys on. A misspelling of one of these is corrected
# before matching ("fals positive" -> "false positive"), because an analyst
# typing fast should not have to retype a command. Short and common words are
# left out on purpose: correcting "now" to "low" would invent a priority.
_KEYWORDS = ("false", "positive", "fraud", "fraudulent", "confirmed", "urgent",
             "high", "medium", "priority", "close", "closed", "reopen",
             "escalate", "note", "notes", "flagged", "mark")


def _edit_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein distance: counts one typo, including a swap."""
    if abs(len(a) - len(b)) > 1:
        return 2
    prev2 = None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            if i > 1 and j > 1 and ca == b[j - 2] and cb == a[i - 2]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[len(b)]


def _closest_keyword(word: str) -> str | None:
    """The keyword `word` was almost certainly meant to be -- or None.

    Strict on purpose: at most one edit away, and no second keyword at the
    same distance ("closes" sits one edit from both "close" and "closed", so
    it stays untouched). The one exception is a singular/plural pair --
    "nots" is one edit from both "note" and "notes", but the grammar treats
    them alike, so the tie is harmless and the singular wins. A wrong
    correction here files the wrong proposal; no correction at worst asks
    the analyst to retype.
    """
    if word in _KEYWORDS or len(word) < 4:
        return None
    best, best_d, tied = None, 2, []
    for candidate in _KEYWORDS:
        if abs(len(candidate) - len(word)) > 1:
            continue
        d = _edit_distance(word, candidate)
        if d < best_d:
            best, best_d, tied = candidate, d, [candidate]
        elif d == best_d:
            tied.append(candidate)
    if best is None or best_d != 1:
        return None
    if len(tied) == 1:
        return best
    shorter, longer = sorted(tied, key=len)
    if len(tied) == 2 and longer == shorter + "s":
        return shorter
    return None


def _fix_typos(text: str) -> str:
    """Correct one-edit misspellings of the keywords, everything else verbatim."""
    out, last = [], 0
    for match in re.finditer(r"[a-z]+", text):
        out.append(text[last:match.start()])
        out.append(_closest_keyword(match.group(0)) or match.group(0))
        last = match.end()
    out.append(text[last:])
    return "".join(out)


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
    # The word "note" itself misspelled ("add to the nots: ..."): the intent
    # fired on the corrected text, so recover the payload after a separator.
    # Only with a separator -- a loose match could swallow the wrong words,
    # and a note is the one place the analyst's text must stay verbatim.
    found = re.search(r"\bnot[a-z]{0,4}\b\s*[:\-\u2014]\s*(?P<text>.*)$", raw, re.IGNORECASE)
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
    if not text:                                    # nothing but politeness: "hi", "ok then"
        return Intent("greeting", raw=raw)

    # 1 -- somebody is being polite, not asking about the case -----------
    # Answered on its own: a capability list in reply to "thanks" is the
    # fastest way to make an assistant feel like a form.
    if text in _GREETINGS:
        return Intent("greeting", raw=raw)
    if text in _ACKS or re.match(r"^thank(?:s| you)\b", text):
        return Intent("ack", raw=raw)

    # 2 -- a request for the manual --------------------------------------
    if re.match(r"^(help|commands?|what can you do|how can you help|what do you do)\b", text) \
            or text == "?":
        return Intent("help", raw=raw)

    # 2b -- spelling. One-edit misspellings of the keywords are corrected so
    # "fals positive" still lands. The original message stays on the
    # transcript untouched, and a note's text is read from the raw message
    # below, so a misspelled word inside a note is never rewritten.
    text = _fix_typos(text)

    # 3 -- a note: "add this to the note: ...", "note: ..." --------------
    if re.match(r"^notes?\s*[:\-]", text) or (
            re.match(r"^(?:add|append|write|put)\b", text) and re.search(r"\bnotes?\b", text)):
        return Intent("note", _note_text(raw), raw)

    command = bool(re.match(r"^(?:" + "|".join(_COMMAND_VERBS) + r")\b", text))

    # 4 -- a resolution. Both are stated before the generic "close". -----
    if command and re.search(r"\bfalse[\s-]?positive\b", text):
        return Intent("close_false_positive", raw=raw)
    if command and re.search(r"\b(?:confirmed\s+)?(?:fraud|fraudulent)\b", text):
        return Intent("close_fraud", raw=raw)
    if re.match(r"^(?:it'?s|this is|that'?s)\s+(?:a\s+|clearly\s+|just\s+)?false[\s-]?positive\b", text):
        return Intent("close_false_positive", raw=raw)
    if re.match(r"^(?:it'?s|this is|that'?s)\s+(?:clearly\s+|confirmed\s+)?fraud", text):
        return Intent("close_fraud", raw=raw)

    # 5 -- "close it", "mark it closed": which resolution? asked later.
    if re.match(r"^close\b", text) or re.match(r"^mark\b.*\bas closed\b", text):
        return Intent("close", raw=raw)

    # 6 -- priority ------------------------------------------------------
    priority = re.search(r"\b(urgent|high|medium|low)\b", text)
    if priority and (command or "priority" in text):
        return Intent("priority", priority.group(1), raw)
    if re.match(r"^escalate\b", text):
        return Intent("priority", "urgent", raw)

    # 7 -- reopen --------------------------------------------------------
    if re.match(r"^reopen\b", text) or re.search(r"\breopen (?:the |this )?case\b", text):
        return Intent("reopen", raw=raw)

    # 8 -- a question, or something outside the grammar ------------------
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


_TOPICS = (
    # "What is the difference between status and resolution?" comes first: a
    # question about the words is not a question about the case, and the field
    # names below would otherwise capture it.
    ("vocabulary", ("difference between", "what does resolution", "what does status",
                    "meaning of", "what is a resolution", "what is resolution",
                    "what is priority", "what is a state", "status vs", "resolution vs")),
    # The verdict on its own, ahead of "should i ..." in `recommend`: "should I
    # change the resolution?" is a question about the resolution, and reading
    # it as a general recommendation answered about priority instead.
    ("resolution", ("resolution", "verdict", "outcome")),
    ("recommend", ("recommend", "should i", "what should", "next step", "advice", "advise",
                   "suggest", "what do you think", "your call", "worth escalating", "would you",
                   "do you think i should")),
    ("judge", ("is it fraud", "is this fraud", "fraud or not", "legit", "is it real",
               "should i worry", "how bad", "how serious", "is it safe")),
    ("how", ("how did you decide", "how do you know", "which rule", "what rule",
             "how did you get", "walk me through")),
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
    text = _fix_typos(_core(question))
    for name, words in _TOPICS:
        if any(word in text for word in words):
            return name
    return ""


def _asks_to_close(question: str) -> bool:
    """Did the analyst ask about closing, rather than just "what should I do"?

    "Should I close it?" wants a yes or a no first; a recommendation that opens
    with "it's already urgent" answers a different question.
    """
    return bool(re.search(r"\b(close|closing|shut|clear)\b", _fix_typos(_core(question))))


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

        if intent.kind in ("help", "greeting", "ack"):
            return self._social(case, intent)
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

    # -- social turns --------------------------------------------------- #
    def _social(self, case: Case, intent: Intent) -> Reply:
        """Hello, thanks, and the manual.

        Short on purpose: these are the turns where an assistant most easily
        sounds like a leaflet, and a paragraph of capabilities in reply to
        "thanks" is exactly the wrong answer.
        """
        if intent.kind == "greeting":
            kind = (case.tx_type or "").replace("_", " ").lower()
            return self._say(
                case,
                f"Hi. I've read {case.case_id}: {kind} of {_amount(case.amount)} at risk "
                f"{case.risk:.1f}, {case.state_label.lower()}. What do you want to know?",
                intent)
        if intent.kind == "ack":
            return self._say(case, "Anytime. Say the word if you want a change written up.", intent)
        return self._say(case, f"I'm {self.name}. {CAPABILITIES}", intent)

    # -- questions ------------------------------------------------------ #
    def _answer(self, case: Case, question: str) -> Reply:
        topic = _topic(question)
        needs_investigation = topic in ("", "recommend", "judge", "how", "why", "related",
                                        "policy", "risk", "transaction", "status", "resolution")
        investigation = None
        if needs_investigation:
            try:
                investigation = self.agent.investigate(case.number)
            except ToolError:
                investigation = None

        if topic == "recommend":
            text = self._recommend_text(case, investigation, question)
        elif topic == "resolution":
            text = self._resolution_text(case, investigation, question)
        elif topic == "vocabulary":
            text = self._vocabulary_text(case)
        elif topic == "judge":
            text = self._judge_text(case, investigation)
        elif topic == "how":
            text = self._how_text(case, investigation)
        elif topic == "why":
            text = self._why_text(case, investigation)
        elif topic == "related":
            text = self._related_text(case, investigation)
        elif topic == "policy":
            text = self._policy_text(case, investigation)
        elif topic == "risk":
            text = self._risk_text(case, investigation)
        elif topic == "notes":
            text = self._notes_text(case)
        elif topic == "history":
            text = self._history_text(case)
        elif topic == "transaction":
            text = self._transaction_text(case, investigation)
        elif topic == "status":
            text = self._status_text(case)
        else:
            text = self._summary_text(case, investigation)
        return Reply(text, "question", action={"kind": "answer", "topic": topic}, case=case)

    @staticmethod
    def _exact(investigation) -> bool:
        return investigation is not None and any(e.source == "live feed" for e in investigation.evidence)

    # -- the answers, in words a person would use ------------------------ #
    def _summary_text(self, case: Case, investigation) -> str:
        """"Where are we?" -- what it is, what stands out, what I'd do."""
        kind = (case.tx_type or "transaction").replace("_", " ").lower()
        parts = [f"{case.case_id}: {_article(kind)} {kind} of {_amount(case.amount)} at risk "
                 f"{case.risk:.1f}, opened {_when(case.created_at)}."]
        up = ([_fact(r) for r in investigation.explanation.reasons_up[:1]]
              if investigation is not None and investigation.explanation else [])
        if up:
            parts.append(f"What stands out: {_clauses(up)}.")
        elif case.explanation:
            parts.append(f"The note on it: {_plain(case.explanation)}.")
        rec = investigation.recommendation if investigation is not None else None
        if rec is not None and rec.kind == "escalate":
            label = PRIORITY_LABELS.get(rec.proposed_priority or "urgent", "Urgent").lower()
            parts.append(f"I'd make it {label}.")
        elif rec is not None and rec.kind == "close_false_positive":
            parts.append("I'd call it a false positive.")
        elif rec is not None and rec.kind == "close_fraud":
            parts.append("I'd call it fraud.")
        else:
            parts.append(f"I'd leave it at {case.priority_label.lower()}.")
        parts.append("Ask me for the details, or tell me what to change.")
        return " ".join(parts)

    def _why_text(self, case: Case, investigation) -> str:
        if investigation is None or investigation.explanation is None:
            return (f"The case was opened at risk {case.risk:.1f}. The note that came with it: "
                    f"\"{_plain(case.explanation)}\". I can't re-score it here -- the "
                    f"transaction's balances aren't in this session -- so that's all I can vouch "
                    f"for.")
        risk = investigation.risk
        up = [_fact(r) for r in investigation.explanation.reasons_up[:3]]
        down = [_fact(r) for r in investigation.explanation.reasons_down[:1]]
        text = f"Risk {risk:.1f} out of 100."
        if up:
            count = {1: "One thing pushed it there", 2: "Two things pushed it there"}.get(
                len(up), "Three things pushed it there")
            text += f" {count}: {_clauses(up)}."
        if down:
            text += f" The counterweight: {_clauses(down)}."
        if not self._exact(investigation):
            text += (" I couldn't re-score it in this session, so those are the figures recorded "
                     "when the case was opened.")
        return text

    def _judge_text(self, case: Case, investigation) -> str:
        if investigation is None or investigation.explanation is None:
            return (f"I can't tell you whether it's fraud -- nobody can from one transaction. What "
                    f"the record says is risk {case.risk:.1f}, decision \"{case.decision}\".")
        up = [_fact(r) for r in investigation.explanation.reasons_up[:2]]
        down = [_fact(r) for r in investigation.explanation.reasons_down[:1]]
        parts = ["Honestly? I can't tell you for certain -- nobody can from one transaction."]
        if up:
            parts.append(f"What points at fraud: {_clauses(up)}.")
        if down:
            parts.append(f"What points the other way: {_clauses(down)}.")
        if investigation.recommendation.kind == "close_false_positive":
            parts.append("On balance I'd clear it.")
        else:
            parts.append("So no, I wouldn't clear it on this evidence.")
        return " ".join(parts)

    def _how_text(self, case: Case, investigation) -> str:
        """The audit answer, for the one question that asks for the rule."""
        if investigation is None or investigation.recommendation is None:
            return ("I work from a fixed list of rules over the evidence, and I can't run it here: "
                    "the transaction's balances aren't in this session.")
        rec = investigation.recommendation
        text = ("I check a fixed list of rules in order and stop at the first that matches. "
                f"{rec.rule} fired: {rec.rationale}.")
        if investigation.assessment is not None:
            text += f" The policy's own rule reads: \"{investigation.assessment.rule}\"."
        return text

    def _risk_text(self, case: Case, investigation) -> str:
        risk = investigation.risk if investigation is not None else case.risk
        if risk >= 95.0:
            tail = "That's near the top of the scale -- not a borderline call."
        elif risk >= 80.0:
            tail = "High enough that the system hands it to a person instead of clearing it."
        else:
            tail = "Low enough that it wouldn't have been flagged on its own."
        text = f"{risk:.1f} out of 100. {tail}"
        if investigation is not None and not self._exact(investigation):
            text += (" That's the score recorded when the case was opened: I couldn't re-score it "
                     "in this session.")
        return text

    def _policy_text(self, case: Case, investigation) -> str:
        decision = investigation.policy_decision if investigation is not None else case.decision
        if investigation is None or investigation.assessment is None:
            return (f"The case records \"{decision}\" at risk {case.risk:.1f}. I can't re-run the "
                    f"policy without the transaction's balances, so I won't invent a newer answer.")
        if decision == BLOCK:
            return ("It was blocked. At this score the policy stops the transaction on its own, so "
                    "the money shouldn't have moved. What's left for you is whether the block was "
                    "the right call.")
        if decision == REVIEW:
            return ("It went to review rather than block: at this score the policy hands the "
                    "decision to a person instead of stopping the card by itself. That's why it's "
                    "in your queue.")
        return ("The policy cleared it -- it scored below the level this system acts on. It's only "
                "here because somebody kept the alert anyway.")

    def _related_text(self, case: Case, investigation) -> str:
        if investigation is None or investigation.related is None:
            gap = (investigation.gaps[0] if investigation and investigation.gaps
                   else "the live row for this transaction is not available in this session")
            first = gap.split(". ")[0].rstrip(".")
            return (f"I can't check that here: {_first_lower(first)}. Open it from the live feed "
                    f"and I'll search.")
        found = investigation.related
        if not found.items:
            return (f"I compared it with {found.examined:,} others and none share enough to count "
                    f"as related. Nothing to tie it to.")
        lines = []
        for item in found.items[:3]:
            tx = item.transaction
            why = ", ".join(item.relation) if item.relation else "shared evidence"
            lines.append(f"#{tx['id']} ({tx['type'].replace('_', ' ').lower()} of "
                         f"{_amount(tx['amount'])} -- {why})")
        return (f"{len(found.items)} of {found.examined:,} look related: "
                + "; ".join(lines) + ".")

    def _recommend_text(self, case: Case, investigation, question: str = "") -> str:
        """The suggestion: the opinion first, then why, then how to act on it."""
        if investigation is None or investigation.recommendation is None:
            return (f"I can't give you a real recommendation without the transaction's balances. "
                    f"All I have is what the case recorded: risk {case.risk:.1f}, "
                    f"{case.status_label.lower()}, {case.priority_label.lower()} priority. Open it "
                    f"from the live feed and I'll have more to go on.")
        rec = investigation.recommendation
        risk = investigation.risk
        decision = investigation.policy_decision
        cluster = sum(1 for item in (investigation.related.items if investigation.related else ())
                      if (item.risk or 0.0) >= CLUSTER_RISK)

        if rec.kind == "escalate":
            label = PRIORITY_LABELS.get(rec.proposed_priority or "urgent", "Urgent").lower()
            if cluster >= CLUSTER_MIN:
                reason = (f"{cluster} look-alikes around it also score above "
                          f"{CLUSTER_RISK:.0f}, and it's still sitting at "
                          f"{case.priority_label.lower()} priority")
            else:
                reason = (f"risk {risk:.1f} is at the top of the scale and it's still sitting at "
                          f"{case.priority_label.lower()} priority")
            text = f"I'd make it {label}. {_capital(reason)}."
            if rec.requires_override:
                text += (" That does mean going against the system's own call, so it needs an "
                         "override from you.")
            else:
                text += " Nothing moves until you approve it."
            ask = "make it urgent" if label == "urgent" else f"raise it to {label}"
            text += f" Say \"{ask}\" and I'll put it up for approval."

        elif rec.kind == "close_false_positive":
            text = ("I'd call this a false positive. The policy approves the transaction and "
                    "nothing around it supports the alert.")
            text += (" Say \"mark it false positive\" and I'll put it up for approval -- nothing "
                     "moves until you approve it.")

        elif rec.kind == "close_fraud":
            text = ("I'd call this confirmed fraud. The evidence points one way and nothing in the "
                    "record argues with it.")
            text += " Say \"mark it confirmed fraud\" and I'll put it up for approval."

        elif decision == BLOCK:
            text = (f"Nothing to do here. It's already blocked and "
                    f"{case.priority_label.lower()} -- I'd leave it and give the time to "
                    f"something else.")

        elif decision == APPROVE and cluster < CLUSTER_MIN:
            text = ("Honestly, I can't call this one either way. The policy approves it, but "
                    "somebody kept the alert, and I don't have enough to close it over their "
                    "judgement. I'd keep it open for a person to look at.")
        elif cluster >= CLUSTER_MIN:
            text = (f"It's already {case.priority_label.lower()}, and with {cluster} look-alikes "
                    f"around it that's the right slot. I wouldn't file a change.")
        else:
            text = (f"It's already {case.priority_label.lower()}. Risk {risk:.1f} is well past the "
                    f"review line, so escalating wouldn't buy you anything -- I'd leave it where it "
                    f"is and work it.")

        if _asks_to_close(question):
            text = ("Yes, I would. " + text if rec.kind == "close_false_positive"
                    else "Not yet. " + text)
        return text

    def _transaction_text(self, case: Case, investigation) -> str:
        if investigation is not None and self._exact(investigation):
            tx = investigation.transaction
            return (f"#{tx['id']}: {tx['type'].replace('_', ' ').lower()} of "
                    f"{_amount(tx['amount'])}, at {tx['hour']:02d}:00. Before it the sender held "
                    f"{_amount(tx['oldbalanceOrg'])} and the receiver "
                    f"{_amount(tx['oldbalanceDest'])}.")
        return (f"The case records #{case.txn_id}: "
                f"{(case.tx_type or 'unknown type').replace('_', ' ').lower()} of "
                f"{_amount(case.amount)}. The full row isn't in this session's feed, so I can't "
                f"tell you the balances.")

    def _notes_text(self, case: Case) -> str:
        if not case.notes:
            return ("Nothing in the notes yet. Say \"add this to the note: ...\" and I'll write it "
                    "down.")
        notes = " ".join(case.notes.split())
        if len(notes) > 320:
            notes = notes[:317].rstrip() + "..."
        return f"The notes say: \"{notes}\""

    def _history_text(self, case: Case) -> str:
        found = self.tools.get_case(case.number)
        events = found.events[:3] if found is not None else ()
        if not events:
            return f"Nothing yet -- no edit has touched {case.case_id} since it was opened."
        lines = "; ".join(f"{e.detail} ({_when(e.created_at)})" for e in events)
        return f"The last {len(events)}: {lines}."

    def _status_text(self, case: Case) -> str:
        """Where the case stands: state, resolution and priority, one word each.

        The three fields are orthogonal, so the answer names all three instead
        of letting "status" stand for whichever one the reader had in mind.
        """
        priority = case.priority_label.lower()
        if case.closed:
            verdict = ("without a verdict" if case.status == "investigating"
                       else f"as {case.status_label.lower()}")
            return (f"Closed {verdict}. {_capital(priority)} priority, last change "
                    f"{_when(case.updated_at)}.")
        if case.status == "investigating":
            text = f"Open, still investigating -- no verdict yet. {_capital(priority)} priority"
            if str(case.created_at) == str(case.updated_at):
                return text + f", and nothing has touched it since it opened {_when(case.created_at)}."
            return text + f", last change {_when(case.updated_at)}."
        return (f"Open, with the resolution already set to {case.status_label.lower()} -- it isn't "
                f"closed yet. {_capital(priority)} priority, last change {_when(case.updated_at)}.")

    def _resolution_text(self, case: Case, investigation, question: str = "") -> str:
        """The verdict field on its own: what it says, and whether to change it."""
        if case.status == "investigating":
            text = ("The resolution is still investigating -- no verdict recorded. It becomes "
                    "false positive or confirmed fraud when you decide, and it's separate from "
                    "the case's state (open or closed) and from its priority.")
        else:
            text = f"The resolution says {case.status_label.lower()}."

        asks = bool(re.search(r"should i|change|update|set it|different|right call",
                              _core(question)))
        if not asks:
            return text
        rec = investigation.recommendation if investigation is not None else None
        if rec is None:
            return text + " I can't tell you whether to change it without the transaction's balances."
        if rec.kind == "close_false_positive":
            return (text + " On this evidence I'd set it to false positive and close -- say \"mark "
                           "it false positive\" and I'll put it up for approval.")
        if rec.kind == "close_fraud":
            return (text + " On this evidence I'd set it to confirmed fraud -- say \"mark it "
                           "confirmed fraud\" and I'll put it up for approval.")
        if case.status != "investigating":
            # A verdict is already on record: recommending "investigating" would
            # read as if the case had none.
            return (text + " I wouldn't change it: nothing in the evidence argues for a different "
                           "verdict than the one on record.")
        return (text + " I wouldn't set one yet: nothing in the evidence supports either verdict, "
                       "so investigating is the honest answer.")

    def _vocabulary_text(self, case: Case) -> str:
        """The three fields, in the words the rest of the app now uses."""
        return ("Three separate things, and they move independently. State -- open or closed: "
                "whether you're still working the case; here it's "
                f"{case.state_label.lower()}. Resolution -- investigating, confirmed fraud or "
                "false positive: the verdict on the transaction; here it's "
                f"{case.status_label.lower()}. Priority -- low, medium, high or urgent: how hard "
                "to work it; here it's " + case.priority_label.lower() + ". The system's own "
                "decision (approved, sent to review, blocked) is a fourth field, and nobody edits "
                "it.")

    @staticmethod
    def _boundary(case: Case) -> str:
        return (f"I didn't follow that one. Try \"why was it flagged?\", \"what do you "
                f"recommend?\" or \"mark it false positive\" -- I only work on {case.case_id}.")


# --------------------------------------------------------------------------- #
# Saying it like a person
# --------------------------------------------------------------------------- #
def _amount(value) -> str:
    """A number the way somebody would say it out loud: 477k, not 477,066.97."""
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return "an unknown amount"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if abs(number) >= 10_000:
        return f"{number / 1_000:.0f}k"
    return f"{number:,.0f}"


def _fact(reason) -> str:
    """One engine reason as a clause: "amount 1.26M where 172k is typical"."""
    text = str(getattr(reason, "text", "") or "").strip()
    found = re.search(r"^(.*?)\s*\(typical ([^)]*)\)\s*$", text)
    if found:
        text = f"{found.group(1)} where {found.group(2)} is typical"
    text = _spoken_numbers(text)
    if text[:1].isupper() and not text[:2].isupper():     # keep CASH_OUT as it is
        text = text[:1].lower() + text[1:]
    return text


def _spoken_numbers(text: str) -> str:
    """1,258,316 -> 1.26M. A figure read aloud is a shorter figure."""
    return re.sub(r"\b\d{1,3}(?:,\d{3})+\b", lambda m: _amount(m.group(0).replace(",", "")), text)


def _clauses(items) -> str:
    """["a", "b", "c"] -> "a, b and c"."""
    items = [str(i) for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _plain(explanation: str) -> str:
    """"Transaction amount - Amount 1.26M (typical 172k)" -> "amount 1.26M (typical 172k)".

    The stored explanation is a list of "label - fact" lines; read aloud, the
    label is noise.
    """
    text = " ".join(str(explanation or "").split())
    if " - " in text:
        text = text.split(" - ", 1)[1]
    return _first_lower(_spoken_numbers(text))


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _first_lower(text: str) -> str:
    """A sentence that has to sit mid-sentence: "The live row..." -> "the live row..."."""
    if text[:2].isupper():                                # an acronym, leave it
        return text
    return text[:1].lower() + text[1:]


def _when(value: str) -> str:
    """A stored UTC timestamp, as "3 hours ago" -- or the date when that is old."""
    from datetime import datetime, timezone

    try:
        when = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return "at an unknown time"
    now = datetime.now(timezone.utc) if when.tzinfo else datetime.now()
    seconds = (now - when).total_seconds()
    if seconds < 0:
        return when.astimezone().strftime("%d %b, %H:%M")
    if seconds < 90:
        return "just now"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)} minute{'s' if minutes >= 2 else ''} ago"
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)} hour{'s' if hours >= 2 else ''} ago"
    days = hours / 24
    if days < 7:
        return f"{int(days)} day{'s' if days >= 2 else ''} ago"
    return when.astimezone().strftime("%d %b %Y")
