"""Proposals: what an assistant recommends, waiting for a human to agree.

The boundary this module exists to enforce is simple and worth stating plainly:

    the agent may read anything and propose anything, but it may not change a case

An investigation ends with a row in this table, not with an edit to `cases`.
A person then approves (which runs the change through `src/tools.update_case`,
the same function an analyst's buttons use) or rejects it, and either way the
proposal keeps its record. Nothing here is ever deleted: a rejected proposal is
evidence that a human disagreed, which is exactly the kind of thing an audit
looks for.

Only one proposal per case can be pending. Asking for a second investigation
supersedes the first rather than stacking contradictory recommendations in the
queue; the superseded row keeps its text so the history stays readable.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.cases import parse_case_id

# What the agent is asking for. `keep_open` is the "no change" case: the
# investigation found nothing that justifies moving the case, which is a
# legitimate outcome and should still be visible to a person.
KINDS = ("escalate", "close_fraud", "close_false_positive", "keep_open")
STATUSES = ("pending", "approved", "rejected", "superseded")

KIND_LABELS = {
    "escalate": "Escalate",
    "close_fraud": "Close as confirmed fraud",
    "close_false_positive": "Close as false positive",
    "keep_open": "Keep open",
}
STATUS_LABELS = {
    "pending": "Awaiting approval",
    "approved": "Approved",
    "rejected": "Rejected",
    "superseded": "Superseded",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS proposals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id           INTEGER NOT NULL,
    kind              TEXT    NOT NULL,
    status            TEXT    NOT NULL DEFAULT 'pending',
    proposed_status   TEXT,
    proposed_state    TEXT,
    proposed_priority TEXT,
    risk              REAL    NOT NULL DEFAULT 0,
    policy_decision   TEXT    NOT NULL DEFAULT '',
    rationale         TEXT    NOT NULL DEFAULT '',
    narrative         TEXT    NOT NULL DEFAULT '',
    evidence          TEXT    NOT NULL DEFAULT '{}',
    agent             TEXT    NOT NULL DEFAULT '',
    requires_override INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT    NOT NULL,
    decided_at        TEXT,
    decided_by        TEXT,
    decision_note     TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_proposals_case   ON proposals(case_id);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status);
"""

_COLUMNS = ("id", "case_id", "kind", "status", "proposed_status", "proposed_state",
            "proposed_priority", "risk", "policy_decision", "rationale", "narrative",
            "evidence", "agent", "requires_override", "created_at", "decided_at",
            "decided_by", "decision_note")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Proposal:
    """One recommendation, with the evidence it was based on and who decided."""
    number: int
    case_number: int
    kind: str
    status: str = "pending"
    proposed_status: str | None = None
    proposed_state: str | None = None
    proposed_priority: str | None = None
    risk: float = 0.0
    policy_decision: str = ""
    rationale: str = ""
    narrative: str = ""
    evidence: dict | None = None
    agent: str = ""
    requires_override: bool = False
    created_at: str = ""
    decided_at: str | None = None
    decided_by: str | None = None
    decision_note: str = ""

    @property
    def case_id(self) -> str:
        from src.cases import case_id as render

        return render(self.case_number)

    @property
    def kind_label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)

    @property
    def pending(self) -> bool:
        return self.status == "pending"

    @property
    def changes(self) -> list[str]:
        """What approving this would actually do, in the analyst's words."""
        out = []
        if self.proposed_status:
            from src.cases import STATUS_LABELS as LABELS

            out.append(f"Resolution -> {LABELS.get(self.proposed_status, self.proposed_status)}")
        if self.proposed_state == "closed":
            out.append("State -> Closed")
        elif self.proposed_state == "open":
            out.append("State -> Open")
        if self.proposed_priority:
            from src.cases import PRIORITY_LABELS as LABELS

            out.append(f"Priority -> {LABELS.get(self.proposed_priority, self.proposed_priority)}")
        return out

    @property
    def changes_anything(self) -> bool:
        return bool(self.changes)

    def to_dict(self) -> dict:
        return {"proposal": int(self.number), "case": self.case_id, "kind": self.kind,
                "status": self.status, "risk": float(self.risk),
                "policy_decision": self.policy_decision,
                "changes": self.changes, "rationale": self.rationale,
                "narrative": self.narrative, "evidence": self.evidence or {},
                "agent": self.agent, "requires_override": bool(self.requires_override),
                "created_at": self.created_at, "decided_at": self.decided_at,
                "decided_by": self.decided_by, "decision_note": self.decision_note}


class ProposalStore:
    """SQLite-backed queue of recommendations awaiting a person.

    Same conventions as the case and audit stores: a connection per operation,
    WAL, no ORM, `:memory:` for tests.
    """

    def __init__(self, path: str | Path = "data/proposals.db"):
        self.path = str(path)
        self._memory = None if self.path != ":memory:" else sqlite3.connect(":memory:")
        if self._memory is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            if self._memory is None:          # ':memory:' is shared, so it must stay open
                conn.close()

    # -- plumbing ------------------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        if self._memory is not None:
            conn = self._memory
            conn.row_factory = sqlite3.Row
        else:
            conn = sqlite3.connect(self.path, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def close(self) -> None:
        if self._memory is not None:
            self._memory.close()
            self._memory = None

    @staticmethod
    def _row(row: sqlite3.Row) -> Proposal:
        try:
            evidence = json.loads(row["evidence"] or "{}")
        except json.JSONDecodeError:              # a hand-edited database should not crash the page
            evidence = {}
        return Proposal(
            number=row["id"], case_number=row["case_id"], kind=row["kind"], status=row["status"],
            proposed_status=row["proposed_status"], proposed_state=row["proposed_state"],
            proposed_priority=row["proposed_priority"], risk=row["risk"],
            policy_decision=row["policy_decision"], rationale=row["rationale"],
            narrative=row["narrative"], evidence=evidence, agent=row["agent"],
            requires_override=bool(row["requires_override"]), created_at=row["created_at"],
            decided_at=row["decided_at"], decided_by=row["decided_by"],
            decision_note=row["decision_note"],
        )

    # -- reading -------------------------------------------------------- #
    def get(self, number: int | str) -> Proposal | None:
        conn = self._connect()
        try:
            row = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM proposals WHERE id = ?",
                               (int(number),)).fetchone()
        finally:
            if self._memory is None:
                conn.close()
        return None if row is None else self._row(row)

    def pending_for(self, case: int | str) -> Proposal | None:
        """The proposal currently waiting for a decision on this case, if any."""
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM proposals"
                " WHERE case_id = ? AND status = 'pending' ORDER BY id DESC LIMIT 1",
                (parse_case_id(case),)).fetchone()
        finally:
            if self._memory is None:
                conn.close()
        return None if row is None else self._row(row)

    def for_case(self, case: int | str, limit: int = 20) -> list[Proposal]:
        """Every proposal ever made for a case, newest first."""
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM proposals"
                " WHERE case_id = ? ORDER BY id DESC LIMIT ?",
                (parse_case_id(case), int(limit))).fetchall()
        finally:
            if self._memory is None:
                conn.close()
        return [self._row(r) for r in rows]

    def list(self, status: str | None = None, limit: int = 200) -> list[Proposal]:
        """Proposals newest first; `status` filters to one lifecycle state."""
        if status is not None and status not in STATUSES:
            raise ValueError(f"Unknown proposal status {status!r}. Use one of {', '.join(STATUSES)}.")
        sql = f"SELECT {', '.join(_COLUMNS)} FROM proposals"
        params: tuple = ()
        if status:
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY id DESC LIMIT ?"
        conn = self._connect()
        try:
            rows = conn.execute(sql, params + (int(limit),)).fetchall()
        finally:
            if self._memory is None:
                conn.close()
        return [self._row(r) for r in rows]

    def pending(self, limit: int = 200) -> list[Proposal]:
        """The approval queue, oldest first: the one that has waited longest comes first."""
        items = self.list(status="pending", limit=limit)
        return list(reversed(items))

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in STATUSES}
        conn = self._connect()
        try:
            for row in conn.execute("SELECT status, COUNT(*) FROM proposals GROUP BY status"):
                out[str(row[0])] = int(row[1])
        finally:
            if self._memory is None:
                conn.close()
        out["all"] = sum(out[s] for s in STATUSES)
        return out

    # -- writing -------------------------------------------------------- #
    def create(self, case: int | str, kind: str, *, rationale: str = "", narrative: str = "",
               evidence: dict | None = None, risk: float = 0.0, policy_decision: str = "",
               proposed_status: str | None = None, proposed_state: str | None = None,
               proposed_priority: str | None = None, agent: str = "",
               requires_override: bool = False) -> Proposal:
        """File a recommendation. Any pending proposal for the case is superseded."""
        if kind not in KINDS:
            raise ValueError(f"Unknown proposal kind {kind!r}. Use one of {', '.join(KINDS)}.")
        number = parse_case_id(case)
        stamp = _now()
        conn = self._connect()
        try:
            conn.execute("UPDATE proposals SET status = 'superseded', decided_at = ?"
                         " WHERE case_id = ? AND status = 'pending'", (stamp, number))
            cur = conn.execute(
                "INSERT INTO proposals (case_id, kind, status, proposed_status, proposed_state,"
                " proposed_priority, risk, policy_decision, rationale, narrative, evidence,"
                " agent, requires_override, created_at)"
                " VALUES (?,?,'pending',?,?,?,?,?,?,?,?,?,?,?)",
                (number, kind, proposed_status, proposed_state, proposed_priority, float(risk),
                 str(policy_decision), str(rationale), str(narrative),
                 json.dumps(evidence or {}, sort_keys=True), str(agent),
                 1 if requires_override else 0, stamp))
            conn.commit()
            fresh = int(cur.lastrowid)
        finally:
            if self._memory is None:
                conn.close()
        found = self.get(fresh)
        if found is None:                          # cannot happen; keeps the return type honest
            raise RuntimeError("The proposal could not be read back after writing it.")
        return found

    def decide(self, number: int | str, approved: bool, *, by: str = "analyst",
               note: str = "") -> Proposal:
        """Record a human's verdict. Only a pending proposal can be decided, once."""
        current = self.get(number)
        if current is None:
            raise KeyError(f"Proposal {number!r} does not exist.")
        if not current.pending:
            raise ValueError(f"Proposal {number} is already {current.status_label.lower()}.")
        stamp = _now()
        conn = self._connect()
        try:
            conn.execute("UPDATE proposals SET status = ?, decided_at = ?, decided_by = ?,"
                         " decision_note = ? WHERE id = ?",
                         ("approved" if approved else "rejected", stamp, str(by), str(note),
                          int(current.number)))
            conn.commit()
        finally:
            if self._memory is None:
                conn.close()
        found = self.get(current.number)
        if found is None:
            raise RuntimeError("The proposal disappeared while it was being decided.")
        return found
