"""The audit log: every score and decision, with the policy that produced it.

The live feed is deliberately stateless -- its rows live in memory, get trimmed
and disappear when the browser is refreshed. Cases are the opposite but only
cover the alerts an analyst kept. This module closes the gap in between: it
writes one row, durably, for *every* transaction the system decided on, and
stores the policy settings that were in force at that moment under a short
version hash.

That gives a complete chain for any transaction:

    transaction -> model score -> decision -> policy version -> case -> outcome

Design notes
------------
    * policy_versions holds each distinct policy ONCE (snapshot as JSON, keyed
      by its hash); a decision row only carries the hash. Changing a threshold
      therefore creates a new version instead of rewriting history, so "why was
      this blocked?" can always be answered with the settings of that moment.
    * session identifies one run of the live simulator. Transaction ids restart
      with every new simulation, so (session, txn_id) is the real identity.
    * Writes are idempotent: INSERT OR IGNORE against the unique index on
      (session, txn_id), so a retry or a double tick cannot duplicate a decision.
    * Nothing in here is ever updated. An audit trail that can be edited is not
      an audit trail.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.cases import DECISION_LABELS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session     TEXT    NOT NULL,
    txn_id      INTEGER NOT NULL,
    sim_minute  INTEGER NOT NULL,
    recorded_at TEXT    NOT NULL,
    amount      REAL    NOT NULL,
    tx_type     TEXT    NOT NULL DEFAULT '',
    score       REAL    NOT NULL,
    decision    TEXT    NOT NULL,
    reason      TEXT    NOT NULL DEFAULT '',
    model_tag   TEXT    NOT NULL DEFAULT '',
    policy_hash TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_decisions_txn    ON decisions(session, txn_id);
CREATE INDEX        IF NOT EXISTS idx_decisions_when   ON decisions(recorded_at);
CREATE INDEX        IF NOT EXISTS idx_decisions_action ON decisions(decision);
CREATE INDEX        IF NOT EXISTS idx_decisions_policy ON decisions(policy_hash);
CREATE TABLE IF NOT EXISTS policy_versions (
    hash       TEXT PRIMARY KEY,
    snapshot   TEXT NOT NULL,
    first_seen TEXT NOT NULL
);
"""

_COLUMNS = ("id", "session", "txn_id", "sim_minute", "recorded_at", "amount", "tx_type",
            "score", "decision", "reason", "model_tag", "policy_hash")


def _now() -> str:
    """UTC timestamp, second resolution (the log's own clock, not the simulation's)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Policy snapshots
# --------------------------------------------------------------------------- #
def policy_snapshot(policy) -> dict:
    """The decision settings of a Policy, in a form that survives JSON.

    Infinite thresholds (never block / review nothing) are stored as null so the
    snapshot stays valid JSON and reads as "no threshold" wherever it is shown.
    """
    def num(value: float):
        value = float(value)
        return None if math.isinf(value) else round(value, 4)

    return {
        "block_threshold": num(policy.block_threshold),
        "review_min_score": num(policy.review_min_score),
        "daily_capacity": int(policy.daily_capacity),
        "capacity_mode": str(policy.capacity_mode),
    }


def snapshot_hash(snapshot: dict) -> str:
    """Short, stable id for a policy: the same settings always share a version."""
    blob = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def threshold_text(value) -> str:
    """A threshold the way an analyst reads it: 'never', 'off' or the number."""
    if value is None:
        return "none"
    return f"{float(value):g}"


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DecisionRecord:
    """One audited decision: what arrived, what the model said, what was done."""
    number: int
    session: str
    txn_id: int
    sim_minute: int
    recorded_at: str
    amount: float
    tx_type: str
    score: float
    decision: str
    reason: str
    model_tag: str
    policy_hash: str

    @property
    def decision_label(self) -> str:
        return DECISION_LABELS.get(self.decision, self.decision.title())


class AuditLog:
    """SQLite-backed, append-only log of decisions and the policies behind them."""

    def __init__(self, path: str | Path = "data/audit.db"):
        self.path = str(path)
        self._memory = None if self.path != ":memory:" else sqlite3.connect(":memory:")
        if self._memory is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
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

    def _version_row(self, conn: sqlite3.Connection, snapshot: dict, digest: str) -> None:
        conn.execute("INSERT OR IGNORE INTO policy_versions (hash, snapshot, first_seen)"
                     " VALUES (?,?,?)",
                     (digest, json.dumps(snapshot, sort_keys=True), _now()))

    # -- writing -------------------------------------------------------- #
    def record(self, rows, policy, model_tag: str = "", session: str = "") -> int:
        """Append every row of `rows` (a scored/decided DataFrame) to the log.

        Returns how many decisions were newly written. Re-recording the same
        transaction of the same session is a no-op, never a duplicate.
        """
        if rows is None or len(rows) == 0:
            return 0
        snapshot = policy_snapshot(policy)
        digest = snapshot_hash(snapshot)
        stamp = _now()
        payload = []
        for r in rows.itertuples(index=False):
            reason = getattr(r, "reason", "")
            payload.append((
                str(session), int(r.id), int(r.minute), stamp, float(r.amount),
                str(getattr(r, "type", "")), float(r.risk), str(r.decision),
                "" if not isinstance(reason, str) else reason, str(model_tag), digest,
            ))
        conn = self._connect()
        try:
            self._version_row(conn, snapshot, digest)
            cur = conn.executemany(
                f"INSERT OR IGNORE INTO decisions ({', '.join(_COLUMNS[1:])})"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)", payload)
            conn.commit()
            return max(cur.rowcount, 0)
        finally:
            if self._memory is None:
                conn.close()

    # -- reading -------------------------------------------------------- #
    def entry(self, txn_id: int, session: str | None = None) -> DecisionRecord | None:
        """The decision recorded for one transaction (latest session if not given)."""
        sql = f"SELECT {', '.join(_COLUMNS)} FROM decisions WHERE txn_id = ?"
        params: tuple = (int(txn_id),)
        if session is not None:
            sql += " AND session = ?"
            params += (str(session),)
        sql += " ORDER BY id DESC LIMIT 1"
        conn = self._connect()
        try:
            row = conn.execute(sql, params).fetchone()
        finally:
            if self._memory is None:
                conn.close()
        return None if row is None else self._row_to_record(row)

    def list(self, decision: str | None = None, policy_hash: str | None = None,
             session: str | None = None, search: str = "", limit: int | None = 200
             ) -> list[DecisionRecord]:
        """Newest first, with the same kind of filters the cases page offers."""
        where, params = [], []
        if decision:
            where.append("decision = ?")
            params.append(decision)
        if policy_hash:
            where.append("policy_hash = ?")
            params.append(policy_hash)
        if session:
            where.append("session = ?")
            params.append(session)
        text = (search or "").strip().lstrip("#")
        if text:
            if text.isdigit():
                where.append("txn_id = ?")
                params.append(int(text))
            else:
                where.append("reason LIKE ?")
                params.append(f"%{text}%")
        sql = f"SELECT {', '.join(_COLUMNS)} FROM decisions"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        conn = self._connect()
        try:
            return [self._row_to_record(r) for r in conn.execute(sql, tuple(params))]
        finally:
            if self._memory is None:
                conn.close()

    def counts(self) -> dict[str, int]:
        conn = self._connect()
        try:
            counts = {"all": 0, "block": 0, "review": 0, "approve": 0,
                      "policies": int(conn.execute("SELECT COUNT(*) FROM policy_versions").fetchone()[0])}
            for row in conn.execute("SELECT decision, COUNT(*) FROM decisions GROUP BY decision"):
                counts[str(row[0])] = int(row[1])
                counts["all"] += int(row[1])
        finally:
            if self._memory is None:
                conn.close()
        return counts

    def policies(self) -> list[dict]:
        """Every policy version used, newest first, with how many decisions it made."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT p.hash, p.snapshot, p.first_seen, COUNT(d.id)"
                " FROM policy_versions p LEFT JOIN decisions d ON d.policy_hash = p.hash"
                " GROUP BY p.hash ORDER BY p.first_seen DESC, p.hash").fetchall()
        finally:
            if self._memory is None:
                conn.close()
        return [{"hash": r[0], "snapshot": json.loads(r[1]), "first_seen": r[2],
                 "decisions": int(r[3])} for r in rows]

    def policy(self, digest: str) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT snapshot FROM policy_versions WHERE hash = ?",
                               (digest,)).fetchone()
        finally:
            if self._memory is None:
                conn.close()
        return None if row is None else json.loads(row[0])

    # -- the whole chain ------------------------------------------------ #
    def chain(self, txn_id: int, cases=None, session: str | None = None) -> dict:
        """Everything known about one transaction, from arrival to investigation.

        `cases` is a CaseStore (optional): when given, the cases opened for this
        transaction and their event trails are included, so the caller can show
        the full path from decision to outcome.
        """
        record = self.entry(txn_id, session=session)
        linked = []
        if cases is not None:
            for case in cases.for_transaction(int(txn_id)):
                linked.append({"case": case, "events": cases.events(case.number)})
        return {"decision": record,
                "policy": self.policy(record.policy_hash) if record else None,
                "cases": linked}

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DecisionRecord:
        return DecisionRecord(number=row["id"], session=row["session"], txn_id=row["txn_id"],
                              sim_minute=row["sim_minute"], recorded_at=row["recorded_at"],
                              amount=row["amount"], tx_type=row["tx_type"], score=row["score"],
                              decision=row["decision"], reason=row["reason"],
                              model_tag=row["model_tag"], policy_hash=row["policy_hash"])
