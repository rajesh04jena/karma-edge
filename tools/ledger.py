################################################################################
# Karma Edge - tools/ledger.py
#
# The Accountability Ledger. Append-only. Every finding gets a dollar number,
# an owner, a confidence, and a timestamp. Nothing is ever silently rewritten,
# because that is how "nobody's fault" happens in the first place.
################################################################################
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import settings
from tools.sql_tools import load_semantic_layer

DDL = """
CREATE TABLE IF NOT EXISTS findings (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    run_id          TEXT,
    title           TEXT NOT NULL,
    dimension       TEXT NOT NULL,     -- overstock|stockout|discount_depth|...
    entity_type     TEXT,              -- sku|category|store|vendor
    entity_id       TEXT,
    dollar_impact   REAL NOT NULL,     -- annualised margin at stake
    confidence      REAL NOT NULL,     -- 0..1
    owner_function  TEXT NOT NULL,
    evidence        TEXT NOT NULL,     -- JSON: tool calls + numbers behind it
    recommendation  TEXT,
    status          TEXT NOT NULL DEFAULT 'open',
    superseded_by   TEXT
);
CREATE TABLE IF NOT EXISTS finding_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    finding_id  TEXT NOT NULL,
    at          TEXT NOT NULL,
    actor       TEXT NOT NULL,         -- agent name or human user
    action      TEXT NOT NULL,         -- created|critiqued|approved|rejected|closed
    note        TEXT
);
CREATE INDEX IF NOT EXISTS ix_findings_dim ON findings(dimension);
CREATE INDEX IF NOT EXISTS ix_findings_entity ON findings(entity_type, entity_id);
"""


def _conn() -> sqlite3.Connection:
    path = Path(settings.ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def owner_for(dimension: str) -> str:
    return (load_semantic_layer().get("ownership") or {}).get(dimension, "unassigned")


def write_finding(
    title: str,
    dimension: str,
    dollar_impact: float,
    confidence: float,
    evidence: Dict[str, Any] | str,
    entity_type: str = "sku",
    entity_id: str = "",
    recommendation: str = "",
    owner_function: Optional[str] = None,
    run_id: Optional[str] = None,
    actor: str = "agent",
) -> str:
    fid = f"KE-{uuid.uuid4().hex[:8].upper()}"
    conn = _conn()
    conn.execute(
        "INSERT INTO findings (id,created_at,run_id,title,dimension,entity_type,entity_id,"
        "dollar_impact,confidence,owner_function,evidence,recommendation,status) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?, 'open')",
        (
            fid, _now(), run_id, title, dimension, entity_type, entity_id,
            float(dollar_impact), float(confidence),
            owner_function or owner_for(dimension),
            evidence if isinstance(evidence, str) else json.dumps(evidence, default=str),
            recommendation,
        ),
    )
    conn.execute(
        "INSERT INTO finding_events (finding_id,at,actor,action,note) VALUES (?,?,?,?,?)",
        (fid, _now(), actor, "created", title),
    )
    conn.commit()
    conn.close()
    return fid


def list_findings(status: str = "open", limit: int = 25) -> List[Dict[str, Any]]:
    conn = _conn()
    q = "SELECT * FROM findings"
    args: tuple = ()
    if status and status != "all":
        q += " WHERE status = ?"
        args = (status,)
    q += " ORDER BY dollar_impact DESC LIMIT ?"
    rows = [dict(r) for r in conn.execute(q, args + (limit,))]
    conn.close()
    return rows


def record_event(finding_id: str, actor: str, action: str, note: str = "") -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO finding_events (finding_id,at,actor,action,note) VALUES (?,?,?,?,?)",
        (finding_id, _now(), actor, action, note),
    )
    if action in ("approved", "rejected", "closed"):
        conn.execute("UPDATE findings SET status=? WHERE id=?",
                     ({"approved": "approved", "rejected": "rejected", "closed": "closed"}[action], finding_id))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------
class FindingInput(BaseModel):
    title: str = Field(description="One-line statement of the margin leak, with the number in it.")
    dimension: str = Field(description="overstock|stockout|discount_depth|assortment_error|competitor_undercut|ad_efficiency|forecast_error|lead_time|price_position|placement")
    dollar_impact: float = Field(description="Annualised margin dollars at stake. Never guess blind; derive it from a tool result.")
    confidence: float = Field(description="0..1 confidence given the evidence you actually have.")
    evidence: str = Field(description="The numbers and queries that support this. Be specific.")
    entity_type: str = Field(default="sku")
    entity_id: str = Field(default="", description="e.g. SKU-4521 or 'smallappliance'")
    recommendation: str = Field(default="", description="The single action the owner should take.")


@tool("log_finding", args_schema=FindingInput)
def log_finding(
    title: str,
    dimension: str,
    dollar_impact: float,
    confidence: float,
    evidence: str,
    entity_type: str = "sku",
    entity_id: str = "",
    recommendation: str = "",
) -> str:
    """Append a margin finding to the Accountability Ledger with a dollar impact
    and an owning function. Only call this once your numbers survived the
    critique step. The owner is derived from the semantic layer's ownership map,
    so you cannot quietly assign a leak to 'the market'."""
    fid = write_finding(
        title=title, dimension=dimension, dollar_impact=dollar_impact, confidence=confidence,
        evidence=evidence, entity_type=entity_type, entity_id=entity_id, recommendation=recommendation,
    )
    return f"logged {fid} owner={owner_for(dimension)} impact=${dollar_impact:,.0f} confidence={confidence}"


@tool("read_ledger")
def read_ledger(status: str = "open", limit: int = 15) -> str:
    """Read prior findings from the Accountability Ledger so the same leak is not
    rediscovered with a new name every quarter."""
    rows = list_findings(status, limit)
    if not rows:
        return "Ledger is empty. Either the business is perfect or nobody has looked yet."
    return "\n".join(
        f"{r['id']} [{r['dimension']}] {r['entity_id']} ${r['dollar_impact']:,.0f} "
        f"conf={r['confidence']} owner={r['owner_function']} status={r['status']} :: {r['title']}"
        for r in rows
    )
