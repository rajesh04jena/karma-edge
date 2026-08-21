################################################################################
# Karma Edge - tools/sql_tools.py
#
# SQL is a tool, not a religion. Read-only, guarded, semantic-layer aware.
################################################################################
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import ROOT, settings

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|replace|pragma|vacuum)\b", re.I
)
_METRICS_PATH = ROOT / "semantic" / "metrics.yml"


def load_semantic_layer() -> Dict[str, Any]:
    if not _METRICS_PATH.exists():
        return {"metrics": {}, "ownership": {}}
    return yaml.safe_load(_METRICS_PATH.read_text()) or {}


def connect(readonly: bool = True) -> sqlite3.Connection:
    path = Path(settings.db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Seed the demo warehouse first: python -m data.seed"
        )
    conn = sqlite3.connect(f"file:{path}?mode={'ro' if readonly else 'rw'}", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def describe_schema() -> str:
    """Compact schema card injected into every SQL agent's system prompt."""
    conn = connect()
    lines: List[str] = []
    for (name, kind) in conn.execute(
        "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY type DESC, name"
    ):
        cols = [f"{r[1]} {r[2]}" for r in conn.execute(f"PRAGMA table_info({name})")]
        lines.append(f"{kind.upper()} {name}({', '.join(cols)})")
    conn.close()
    sem = load_semantic_layer()
    metric_lines = [f"  {k}: {v['sql']}  -- owner={v.get('owner')}" for k, v in (sem.get("metrics") or {}).items()]
    return "\n".join(lines) + "\n\nAPPROVED METRIC DEFINITIONS (use these exact expressions):\n" + "\n".join(metric_lines)


def run_sql(sql: str, limit: int = 200) -> Dict[str, Any]:
    """Execute one read-only SELECT and return rows + metadata. No exceptions leak."""
    sql = sql.strip().rstrip(";")
    if _FORBIDDEN.search(sql):
        return {"ok": False, "error": "Only read-only SELECT statements are allowed.", "rows": []}
    if ";" in sql:
        return {"ok": False, "error": "One statement at a time, please.", "rows": []}
    if not re.match(r"^\s*(select|with)\b", sql, re.I):
        return {"ok": False, "error": "Query must start with SELECT or WITH.", "rows": []}
    try:
        conn = connect()
        cur = conn.execute(sql)
        rows = [dict(r) for r in cur.fetchmany(limit)]
        cols = [d[0] for d in (cur.description or [])]
        conn.close()
        return {"ok": True, "sql": sql, "columns": cols, "row_count": len(rows), "rows": rows}
    except Exception as exc:  # surfaced to the LLM as feedback, on purpose
        return {"ok": False, "sql": sql, "error": f"{type(exc).__name__}: {exc}", "rows": []}


class SqlInput(BaseModel):
    sql: str = Field(description="A single read-only SQLite SELECT/WITH statement.")
    limit: int = Field(default=200, description="Max rows to return.")


@tool("sql_query", args_schema=SqlInput)
def sql_query(sql: str, limit: int = 200) -> str:
    """Run a read-only SQL query against the retail warehouse (sales, inventory,
    products, stores, gl_entries, cashflow, competitor_prices, ad_spend, and the
    v_sales_margin view). Use the approved metric definitions from the semantic
    layer rather than inventing arithmetic."""
    result = run_sql(sql, limit)
    if not result["ok"]:
        return f"QUERY FAILED: {result['error']}\nFix the SQL and try again."
    head = result["rows"][:25]
    return (
        f"rows={result['row_count']} columns={result['columns']}\n"
        + "\n".join(str(r) for r in head)
        + ("\n... (truncated)" if result["row_count"] > 25 else "")
    )


class MetricInput(BaseModel):
    metric: str = Field(description="Metric name from the semantic layer, e.g. gross_margin.")
    dimensions: List[str] = Field(default_factory=list, description="Group-by columns, e.g. ['category'].")
    where: Optional[str] = Field(default=None, description="Optional SQL WHERE clause without the WHERE keyword.")
    limit: int = Field(default=50)


@tool("semantic_metric", args_schema=MetricInput)
def semantic_metric(metric: str, dimensions: List[str] | None = None, where: str | None = None, limit: int = 50) -> str:
    """Compute an approved metric by name, grouped by optional dimensions. Prefer
    this over hand-written SQL: it guarantees the metric matches the definition
    finance signed off on."""
    sem = load_semantic_layer()
    metrics = sem.get("metrics") or {}
    if metric not in metrics:
        return f"Unknown metric {metric!r}. Available: {', '.join(sorted(metrics))}"
    spec = metrics[metric]
    dims = [d for d in (dimensions or []) if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", d)]
    select = ", ".join(dims + [f"{spec['sql']} AS {metric}"])
    sql = f"SELECT {select} FROM {spec['source']}"
    if where and not _FORBIDDEN.search(where):
        sql += f" WHERE {where}"
    if dims:
        sql += " GROUP BY " + ", ".join(dims)
    sql += f" LIMIT {int(limit)}"
    result = run_sql(sql, limit)
    if not result["ok"]:
        return f"METRIC FAILED: {result['error']} (sql: {sql})"
    return f"metric={metric} owner={spec.get('owner')} sql={sql}\n" + "\n".join(str(r) for r in result["rows"])


@tool("list_metrics")
def list_metrics() -> str:
    """List every approved metric with its owning function and definition."""
    sem = load_semantic_layer()
    return "\n".join(
        f"{k}: {v['sql']} (owner={v.get('owner')}) - {v.get('description','')}"
        for k, v in (sem.get("metrics") or {}).items()
    )
