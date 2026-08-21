################################################################################
# Karma Edge - offline smoke tests. No API keys, no network.
#   pytest -q
################################################################################
from __future__ import annotations

import os

os.environ.setdefault("MODEL_PROVIDER", "fake")
os.environ.setdefault("HITL_ENABLED", "false")

import pytest  # noqa: E402

from data import seed  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def warehouse():
    seed.build()
    yield


def test_schema_and_metrics():
    from tools.sql_tools import describe_schema, load_semantic_layer

    schema = describe_schema()
    assert "v_sales_margin" in schema
    assert "gross_margin" in schema
    assert load_semantic_layer()["ownership"]["overstock"] == "buying"


def test_readonly_guard():
    from tools.sql_tools import run_sql

    assert run_sql("DELETE FROM sales")["ok"] is False
    assert run_sql("SELECT 1 AS x")["ok"] is True


def test_semantic_metric_tool():
    from tools.sql_tools import semantic_metric

    out = semantic_metric.invoke({"metric": "gross_margin", "dimensions": ["category"]})
    assert "gross_margin" in out and "owner=pricing" in out


def test_forecasting_selects_a_model():
    from tools.forecasting import forecast_series, select_and_forecast

    y = [10 + (i % 12) for i in range(80)]
    out = select_and_forecast(y, 6, season=12)
    assert len(out["forecast"]) == 6
    assert out["backtest_mape"] is not None
    assert "model_selected" in forecast_series.invoke({"metric": "units", "horizon": 4})


def test_elasticity_is_negative_ish():
    from tools.elasticity import optimize_price

    opt = optimize_price(base_price=100, base_demand=50, elasticity=-1.8, unit_cost=40)
    assert opt["price"] > 40
    assert opt["profit"] >= opt["current_profit"]


def test_inventory_health_runs():
    from tools.inventory import inventory_health

    assert "stockout_risk=" in inventory_health.invoke({})


def test_ledger_append_and_owner_map():
    from tools.ledger import list_findings, log_finding

    msg = log_finding.invoke({
        "title": "Overstock on SKU-4521 is eating margin",
        "dimension": "overstock",
        "dollar_impact": 412000.0,
        "confidence": 0.7,
        "evidence": "weeks_cover=19.4 capital=$318k from inventory_health",
        "entity_id": "SKU-4521",
        "recommendation": "Cancel the open PO and claim vendor markdown support.",
    })
    assert "owner=buying" in msg
    assert any(f["entity_id"] == "SKU-4521" for f in list_findings("open", 20))


def test_graph_runs_end_to_end_offline():
    from graph.supervisor import ask

    state = ask("Where is gross margin leaking and who owns it?")
    assert state.get("visited")
    assert state.get("iteration", 0) >= 1
    assert state.get("final")


def test_provider_registry_lists_chinese_apis():
    from app.llm import PROVIDERS, provider_status

    for key in ("zhipu", "deepseek", "qwen", "moonshot", "siliconflow", "openrouter"):
        assert key in PROVIDERS
    assert any(p["provider"] == "fake" and p["ready"] for p in provider_status())
