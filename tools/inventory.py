################################################################################
# Karma Edge - tools/inventory.py
#
# Supply-chain diagnostics as embedded functions. Stockouts, overstock, and the
# lead-time arithmetic nobody wants their name on.
################################################################################
from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.forecasting import select_and_forecast
from tools.sql_tools import run_sql


class CoverInput(BaseModel):
    sku: Optional[str] = Field(default=None, description="Restrict to one SKU.")
    threshold_low: float = Field(default=2.0, description="Weeks of cover below which we flag stockout risk.")
    threshold_high: float = Field(default=12.0, description="Weeks of cover above which we flag overstock.")


@tool("inventory_health", args_schema=CoverInput)
def inventory_health(sku: str | None = None, threshold_low: float = 2.0, threshold_high: float = 12.0) -> str:
    """Compute weeks-of-cover per SKU/store from latest on-hand against recent
    run rate, and flag stockout risk, overstock, and the dollars trapped in each.
    This is the tool that turns 'inventory feels heavy' into a number."""
    where = f"AND s.sku = '{sku}'" if sku else ""
    res = run_sql(
        f"""
        WITH latest AS (SELECT MAX(date) AS d FROM inventory),
        rate AS (
            SELECT sku, store_id, AVG(units) AS wk_units
            FROM v_sales_margin
            WHERE date >= (SELECT date FROM sales ORDER BY date DESC LIMIT 1 OFFSET 8)
            GROUP BY sku, store_id
        )
        SELECT i.sku, i.store_id, p.title, p.unit_cost, p.lead_time_days,
               i.on_hand_units, COALESCE(r.wk_units, 0) AS wk_units,
               ROUND(i.on_hand_units / NULLIF(r.wk_units, 0), 1) AS weeks_cover,
               ROUND(i.on_hand_units * p.unit_cost, 0) AS capital_tied
        FROM inventory i
        JOIN products p ON p.sku = i.sku
        LEFT JOIN rate r ON r.sku = i.sku AND r.store_id = i.store_id
        JOIN latest l ON i.date = l.d
        JOIN v_sales_margin s ON s.sku = i.sku
        WHERE 1=1 {where}
        GROUP BY i.sku, i.store_id
        ORDER BY capital_tied DESC
        """,
        limit=100,
    )
    if not res["ok"]:
        return f"INVENTORY CHECK FAILED: {res['error']}"

    stockouts, overstock, lines = [], [], []
    for r in res["rows"]:
        wc = r["weeks_cover"]
        tag = ""
        if wc is not None and wc < threshold_low:
            tag = "STOCKOUT_RISK"
            stockouts.append(r)
        elif wc is not None and wc > threshold_high:
            tag = "OVERSTOCK"
            overstock.append(r)
        lines.append(
            f"{r['sku']} {r['store_id']} on_hand={r['on_hand_units']} wk_rate={r['wk_units']:.1f} "
            f"cover={wc} lead_time={r['lead_time_days']}d capital=${r['capital_tied']:,.0f} {tag}"
        )
    trapped = sum(r["capital_tied"] or 0 for r in overstock)
    lost = sum((r["wk_units"] or 0) * 4 * 0.35 * (r["unit_cost"] or 0) for r in stockouts)
    return (
        f"rows={len(res['rows'])} stockout_risk={len(stockouts)} overstock={len(overstock)}\n"
        f"capital_trapped_in_overstock=${trapped:,.0f} est_4wk_lost_margin_from_stockouts=${lost:,.0f}\n"
        + "\n".join(lines[:40])
    )


class ReorderInput(BaseModel):
    sku: str = Field(description="SKU to plan a buy for.")
    service_level: float = Field(default=0.95, description="Target service level, 0.5-0.99.")


@tool("reorder_plan", args_schema=ReorderInput)
def reorder_plan(sku: str, service_level: float = 0.95) -> str:
    """Turn a demand forecast plus the vendor lead time into a reorder point and
    order quantity, so 'we ran out' becomes 'someone ignored this number'."""
    meta = run_sql(f"SELECT lead_time_days, unit_cost, list_price, owner_function FROM products WHERE sku='{sku}'")
    if not meta["ok"] or not meta["rows"]:
        return f"unknown sku {sku}"
    m = meta["rows"][0]
    hist = run_sql(
        f"SELECT date, SUM(units) AS v FROM v_sales_margin WHERE sku='{sku}' GROUP BY date ORDER BY date",
        limit=5000,
    )
    series = [r["v"] for r in hist["rows"]]
    lead_weeks = max(1, round(m["lead_time_days"] / 7))
    fc = select_and_forecast(series, lead_weeks + 4, season=52)
    lead_demand = sum(fc["forecast"][:lead_weeks])
    band = sum(u - l for u, l in zip(fc["upper_80"][:lead_weeks], fc["lower_80"][:lead_weeks])) / 2
    z = {0.9: 1.28, 0.95: 1.65, 0.99: 2.33}.get(round(service_level, 2), 1.65)
    safety = band * (z / 1.28) / 2
    on_hand = run_sql(
        f"SELECT SUM(on_hand_units) AS oh FROM inventory WHERE sku='{sku}' AND date=(SELECT MAX(date) FROM inventory)"
    )["rows"][0]["oh"] or 0
    rop = lead_demand + safety
    qty = max(0, rop + sum(fc["forecast"][lead_weeks:]) - on_hand)
    return (
        f"sku={sku} owner={m['owner_function']} model={fc['model']} mape={fc['backtest_mape']}\n"
        f"lead_time={m['lead_time_days']}d ({lead_weeks}w) lead_time_demand={lead_demand:.0f} "
        f"safety_stock={safety:.0f} reorder_point={rop:.0f}\n"
        f"on_hand={on_hand} -> recommended_order_qty={qty:.0f} "
        f"(committed_capital=${qty * m['unit_cost']:,.0f})"
    )
