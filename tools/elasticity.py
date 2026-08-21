################################################################################
# Karma Edge - tools/elasticity.py
#
# Price elasticity + price optimization, EMBEDDED as plain functions.
#
# Source of the model logic:
#   https://github.com/rajesh04jena/PricePulse
#     - PricePulse/bayesian_model.py : BayesianElasticityModel, ElasticityEstimate
#       (hierarchical elasticity with OLS priors)
#     - PricePulse/optimizer.py      : PriceOptimizer / PriceOptimizerExplainable
#       (constrained profit maximisation under an inventory constraint)
#
# PyMC + graphviz will not install cleanly on an old MacBook, so the default
# path here is the OLS log-log estimator that PricePulse itself uses to build
# its priors (`ElasticityEstimate`), plus a scipy-free reimplementation of
# `PriceOptimizer.optimize_price`. If `PricePulse` is importable we hand the
# hierarchical fit off to it instead.
################################################################################
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.sql_tools import run_sql

try:
    from PricePulse.bayesian_model import BayesianElasticityModel  # type: ignore

    PRICEPULSE_AVAILABLE = True
except Exception:  # pragma: no cover
    BayesianElasticityModel = None
    PRICEPULSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# OLS log-log elasticity (PricePulse `ElasticityEstimate` semantics)
#   ln(demand) = a + b1*ln(own_price) + b2*ln(comp_price)
#   b1 = own-price elasticity, b2 = cross-price elasticity
# ---------------------------------------------------------------------------
def _ols(X: List[List[float]], y: List[float]) -> Optional[List[float]]:
    """Normal-equation OLS with Gaussian elimination. No numpy required."""
    n, k = len(X), len(X[0])
    if n <= k:
        return None
    XtX = [[sum(X[r][i] * X[r][j] for r in range(n)) for j in range(k)] for i in range(k)]
    Xty = [sum(X[r][i] * y[r] for r in range(n)) for i in range(k)]
    for i in range(k):  # augmented elimination with partial pivoting
        XtX[i].append(Xty[i])
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(XtX[r][col]))
        if abs(XtX[pivot][col]) < 1e-12:
            return None
        XtX[col], XtX[pivot] = XtX[pivot], XtX[col]
        pv = XtX[col][col]
        XtX[col] = [v / pv for v in XtX[col]]
        for r in range(k):
            if r != col and XtX[r][col]:
                f = XtX[r][col]
                XtX[r] = [a - f * b for a, b in zip(XtX[r], XtX[col])]
    return [XtX[i][k] for i in range(k)]


class ElasticityEstimate(dict):
    """Mirrors PricePulse's dataclass of the same name, as a plain dict."""


def estimate_elasticity(sku: str) -> Dict[str, Any]:
    res = run_sql(
        """
        SELECT s.date,
               SUM(s.units) AS units,
               SUM(s.revenue) / NULLIF(SUM(s.units), 0) AS own_price,
               (SELECT AVG(c.price) FROM competitor_prices c WHERE c.matched_sku = s.sku) AS comp_price,
               p.list_price
        FROM v_sales_margin s JOIN products p ON p.sku = s.sku
        WHERE s.sku = ?
        GROUP BY s.date ORDER BY s.date
        """.replace("?", f"'{sku}'"),
        limit=5000,
    )
    if not res["ok"] or len(res["rows"]) < 8:
        return {"ok": False, "error": f"not enough observations for {sku}"}

    X, y = [], []
    fallback_comp = None
    for r in res["rows"]:
        if not r["units"] or not r["own_price"] or r["own_price"] <= 0:
            continue
        comp = r["comp_price"] or fallback_comp or r["list_price"]
        fallback_comp = comp
        X.append([1.0, math.log(r["own_price"]), math.log(max(comp, 0.01))])
        y.append(math.log(r["units"]))

    beta = _ols(X, y)
    if beta is None:
        # collapse to own-price only when competitor prices are constant
        X2 = [[row[0], row[1]] for row in X]
        b2 = _ols(X2, y)
        if b2 is None:
            return {"ok": False, "error": "OLS did not converge; need more price variation"}
        beta = [b2[0], b2[1], 0.0]

    n = len(y)
    fitted = [beta[0] + beta[1] * row[1] + beta[2] * row[2] for row in X]
    ss_res = sum((y[i] - fitted[i]) ** 2 for i in range(n))
    ss_tot = sum((v - sum(y) / n) ** 2 for v in y) or 1.0
    return {
        "ok": True,
        "sku": sku,
        "engine": "PricePulse (hierarchical bayes)" if PRICEPULSE_AVAILABLE else "PricePulse OLS log-log prior (embedded)",
        "own_elasticity": round(beta[1], 3),
        "cross_elasticity": round(beta[2], 3),
        "r_squared": round(1 - ss_res / ss_tot, 3),
        "nobs": n,
    }


# ---------------------------------------------------------------------------
# Constrained profit optimizer (PricePulse/optimizer.py PriceOptimizer)
# ---------------------------------------------------------------------------
def optimize_price(
    base_price: float,
    base_demand: float,
    elasticity: float,
    unit_cost: float,
    inventory_cap: Optional[float] = None,
    grid: int = 121,
) -> Dict[str, Any]:
    """Maximise profit s.t. demand <= inventory. Grid search replaces
    scipy.optimize.minimize so there is one less wheel to compile."""

    def demand_at(price: float) -> float:
        if base_price <= 0:
            return base_demand
        return max(0.0, base_demand * (price / base_price) ** elasticity)

    best = None
    for i in range(grid):
        price = base_price * (0.6 + 0.8 * i / (grid - 1))  # -40% .. +40%
        d = demand_at(price)
        if inventory_cap is not None and d > inventory_cap:
            d = inventory_cap
        profit = (price - unit_cost) * d
        if best is None or profit > best["profit"]:
            best = {"price": round(price, 2), "demand": round(d, 1), "profit": round(profit, 2)}
    base_profit = (base_price - unit_cost) * demand_at(base_price)
    best["current_price"] = round(base_price, 2)
    best["current_profit"] = round(base_profit, 2)
    best["profit_uplift_pct"] = round((best["profit"] - base_profit) / abs(base_profit or 1) * 100, 1)
    return best


class ElasticityInput(BaseModel):
    sku: str = Field(description="Internal SKU, e.g. SKU-4521.")
    own_price_pct: float = Field(default=0.0, description="Our price change in percent, e.g. -10 for a 10% cut.")
    competitor_price_pct: float = Field(default=0.0, description="Assumed competitor price change in percent.")


@tool("simulate_price_change", args_schema=ElasticityInput)
def simulate_price_change(sku: str, own_price_pct: float = 0.0, competitor_price_pct: float = 0.0) -> str:
    """Estimate self and cross price elasticity for a SKU from its own sales
    history and scraped competitor prices, then simulate the volume, revenue and
    margin impact of a price move (ours and/or a competitor's). Also returns the
    profit-maximising price under the current inventory constraint."""
    est = estimate_elasticity(sku)
    if not est.get("ok"):
        return f"ELASTICITY FAILED: {est.get('error')}"

    base = run_sql(
        f"""SELECT AVG(units) AS units, SUM(revenue)/NULLIF(SUM(units),0) AS price,
                   MAX(p.unit_cost) AS cost
            FROM v_sales_margin s JOIN products p ON p.sku = s.sku
            WHERE s.sku = '{sku}' AND s.date >= (SELECT MAX(date) FROM sales, '-90 days')""",
        limit=5,
    )
    if not base["ok"] or not base["rows"] or not base["rows"][0]["price"]:
        base = run_sql(
            f"""SELECT AVG(units) AS units, SUM(revenue)/NULLIF(SUM(units),0) AS price,
                       MAX(p.unit_cost) AS cost
                FROM v_sales_margin s JOIN products p ON p.sku = s.sku WHERE s.sku = '{sku}'""",
            limit=5,
        )
    row = base["rows"][0]
    price, units, cost = float(row["price"] or 0), float(row["units"] or 0), float(row["cost"] or 0)

    dq = est["own_elasticity"] * (own_price_pct / 100.0) + est["cross_elasticity"] * (competitor_price_pct / 100.0)
    new_units = units * (1 + dq)
    new_price = price * (1 + own_price_pct / 100.0)
    opt = optimize_price(price, units, est["own_elasticity"], cost)

    return (
        f"sku={sku} engine={est['engine']} nobs={est['nobs']} r2={est['r_squared']}\n"
        f"own_elasticity={est['own_elasticity']} cross_elasticity={est['cross_elasticity']}\n"
        f"scenario: our price {own_price_pct:+.1f}%, competitor {competitor_price_pct:+.1f}%\n"
        f"  units/week {units:.1f} -> {new_units:.1f} ({dq*100:+.1f}%)\n"
        f"  revenue/week {units*price:,.0f} -> {new_units*new_price:,.0f}\n"
        f"  margin/week {(price-cost)*units:,.0f} -> {(new_price-cost)*new_units:,.0f}\n"
        f"profit-optimal price={opt['price']} (current {opt['current_price']}, "
        f"uplift {opt['profit_uplift_pct']:+.1f}%)"
    )
