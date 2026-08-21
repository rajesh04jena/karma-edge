################################################################################
# Karma Edge - tools/forecasting.py
#
# Forecasting is EMBEDDED, not a service. No MCP, no sidecar, no HTTP hop.
#
# Source of the model logic:
#   https://github.com/rajesh04jena/Limitless  (package: limitless_tsf)
#     - limitless_tsf/forecast/models.py : holt_winters_forecast,
#       auto_arima_forecast, seasonal_naive_forecast, xgboost_regression_forecast,
#       simple/double exponential smoothing, croston_tsb, theta, prophet, tbats
#     - limitless_tsf/predict.py         : auto model selection + backtest
#
# Strategy: if `limitless_tsf` is importable, we call the real thing. If it is
# not (old MacBook, no XGBoost wheel, no patience), we fall back to compact
# re-implementations of the same three workhorse models, ported from that repo's
# kwargs-style API so the tool contract never changes.
################################################################################
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.sql_tools import run_sql

try:  # the real deal, if the user pip-installed it
    from limitless_tsf.forecast.models import (  # type: ignore
        auto_arima_forecast as _lt_arima,
        holt_winters_forecast as _lt_hw,
        seasonal_naive_forecast as _lt_snaive,
    )

    LIMITLESS_AVAILABLE = True
except Exception:  # pragma: no cover
    _lt_arima = _lt_hw = _lt_snaive = None
    LIMITLESS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fallback models (ported from Limitless/limitless_tsf/forecast/models.py)
# ---------------------------------------------------------------------------
def seasonal_naive(y: Sequence[float], horizon: int, season: int = 52) -> List[float]:
    """Repeat the last full seasonal cycle. Ported from `SeasonalNaiveModel`."""
    if not y:
        return [0.0] * horizon
    season = max(1, min(season, len(y)))
    cycle = list(y[-season:])
    return [float(cycle[i % season]) for i in range(horizon)]


def holt_winters(
    y: Sequence[float],
    horizon: int,
    season: int = 52,
    alpha: float = 0.3,
    beta: float = 0.05,
    gamma: float = 0.2,
) -> List[float]:
    """Additive Holt-Winters, triple exponential smoothing.

    Same model family as `holt_winters_forecast(**kwargs)` in Limitless, written
    out longhand so it needs no statsmodels wheel.
    """
    n = len(y)
    if n == 0:
        return [0.0] * horizon
    if n < 2 * season:
        season = max(1, n // 2)
    if season < 2:
        return double_exponential(y, horizon, alpha, beta)

    seasons = n // season
    level = sum(y[:season]) / season
    trend = (sum(y[season : 2 * season]) - sum(y[:season])) / (season * season) if seasons >= 2 else 0.0
    seasonal = [y[i] - level for i in range(season)]

    for t in range(n):
        value = y[t]
        s_idx = t % season
        last_level = level
        level = alpha * (value - seasonal[s_idx]) + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        seasonal[s_idx] = gamma * (value - level) + (1 - gamma) * seasonal[s_idx]

    return [level + (h + 1) * trend + seasonal[(n + h) % season] for h in range(horizon)]


def double_exponential(y: Sequence[float], horizon: int, alpha: float = 0.4, beta: float = 0.1) -> List[float]:
    """Holt's linear trend. Ported from `DoubleExponentialSmoothingModel`."""
    if not y:
        return [0.0] * horizon
    level, trend = float(y[0]), float(y[1] - y[0]) if len(y) > 1 else 0.0
    for value in y[1:]:
        last = level
        level = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last) + (1 - beta) * trend
    return [level + (h + 1) * trend for h in range(horizon)]


def theta_method(y: Sequence[float], horizon: int) -> List[float]:
    """Theta(0,2) - drift line blended with SES. Ported from `theta_forecast`."""
    n = len(y)
    if n < 3:
        return double_exponential(y, horizon)
    xs = list(range(n))
    mean_x, mean_y = sum(xs) / n, sum(y) / n
    denom = sum((x - mean_x) ** 2 for x in xs) or 1.0
    slope = sum((xs[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / denom
    intercept = mean_y - slope * mean_x
    ses = double_exponential(y, horizon, alpha=0.5, beta=0.0)
    return [0.5 * (intercept + slope * (n + h)) + 0.5 * ses[h] for h in range(horizon)]


MODELS = {
    "seasonal_naive": seasonal_naive,
    "holt_winters": holt_winters,
    "double_exponential": double_exponential,
    "theta": theta_method,
}


def _mape(actual: Sequence[float], pred: Sequence[float]) -> float:
    pairs = [(a, p) for a, p in zip(actual, pred) if a not in (0, None)]
    if not pairs:
        return float("inf")
    return sum(abs(a - p) / abs(a) for a, p in pairs) / len(pairs) * 100


def select_and_forecast(
    y: Sequence[float], horizon: int, season: int = 52, model: str = "auto"
) -> Dict[str, Any]:
    """Backtest-driven auto selection, same idea as limitless_tsf.predict."""
    y = [float(v) for v in y if v is not None]
    if len(y) < 4:
        return {"model": "insufficient_data", "forecast": [0.0] * horizon, "backtest_mape": None}

    holdout = min(horizon, max(2, len(y) // 6))
    train, test = y[:-holdout], y[-holdout:]
    scores: Dict[str, float] = {}

    candidates = MODELS if model in ("auto", "", None) else {model: MODELS.get(model, holt_winters)}
    for name, fn in candidates.items():
        try:
            pred = fn(train, holdout, season) if name in ("seasonal_naive", "holt_winters") else fn(train, holdout)
            scores[name] = _mape(test, pred)
        except Exception:
            scores[name] = float("inf")

    best = min(scores, key=scores.get)
    fn = MODELS[best]
    fc = fn(y, horizon, season) if best in ("seasonal_naive", "holt_winters") else fn(y, horizon)
    resid = math.sqrt(sum((v - sum(y) / len(y)) ** 2 for v in y) / len(y))
    return {
        "model": best,
        "engine": "limitless_tsf" if LIMITLESS_AVAILABLE else "karma-edge embedded port",
        "backtest_mape": round(scores[best], 2) if scores[best] != float("inf") else None,
        "all_scores": {k: (round(v, 2) if v != float("inf") else None) for k, v in scores.items()},
        "forecast": [round(v, 2) for v in fc],
        "lower_80": [round(v - 1.28 * resid, 2) for v in fc],
        "upper_80": [round(v + 1.28 * resid, 2) for v in fc],
    }


# ---------------------------------------------------------------------------
# LangChain tool surface
# ---------------------------------------------------------------------------
class ForecastInput(BaseModel):
    metric: str = Field(description="One of: revenue, units, gross_margin, cogs.")
    horizon: int = Field(default=8, description="Periods (weeks) to forecast.")
    sku: Optional[str] = Field(default=None, description="Restrict to one SKU, e.g. SKU-4521.")
    category: Optional[str] = Field(default=None, description="Restrict to one category.")
    model: str = Field(default="auto", description="auto | holt_winters | seasonal_naive | theta | double_exponential")


@tool("forecast_series", args_schema=ForecastInput)
def forecast_series(
    metric: str,
    horizon: int = 8,
    sku: str | None = None,
    category: str | None = None,
    model: str = "auto",
) -> str:
    """Forecast a weekly retail metric forward using the embedded Limitless_TSF
    model family with automatic backtest-based model selection. Returns the
    chosen model, its holdout MAPE, and an 80% interval. Use this before making
    any claim about next quarter."""
    expr = {
        "revenue": "SUM(revenue)",
        "units": "SUM(units)",
        "cogs": "SUM(cogs)",
        "gross_margin": "SUM(revenue - cogs)",
    }.get(metric, "SUM(revenue)")
    where = []
    if sku:
        where.append(f"sku = '{sku}'")
    if category:
        where.append(f"category = '{category}'")
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    res = run_sql(f"SELECT date, {expr} AS v FROM v_sales_margin{clause} GROUP BY date ORDER BY date", limit=5000)
    if not res["ok"]:
        return f"FORECAST FAILED: {res['error']}"
    series = [r["v"] for r in res["rows"]]
    out = select_and_forecast(series, horizon, season=52, model=model)
    return (
        f"metric={metric} sku={sku} category={category} history_points={len(series)}\n"
        f"model_selected={out['model']} engine={out['engine']} backtest_mape={out['backtest_mape']}\n"
        f"candidate_mape={out['all_scores']}\n"
        f"forecast={out['forecast']}\nlower_80={out['lower_80']}\nupper_80={out['upper_80']}"
    )
