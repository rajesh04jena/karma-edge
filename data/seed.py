################################################################################
# Karma Edge - data/seed.py
# Builds the demo SQLite warehouse. Idempotent: run it as often as you like.
#   python -m data.seed
################################################################################
from __future__ import annotations

import math
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from app.config import settings

HERE = Path(__file__).resolve().parent
random.seed(42)

PRODUCTS = [
    # sku, title, brand, category, unit_cost, list_price, lead_time, owner
    ("SKU-1001", "Trail Runner 5 Mens", "Aeron", "footwear", 28.0, 79.99, 21, "buying"),
    ("SKU-1002", "Trail Runner 5 Womens", "Aeron", "footwear", 28.0, 79.99, 21, "buying"),
    ("SKU-2001", "Alpine Snow Boot", "Northwall", "footwear", 44.0, 139.99, 45, "buying"),
    ("SKU-2002", "Insulated Parka", "Northwall", "outerwear", 61.0, 199.99, 52, "supply_chain"),
    ("SKU-3001", "Cotton Crew Tee 3pk", "Basics Co", "apparel", 6.5, 24.99, 14, "pricing"),
    ("SKU-3002", "Fleece Hoodie", "Basics Co", "apparel", 14.0, 49.99, 18, "pricing"),
    ("SKU-4521", "Smart Air Fryer 5.5L", "Kitchenly", "smallappliance", 52.0, 129.99, 34, "buying"),
    ("SKU-4522", "Stand Mixer Pro", "Kitchenly", "smallappliance", 96.0, 249.99, 40, "supply_chain"),
    ("SKU-5001", "Noise Cancelling Buds", "Soniq", "electronics", 33.0, 99.99, 26, "pricing"),
    ("SKU-5002", "4K Streaming Stick", "Soniq", "electronics", 21.0, 54.99, 19, "ads"),
]

STORES = [
    ("ST-01", "Downtown Flagship", "Northeast", "cold"),
    ("ST-02", "Lakeside Mall", "Midwest", "cold"),
    ("ST-03", "Sunbelt Plaza", "Southeast", "hot"),
    ("ST-04", "Harbor Outlet", "West", "temperate"),
    ("ST-05", "Online DC", "National", "n/a"),
]

WEEKS = 78  # ~18 months, the exact horizon over which "nobody's fault" compounds


def _seasonal(sku: str, week_index: int, climate: str) -> float:
    winter = math.cos((week_index / 52.0) * 2 * math.pi) * 0.5 + 0.5
    if sku in ("SKU-2001", "SKU-2002"):
        base = 0.15 + 1.6 * winter
        if climate == "hot":
            base *= 0.06  # snow boots in the Sunbelt. The boardroom loved it.
        return base
    if sku.startswith("SKU-30"):
        return 1.0 + 0.25 * math.sin((week_index / 52.0) * 2 * math.pi)
    if sku == "SKU-4521":
        return 1.0 + 0.9 * max(0.0, math.sin((week_index / 52.0) * 2 * math.pi + 1.1))
    return 1.0


def build(db_path: Path | None = None) -> Path:
    db_path = Path(db_path or settings.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript((HERE / "schema.sql").read_text())

    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?)", PRODUCTS)
    conn.executemany("INSERT INTO stores VALUES (?,?,?,?)", STORES)

    start = date.today() - timedelta(weeks=WEEKS)
    sales_rows, inv_rows = [], []
    for w in range(WEEKS):
        d = (start + timedelta(weeks=w)).isoformat()
        for sku, _t, _b, cat, cost, price, lead, _own in PRODUCTS:
            for store_id, _n, _r, climate in STORES:
                season = _seasonal(sku, w, climate)
                base = {"footwear": 18, "outerwear": 9, "apparel": 42,
                        "smallappliance": 11, "electronics": 26}[cat]
                if store_id == "ST-05":
                    base = int(base * 2.4)
                units = max(0, int(random.gauss(base * season, base * 0.22)))
                if units == 0:
                    continue

                # The margin story: discounting escalates in the last 26 weeks
                # for smallappliance because nobody owned the buy quantity.
                discount = 0.0
                if cat == "smallappliance" and w > WEEKS - 26:
                    discount = 0.18 + 0.12 * ((w - (WEEKS - 26)) / 26.0)
                elif cat == "apparel":
                    discount = 0.05
                net_price = price * (1 - discount)
                revenue = round(units * net_price, 2)
                cogs = round(units * cost, 2)
                sales_rows.append((d, sku, store_id, units, revenue, cogs, round(discount, 3)))

                on_hand = max(0, int(units * random.uniform(1.2, 4.5)))
                if sku == "SKU-4521" and w > WEEKS - 20:
                    on_hand = int(on_hand * 3.2)  # overstock, quietly
                if sku == "SKU-5001" and w > WEEKS - 6:
                    on_hand = int(on_hand * 0.12)  # stockout risk, loudly
                inv_rows.append((d, sku, store_id, on_hand, random.choice([0, 0, 60, 120])))

    conn.executemany(
        "INSERT INTO sales (date,sku,store_id,units,revenue,cogs,discount) VALUES (?,?,?,?,?,?,?)",
        sales_rows,
    )
    conn.executemany("INSERT OR REPLACE INTO inventory VALUES (?,?,?,?,?)", inv_rows)

    # GL + cashflow, quarterly
    periods = sorted({f"{date.fromisoformat(r[0]).year}-Q{((date.fromisoformat(r[0]).month - 1) // 3) + 1}"
                      for r in sales_rows})
    gl, cf = [], []
    for i, p in enumerate(periods):
        rev = sum(r[4] for r in sales_rows if p == f"{date.fromisoformat(r[0]).year}-Q{((date.fromisoformat(r[0]).month - 1) // 3) + 1}")
        cost = sum(r[5] for r in sales_rows if p == f"{date.fromisoformat(r[0]).year}-Q{((date.fromisoformat(r[0]).month - 1) // 3) + 1}")
        opex = rev * 0.22
        gl += [(p, "revenue", round(rev, 2)), (p, "cogs", round(-cost, 2)),
               (p, "opex", round(-opex, 2)), (p, "pat", round((rev - cost - opex) * 0.74, 2))]
        cf.append((p, round(rev * 0.94, 2), round((cost + opex) * 1.02, 2),
                   round(500000 + (rev * 0.94 - (cost + opex) * 1.02) * (i + 1) / 3, 2)))
    conn.executemany("INSERT INTO gl_entries (period,account,amount) VALUES (?,?,?)", gl)
    conn.executemany("INSERT OR REPLACE INTO cashflow VALUES (?,?,?,?)", cf)

    conn.executemany(
        "INSERT OR REPLACE INTO ad_spend VALUES (?,?,?,?,?)",
        [(p, "SKU-4521", "search", 12000.0, 41000.0) for p in periods[-2:]]
        + [(p, "SKU-5002", "social", 8000.0, 15000.0) for p in periods[-2:]],
    )

    conn.commit()
    counts = {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("products", "stores", "sales", "inventory", "gl_entries", "cashflow", "ad_spend")
    }
    conn.close()
    print(f"seeded {db_path}")
    for t, c in counts.items():
        print(f"  {t:<14} {c:>7}")
    return db_path


if __name__ == "__main__":
    build()
