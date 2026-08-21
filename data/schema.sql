-- Karma Edge demo warehouse (SQLite). Deliberately small, deliberately messy
-- in the places where real retail data is messy, so the critique loop has
-- something real to catch.

DROP VIEW IF EXISTS v_sales_margin;
DROP TABLE IF EXISTS sales;
DROP TABLE IF EXISTS inventory;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS stores;
DROP TABLE IF EXISTS gl_entries;
DROP TABLE IF EXISTS cashflow;
DROP TABLE IF EXISTS competitor_prices;
DROP TABLE IF EXISTS ad_spend;

CREATE TABLE products (
    sku            TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    brand          TEXT NOT NULL,
    category       TEXT NOT NULL,
    unit_cost      REAL NOT NULL,
    list_price     REAL NOT NULL,
    lead_time_days INTEGER NOT NULL,
    owner_function TEXT NOT NULL      -- buying | pricing | supply_chain | ads
);

CREATE TABLE stores (
    store_id TEXT PRIMARY KEY,
    name     TEXT NOT NULL,
    region   TEXT NOT NULL,
    climate  TEXT NOT NULL            -- why snow boots in Florida is a bad idea
);

CREATE TABLE sales (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    date      TEXT NOT NULL,
    sku       TEXT NOT NULL REFERENCES products(sku),
    store_id  TEXT NOT NULL REFERENCES stores(store_id),
    units     INTEGER NOT NULL,
    revenue   REAL NOT NULL,
    cogs      REAL NOT NULL,
    discount  REAL NOT NULL DEFAULT 0
);

CREATE TABLE inventory (
    date          TEXT NOT NULL,
    sku           TEXT NOT NULL REFERENCES products(sku),
    store_id      TEXT NOT NULL REFERENCES stores(store_id),
    on_hand_units INTEGER NOT NULL,
    on_order_units INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (date, sku, store_id)
);

CREATE TABLE gl_entries (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    period  TEXT NOT NULL,
    account TEXT NOT NULL,
    amount  REAL NOT NULL
);

CREATE TABLE cashflow (
    period    TEXT PRIMARY KEY,
    inflow    REAL NOT NULL,
    outflow   REAL NOT NULL,
    closing_balance REAL NOT NULL
);

CREATE TABLE competitor_prices (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    scraped_at   TEXT NOT NULL,
    competitor   TEXT NOT NULL,
    domain       TEXT,
    competitor_sku TEXT,
    title        TEXT,
    brand        TEXT,
    category     TEXT,
    price        REAL,
    was_price    REAL,
    availability TEXT,
    url          TEXT,
    matched_sku  TEXT,
    match_score  REAL
);

CREATE TABLE ad_spend (
    period   TEXT NOT NULL,
    sku      TEXT NOT NULL,
    channel  TEXT NOT NULL,
    spend    REAL NOT NULL,
    attributed_revenue REAL NOT NULL,
    PRIMARY KEY (period, sku, channel)
);

CREATE VIEW v_sales_margin AS
SELECT s.date,
       s.sku,
       p.category,
       p.brand,
       p.owner_function,
       s.store_id,
       st.region,
       s.units,
       s.revenue,
       s.cogs,
       s.discount,
       (s.revenue - s.cogs) AS gross_margin
FROM sales s
JOIN products p ON p.sku = s.sku
JOIN stores  st ON st.store_id = s.store_id;
