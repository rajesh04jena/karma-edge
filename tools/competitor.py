################################################################################
# Karma Edge - tools/competitor.py
#
# NOT HARDCODED. There is no competitor domain, no CSS selector table, and no
# retailer name anywhere in this file.
#
# Instead: an LLM-guided extraction pipeline.
#   1. discover_competitors(retailer, market)  -> web research names + domains
#   2. discover_catalog_urls(domain, category) -> sitemap/robots/nav walking
#   3. extract_products(url)                   -> JSON-LD first, LLM parse second
#   4. match_to_internal(products)             -> fuzzy match onto our SKUs
#   5. persist -> competitor_prices table, then pricing agent reads it via SQL
#
# Derived from https://github.com/rajesh04jena/Competitor_Pricing_Assortment,
# generalised: that repo pins selectors per retailer; this rediscovers them.
################################################################################
from __future__ import annotations

import difflib
import json
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import settings
from tools.sql_tools import run_sql
from tools.websearch import fetch_page, web_search_raw

PRICE_RE = re.compile(r"(?:[$£€]|USD\s?)\s?([0-9][0-9,]*\.?[0-9]{0,2})")


# ---------------------------------------------------------------------------
# 1. Who are we even competing with?
# ---------------------------------------------------------------------------
def discover_competitors(retailer: str, market: str = "US", category: str = "") -> List[Dict[str, str]]:
    queries = [
        f"{retailer} competitors {market} {category} retail".strip(),
        f'"{retailer}" vs alternatives online store {category}'.strip(),
        f"{retailer} official online store {market}",
    ]
    hits: List[Dict[str, str]] = []
    for q in queries:
        hits += web_search_raw(q, 8)

    # rank candidate domains by frequency, drop the aggregators
    noise = ("wikipedia", "linkedin", "crunchbase", "reddit", "youtube", "facebook",
             "similarweb", "statista", "glassdoor", "indeed", "bloomberg", "medium")
    scores: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        m = re.match(r"https?://([^/]+)", h.get("url", ""))
        if not m:
            continue
        host = m.group(1).replace("www.", "")
        if any(n in host for n in noise):
            continue
        entry = scores.setdefault(host, {"domain": host, "hits": 0, "titles": []})
        entry["hits"] += 1
        entry["titles"].append(h.get("title", "")[:70])
    return sorted(scores.values(), key=lambda x: -x["hits"])[:8]


# ---------------------------------------------------------------------------
# 2. Where does the catalog live?
# ---------------------------------------------------------------------------
def discover_catalog_urls(domain: str, category: str = "", max_urls: int = 6) -> List[str]:
    base = domain if domain.startswith("http") else f"https://{domain}"
    candidates: List[str] = []

    robots = fetch_page(f"{base}/robots.txt", max_chars=6000)
    sitemaps = re.findall(r"https?://[^\s\"<>]+\.xml", robots.get("text", "")) if robots.get("ok") else []
    for sm in sitemaps[:3]:
        page = fetch_page(sm, max_chars=20000)
        if not page.get("ok"):
            continue
        locs = re.findall(r"<loc>\s*([^<]+)\s*</loc>", page["text"])
        if not locs:
            locs = re.findall(r"https?://[^\s\"<>]+", page["text"])
        kw = [k for k in re.split(r"\W+", category.lower()) if len(k) > 3]
        scored = [u for u in locs if any(k in u.lower() for k in kw)] or locs
        candidates += scored[:max_urls]

    if len(candidates) < max_urls:
        home = fetch_page(base, max_chars=4000)
        if home.get("ok"):
            kw = [k for k in re.split(r"\W+", category.lower()) if len(k) > 3] + [
                "shop", "collections", "category", "catalog", "products", "c/"
            ]
            for link in home["links"]:
                blob = (link["url"] + " " + link["text"]).lower()
                if any(k in blob for k in kw):
                    candidates.append(link["url"])

    seen, out = set(), []
    for u in candidates:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:max_urls]


# ---------------------------------------------------------------------------
# 3. Extraction: structured data first, LLM second, regex last
# ---------------------------------------------------------------------------
def _walk_jsonld(node: Any, sink: List[Dict[str, Any]]) -> None:
    if isinstance(node, list):
        for n in node:
            _walk_jsonld(n, sink)
        return
    if not isinstance(node, dict):
        return
    types = node.get("@type")
    types = [types] if isinstance(types, str) else (types or [])
    if any(str(t).lower() == "product" for t in types):
        offers = node.get("offers") or {}
        if isinstance(offers, list):
            offers = offers[0] if offers else {}
        price = offers.get("price") or offers.get("lowPrice")
        sink.append({
            "title": str(node.get("name", ""))[:160],
            "price": float(price) if price not in (None, "") and str(price).replace(".", "").isdigit() else None,
            "currency": offers.get("priceCurrency", "USD"),
            "availability": str(offers.get("availability", ""))[-24:],
            "brand": (node.get("brand") or {}).get("name") if isinstance(node.get("brand"), dict) else node.get("brand"),
            "source": "jsonld",
        })
    for v in node.values():
        if isinstance(v, (dict, list)):
            _walk_jsonld(v, sink)


def _llm_extract(page_text: str, url: str) -> List[Dict[str, Any]]:
    """Ask the model to read a listing page it has never seen a selector for."""
    from app.llm import get_llm

    llm = get_llm()
    system = (
        "You extract retail product listings from raw page text. Return ONLY a JSON array. "
        'Each element: {"title": str, "price": number, "currency": str, "availability": str, "brand": str}. '
        "Skip navigation, banners, reviews and shipping copy. If no products are present, return []."
    )
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=f"URL: {url}\n\n{page_text[:6000]}")])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        match = re.search(r"\[.*\]", raw, re.S)
        items = json.loads(match.group(0)) if match else []
        for i in items:
            i["source"] = "llm"
        return [i for i in items if isinstance(i, dict) and i.get("title")][:60]
    except Exception:
        return []


def _regex_extract(page_text: str) -> List[Dict[str, Any]]:
    out = []
    for line in page_text.split("\n"):
        m = PRICE_RE.search(line)
        if m and 3 < len(line) < 160:
            try:
                out.append({"title": PRICE_RE.sub("", line).strip()[:160],
                            "price": float(m.group(1).replace(",", "")),
                            "currency": "USD", "availability": "", "brand": "", "source": "regex"})
            except ValueError:
                continue
    return out[:60]


def extract_products(url: str, use_llm: bool = True) -> List[Dict[str, Any]]:
    page = fetch_page(url, max_chars=14000)
    if not page.get("ok"):
        return []
    products: List[Dict[str, Any]] = []
    _walk_jsonld(page.get("jsonld") or [], products)
    products = [p for p in products if p.get("price")]
    if len(products) < 3 and use_llm:
        products += _llm_extract(page["text"], url)
    if len(products) < 3:
        products += _regex_extract(page["text"])
    for p in products:
        p["url"] = url
    # dedupe on title
    seen, out = set(), []
    for p in products:
        key = (p.get("title") or "").lower()[:60]
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# 4. Match onto our own SKUs
# ---------------------------------------------------------------------------
def match_to_internal(products: List[Dict[str, Any]], cutoff: float = 0.52) -> List[Dict[str, Any]]:
    internal = run_sql("SELECT sku, title, brand, list_price FROM products", limit=5000)
    if not internal["ok"]:
        return products
    catalog = internal["rows"]
    titles = [f"{r['brand']} {r['title']}".lower() for r in catalog]
    for p in products:
        candidate = f"{p.get('brand') or ''} {p.get('title') or ''}".strip().lower()
        best = difflib.get_close_matches(candidate, titles, n=1, cutoff=cutoff)
        if best:
            row = catalog[titles.index(best[0])]
            p["matched_sku"] = row["sku"]
            p["match_confidence"] = round(difflib.SequenceMatcher(None, candidate, best[0]).ratio(), 3)
            p["our_price"] = row["list_price"]
        else:
            p["matched_sku"] = None
            p["match_confidence"] = 0.0
    return products


def persist(competitor: str, products: List[Dict[str, Any]]) -> int:
    path = Path(settings.db_path)
    conn = sqlite3.connect(path)
    rows = [
        (competitor, p.get("url", ""), (p.get("title") or "")[:200], p.get("brand") or "",
         float(p["price"]), p.get("currency", "USD"), p.get("availability", ""),
         p.get("matched_sku"), float(p.get("match_confidence") or 0), date.today().isoformat())
        for p in products if p.get("price")
    ]
    conn.executemany(
        "INSERT INTO competitor_prices (competitor,url,title,brand,price,currency,availability,"
        "matched_sku,match_confidence,scraped_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return len(rows)


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------
class DiscoverInput(BaseModel):
    retailer: str = Field(description="Our retailer, or the competitor we want to research.")
    market: str = Field(default="US")
    category: str = Field(default="", description="Optional category focus, e.g. 'small kitchen appliances'.")


@tool("discover_competitors", args_schema=DiscoverInput)
def discover_competitors_tool(retailer: str, market: str = "US", category: str = "") -> str:
    """Research who a retailer's real competitors are on the open web and return
    their ranked storefront domains. Nothing is hardcoded: call this first, then
    scrape the domain it gives you."""
    found = discover_competitors(retailer, market, category)
    if not found:
        return "No candidate competitor domains found. Try a more specific retailer name or market."
    return "\n".join(f"{i+1}. {c['domain']} (mentions={c['hits']}) e.g. {c['titles'][0]}" for i, c in enumerate(found))


class CatalogInput(BaseModel):
    domain: str = Field(description="Competitor domain, e.g. examplemart.com")
    category: str = Field(default="", description="Category to focus the crawl on.")
    max_urls: int = Field(default=5)


@tool("discover_catalog_urls", args_schema=CatalogInput)
def discover_catalog_urls_tool(domain: str, category: str = "", max_urls: int = 5) -> str:
    """Find a competitor's category/listing URLs via robots.txt, sitemaps, and
    homepage navigation. Use before scrape_competitor so you scrape a real
    listing page instead of a homepage carousel."""
    urls = discover_catalog_urls(domain, category, max_urls)
    return "\n".join(urls) if urls else f"No catalog URLs discovered for {domain}."


class ScrapeInput(BaseModel):
    competitor: str = Field(description="Competitor name to record in the ledger, e.g. 'ExampleMart'.")
    urls: List[str] = Field(description="Listing/category URLs to extract products from.")
    persist_rows: bool = Field(default=True, description="Write results into competitor_prices.")


@tool("scrape_competitor", args_schema=ScrapeInput)
def scrape_competitor(competitor: str, urls: List[str], persist_rows: bool = True) -> str:
    """Extract every product and price from the given competitor listing URLs
    using JSON-LD structured data first and LLM-guided parsing as a fallback,
    fuzzy-match them to our internal SKUs, and store them for price-gap analysis.
    Works on retailers this code has never seen."""
    all_products: List[Dict[str, Any]] = []
    per_url = []
    for u in urls[:8]:
        items = extract_products(u)
        per_url.append(f"  {u} -> {len(items)} products")
        all_products += items
    if not all_products:
        return "Extracted 0 products.\n" + "\n".join(per_url) + "\nTry discover_catalog_urls for better URLs."

    all_products = match_to_internal(all_products)
    written = persist(competitor, all_products) if persist_rows else 0
    matched = [p for p in all_products if p.get("matched_sku")]
    gaps = []
    for p in matched[:15]:
        ours, theirs = p.get("our_price") or 0, p["price"]
        if ours:
            gaps.append(f"  {p['matched_sku']} ours=${ours:.2f} theirs=${theirs:.2f} "
                        f"gap={(theirs-ours)/ours*100:+.1f}% conf={p['match_confidence']} :: {p['title'][:50]}")
    return (
        f"competitor={competitor} extracted={len(all_products)} matched_to_our_skus={len(matched)} "
        f"persisted={written}\n" + "\n".join(per_url)
        + ("\n--- PRICE GAPS ---\n" + "\n".join(gaps) if gaps else "\n(no confident SKU matches)")
    )


@tool("price_gap_report")
def price_gap_report(limit: int = 20) -> str:
    """Summarise where we are undercut, at parity, or leaving money on the table,
    using the competitor prices already scraped into the warehouse."""
    res = run_sql(
        """
        SELECT c.matched_sku AS sku, p.title, p.list_price AS ours,
               ROUND(AVG(c.price), 2) AS theirs, COUNT(*) AS obs,
               ROUND((AVG(c.price) - p.list_price) / p.list_price * 100, 1) AS gap_pct
        FROM competitor_prices c JOIN products p ON p.sku = c.matched_sku
        WHERE c.matched_sku IS NOT NULL
        GROUP BY c.matched_sku ORDER BY ABS(gap_pct) DESC
        """,
        limit=limit,
    )
    if not res["ok"]:
        return f"REPORT FAILED: {res['error']}"
    if not res["rows"]:
        return "No matched competitor prices yet. Run discover_competitors -> discover_catalog_urls -> scrape_competitor."
    return "\n".join(
        f"{r['sku']} ours=${r['ours']:.2f} theirs=${r['theirs']:.2f} gap={r['gap_pct']:+.1f}% "
        f"obs={r['obs']} :: {r['title']}"
        for r in res["rows"]
    )
