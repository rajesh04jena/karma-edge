################################################################################
# Karma Edge - tools/websearch.py
#
# Free web search + fetch, no paid API required.
#   search: DuckDuckGo HTML endpoint (no key) -> optional Tavily if TAVILY_API_KEY
#   fetch : requests + BeautifulSoup, with a text/JSON-LD extractor
#
# This is what makes the competitor agent AUTONOMOUS instead of hardcoded: it
# researches who the competitor actually is, finds their catalog URLs, and reads
# them. Nothing about the retailer is baked into the source.
################################################################################
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List
from urllib.parse import quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from pydantic import BaseModel, Field

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
TIMEOUT = 20
_LAST_CALL: Dict[str, float] = {}


def _polite(host: str, min_gap: float = 1.5) -> None:
    """One request per host per 1.5s. Be the kind of scraper you'd want scraping you."""
    last = _LAST_CALL.get(host, 0.0)
    gap = time.time() - last
    if gap < min_gap:
        time.sleep(min_gap - gap)
    _LAST_CALL[host] = time.time()


def ddg_search(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    _polite("duckduckgo.com")
    r = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": UA},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out: List[Dict[str, str]] = []
    for res in soup.select("div.result")[: max_results * 2]:
        a = res.select_one("a.result__a")
        if not a:
            continue
        snippet = res.select_one(".result__snippet")
        out.append({
            "title": a.get_text(" ", strip=True),
            "url": a.get("href", ""),
            "snippet": snippet.get_text(" ", strip=True) if snippet else "",
        })
        if len(out) >= max_results:
            break
    return out


def tavily_search(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    key = os.getenv("TAVILY_API_KEY")
    r = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "max_results": max_results},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return [
        {"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")[:400]}
        for x in r.json().get("results", [])
    ]


def web_search_raw(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    try:
        if os.getenv("TAVILY_API_KEY"):
            return tavily_search(query, max_results)
    except Exception:
        pass
    try:
        return ddg_search(query, max_results)
    except Exception as exc:
        return [{"title": "search_failed", "url": "", "snippet": f"{type(exc).__name__}: {exc}"}]


def fetch_page(url: str, max_chars: int = 8000) -> Dict[str, Any]:
    """Fetch a URL, return cleaned text, JSON-LD blocks, and same-host links."""
    host = urlparse(url).netloc
    _polite(host)
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}, timeout=TIMEOUT)
        status = r.status_code
        r.raise_for_status()
    except Exception as exc:
        return {"ok": False, "url": url, "error": f"{type(exc).__name__}: {exc}"}

    soup = BeautifulSoup(r.text, "html.parser")
    for bad in soup(["script", "style", "noscript", "svg"]):
        if bad.name == "script" and (bad.get("type") or "").endswith("ld+json"):
            continue
        bad.decompose()

    ld: List[Any] = []
    for node in soup.find_all("script", attrs={"type": re.compile("ld\\+json")}):
        try:
            ld.append(json.loads(node.string or "{}"))
        except Exception:
            continue

    links = []
    for a in soup.find_all("a", href=True)[:400]:
        absolute = urljoin(url, a["href"])
        if urlparse(absolute).netloc == host:
            links.append({"text": a.get_text(" ", strip=True)[:80], "url": absolute})

    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))
    return {
        "ok": True, "url": url, "status": status,
        "title": (soup.title.get_text(strip=True) if soup.title else ""),
        "text": text[:max_chars],
        "jsonld": ld[:12],
        "links": links[:120],
    }


class SearchInput(BaseModel):
    query: str = Field(description="Search query. Be specific: include the retailer name and the market.")
    max_results: int = Field(default=8)


@tool("web_search", args_schema=SearchInput)
def web_search(query: str, max_results: int = 8) -> str:
    """Search the public web (free, keyless DuckDuckGo; Tavily if a key is set).
    Use this to identify who a retailer's real competitors are, find their
    storefront domain, and locate category or sitemap URLs before scraping."""
    results = web_search_raw(query, max_results)
    return "\n".join(f"[{i+1}] {r['title']}\n    {r['url']}\n    {r['snippet'][:240]}" for i, r in enumerate(results))


class FetchInput(BaseModel):
    url: str = Field(description="Absolute URL to fetch.")
    max_chars: int = Field(default=6000)


@tool("fetch_url", args_schema=FetchInput)
def fetch_url(url: str, max_chars: int = 6000) -> str:
    """Fetch a web page and return its visible text, any embedded JSON-LD
    product data, and the same-domain links found on it. Use the links to walk
    from a homepage to a category listing, and the JSON-LD to read prices
    without guessing at CSS selectors."""
    page = fetch_page(url, max_chars)
    if not page["ok"]:
        return f"FETCH FAILED {url}: {page['error']}"
    ld = json.dumps(page["jsonld"])[:2500] if page["jsonld"] else "(none)"
    links = "\n".join(f"  - {l['text']} -> {l['url']}" for l in page["links"][:40])
    return (
        f"url={page['url']} status={page['status']} title={page['title']}\n"
        f"--- JSON-LD ---\n{ld}\n--- LINKS ---\n{links}\n--- TEXT ---\n{page['text']}"
    )


@tool("find_sitemap")
def find_sitemap(domain: str) -> str:
    """Discover a retailer's sitemap or product-feed URLs from robots.txt. This is
    the polite, high-yield way into a catalog: it beats crawling a homepage."""
    domain = domain if domain.startswith("http") else f"https://{domain}"
    out = []
    for path in ("/robots.txt", "/sitemap.xml", "/sitemap_index.xml"):
        page = fetch_page(urljoin(domain, path), max_chars=4000)
        if page.get("ok"):
            found = re.findall(r"https?://[^\s\"<>]+\.xml", page["text"])
            out.append(f"{path}: status={page['status']} sitemaps={sorted(set(found))[:12]}")
        else:
            out.append(f"{path}: {page.get('error')}")
    return "\n".join(out)
