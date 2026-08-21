################################################################################
# Karma Edge - app/config.py
# Central configuration. Everything is env-driven so nothing is hardcoded.
################################################################################
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional, but recommended
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


@dataclass
class Settings:
    # ---- models -------------------------------------------------------------
    model_provider: str = field(default_factory=lambda: _env("MODEL_PROVIDER", "fake"))
    model_name: str = field(default_factory=lambda: _env("MODEL_NAME", ""))
    temperature: float = 0.1
    request_timeout: int = field(default_factory=lambda: _int("LLM_TIMEOUT", 120))
    max_retries: int = field(default_factory=lambda: _int("LLM_MAX_RETRIES", 2))

    # ---- data ---------------------------------------------------------------
    db_path: Path = field(default_factory=lambda: Path(_env("KARMA_DB", str(DATA_DIR / "karma_edge.db"))))
    ledger_path: Path = field(default_factory=lambda: Path(_env("KARMA_LEDGER", str(DATA_DIR / "ledger.db"))))

    # ---- retrieval ----------------------------------------------------------
    retrieval_mode: str = field(default_factory=lambda: _env("RETRIEVAL_MODE", "bm25"))  # bm25|chroma|auto
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "local"))  # local|zhipu
    chroma_dir: Path = field(default_factory=lambda: Path(_env("CHROMA_DIR", str(CACHE_DIR / "chroma"))))

    # ---- critique loop / graph ----------------------------------------------
    max_critique_iterations: int = field(default_factory=lambda: _int("MAX_CRITIQUE_ITERATIONS", 10))
    hitl_enabled: bool = field(default_factory=lambda: _env("HITL_ENABLED", "true").lower() == "true")
    hitl_dollar_threshold: float = field(default_factory=lambda: float(_int("HITL_DOLLAR_THRESHOLD", 250000)))
    agent_recursion_limit: int = field(default_factory=lambda: _int("AGENT_RECURSION_LIMIT", 30))
    graph_recursion_limit: int = field(default_factory=lambda: _int("GRAPH_RECURSION_LIMIT", 60))


    # ---- scraping -----------------------------------------------------------
    scrape_offline: bool = field(default_factory=lambda: _env("SCRAPE_OFFLINE", "true").lower() == "true")
    scrape_delay_seconds: float = 1.5
    scrape_max_pages: int = field(default_factory=lambda: _int("SCRAPE_MAX_PAGES", 5))
    scrape_user_agent: str = field(
        default_factory=lambda: _env(
            "SCRAPE_USER_AGENT",
            "KarmaEdgeBot/0.1 (+https://github.com/rajesh04jena/karma-edge) research use",
        )
    )
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY"))

    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))


settings = Settings()
