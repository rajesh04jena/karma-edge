# 🛠️ Karma Edge — Setup Guide for MacBook Users

Written for an **older MacBook** (Intel or Apple Silicon, 8GB RAM, macOS 11+). Nothing here compiles native code, downloads model weights, or runs a local LLM. The heaviest computation on your machine is a Holt-Winters fit in pure Python.

Total time: **about 5 minutes offline**, plus 2 minutes per API key you want.

---

## 0. What you need

| Requirement | Check with | If missing |
| --- | --- | --- |
| Python 3.10–3.12 | `python3 --version` | See step 1 |
| `pip` | `python3 -m pip --version` | Comes with Python |
| ~200 MB free disk | — | Deps are small; no model weights |
| Internet | — | Only for `pip install`, model APIs, and the competitor agent |

You do **not** need: Docker, conda, Homebrew (optional), Ollama, a GPU, Xcode, or a paid account anywhere.

---

## 1. Get a working Python 3

macOS ships an old `python3`. Check first:

```bash
python3 --version
```

If it prints **3.10, 3.11, or 3.12** — you're done, skip to step 2.

If it's 3.9 or older, or missing:

**Option A — official installer (easiest, no Homebrew):**
Download Python 3.12 for macOS from <https://www.python.org/downloads/macos/>, run the `.pkg`, then reopen Terminal and re-check `python3 --version`.

**Option B — Homebrew:**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12
```

> ⚠️ **Avoid Python 3.13** for now — a few LangChain dependencies still ship no wheels for it, which forces your Mac to compile, which is exactly what we're avoiding.

---

## 2. Get the code and create a virtual environment

```bash
# if you downloaded the zip, just unzip and cd in instead
git clone https://github.com/rajesh04jena/karma-edge.git
cd karma-edge

python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. **Every command below assumes that.** If you open a new Terminal tab, re-run `source .venv/bin/activate` from the project folder.

---

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs LangGraph, LangChain, `langchain-openai` (which talks to every Chinese provider), BeautifulSoup, and `rank-bm25`. All pure Python or prebuilt wheels — no compilation, ~90 seconds on old hardware.

Optional extras, only if you want them:

```bash
pip install streamlit        # browser chat UI
pip install pytest           # to run the test suite
pip install statsmodels      # unlocks real ARIMA inside the forecasting tool
pip install chromadb         # local vector store fallback (RETRIEVAL_MODE=chroma)
```

---

## 4. Configure `.env`

```bash
cp .env.example .env
```

Open `.env` in any editor (`open -a TextEdit .env`). **It already works as-is** — `MODEL_PROVIDER=fake` runs the entire agent graph with a deterministic scripted model, no keys, no network. Prove the plumbing works before you spend a token.

---

## 5. Build the demo warehouse

```bash
python -m data.seed
```

This writes `karma_edge.db` (SQLite) with 18 months of synthetic retail data across 5 categories, 4 regions, and ~40 SKUs — with **deliberately planted margin saboteurs**: a small-appliance over-buy spiralling into deeper markdowns each month, a chronic footwear stockout, snow boots shipped to a warm region, and a competitor undercutting two hero SKUs.

Confirm it worked:

```bash
python -m app.main metrics     # lists the semantic metric layer
```

---

## 6. Run it offline first

```bash
python -m pytest -q                                    # expect: 9 passed
python -m app.main ask "Which category is losing the most gross margin and who owns it?"
```

You'll get a `**Scripted offline answer**` banner plus **real numbers from real SQL** — the tools genuinely execute in `fake` mode; only the reasoning is scripted. If this works, your install is correct and every remaining problem is an API key problem.

---

## 7. Get a free API key (pick one; 2 minutes)

Full comparison table is in the [README](README.md#️-free-model-options-all-hosted-all-openai-compatible-no-local-weights). The short version:

### ⭐ Zhipu GLM — best free tier, recommended first key
1. Go to <https://open.bigmodel.cn/> and register (phone number or email).
2. Open the console → **API Keys** → create a key.
3. Paste into `.env`:
   ```env
   MODEL_PROVIDER=zhipu
   ZHIPU_API_KEY=your_key_here
   ```
`glm-4-flash` is free and fast, and handles tool calling well. This is the best default for a 10–15 prompt test session.

### DeepSeek — strongest reasoning per cent
1. <https://platform.deepseek.com/> → sign up → **API keys**.
2. ```env
   MODEL_PROVIDER=deepseek
   DEEPSEEK_API_KEY=sk-...
   ```
Signup credits, then very cheap. Use this for the hard pricing and competitor prompts.

### Alibaba Qwen (DashScope)
1. <https://bailian.console.alibabacloud.com/> → activate Model Studio → **API-KEY**.
2. ```env
   MODEL_PROVIDER=qwen
   DASHSCOPE_API_KEY=sk-...
   ```
Karma Edge points at the **international** endpoint (`dashscope-intl`). If you registered on the mainland China console, change `base_url` for `qwen` in `app/llm.py` to `https://dashscope.aliyuncs.com/compatible-mode/v1`.

### Moonshot / Kimi
<https://platform.moonshot.cn/> → API keys → `MOONSHOT_API_KEY=sk-...`, `MODEL_PROVIDER=moonshot`.

### SiliconFlow — one key, many free Chinese open models
<https://cloud.siliconflow.cn/> → API keys → `SILICONFLOW_API_KEY=sk-...`, `MODEL_PROVIDER=siliconflow`.

### OpenRouter — aggregator with `:free` variants
<https://openrouter.ai/> → Keys → `OPENROUTER_API_KEY=sk-or-...`, `MODEL_PROVIDER=openrouter`.
Free variants rotate and rate-limit hard; fine for a handful of prompts, frustrating for a long session.

> You can put **several** keys in `.env` at once and switch between them live with `/provider deepseek`. That's the point of the provider factory.

---

## 8. Chat

```bash
python -m app.main
```

```
you > /providers                          # shows every provider and whether its key is set
you > /provider zhipu                     # switch model mid-conversation
you > /provider deepseek deepseek-chat    # switch provider AND model
you > Which category is losing the most gross margin, and who owns it?
you > /ledger                             # dump the accountability ledger
you > /reset                              # clear conversation state
you > /help
you > /quit
```

One-shot mode, handy for scripting:

```bash
python -m app.main ask "Which SKUs are at stockout risk in the next 14 days?"
python -m app.main ledger
python -m app.main metrics
```

Browser UI, with a provider dropdown in the sidebar:

```bash
pip install streamlit
streamlit run app/ui.py
```

Now work through **[PROMPTS.md](PROMPTS.md)** — 15 prompts designed to exercise every agent and tool in order of increasing difficulty.

---

## 9. Optional: enable the competitor agent's better search

The competitor agent works **keyless** using DuckDuckGo. For more reliable discovery, get a free Tavily key (1,000 searches/month) at <https://tavily.com/>:

```env
TAVILY_API_KEY=tvly-...
```

---

## 🚑 Troubleshooting — the five errors you're most likely to hit

### `ModuleNotFoundError: No module named 'langgraph'`
The virtualenv isn't active. Run `source .venv/bin/activate` from the project folder and try again. Confirm with `which python` — it must point inside `.venv`.

### `no such table: sales` / empty results
You skipped step 5. Run `python -m data.seed`. To rebuild from scratch: `rm -f karma_edge.db karma_ledger.db && python -m data.seed`.

### `RuntimeError: ZHIPU_API_KEY is not set.`
The key is missing from `.env`, or the file is named `.env.example` (not `.env`), or you edited `.env` but are running from a different folder. Verify:
```bash
python -c "from app.config import settings; print(settings.model_provider)"
```

### `401 Unauthorized` / `Invalid API key`
Three usual causes: a trailing space or quotes around the key in `.env` (write `ZHIPU_API_KEY=abc123`, not `ZHIPU_API_KEY="abc123 "`); the provider requires a billing/verification step before the key activates; or you're using a mainland-only key against the international endpoint (see the Qwen note in step 7).

### `429 Too Many Requests` — very common on free tiers
Free quotas are per-minute as well as per-day. Wait 60 seconds, or `/provider` over to a different key. If it persists, lower the load: set `MAX_CRITIQUE_ITERATIONS=3` in `.env` — the critique loop is the biggest token consumer, since it can trigger up to 10 extra specialist passes.

### The model ignores tools / returns prose instead of numbers
Small free models sometimes skip tool calling. Switch to a stronger one (`glm-4-plus`, `deepseek-chat`, `qwen-plus`) with `/provider deepseek`. `Qwen2.5-7B` on SiliconFlow is the least reliable at multi-tool ReAct loops.

### It stops and prints "HUMAN APPROVAL REQUIRED"
Working as designed. A finding at or above `HITL_DOLLAR_THRESHOLD` ($250k) halts the graph via a real LangGraph interrupt. Type `approve` or `reject`. To disable for testing: `HITL_ENABLED=false` in `.env`.

### Competitor scraping returns nothing
The target site is JS-rendered or is blocking us. The agent is instructed to report this rather than invent prices — that's the honest outcome, not a bug. For JS-heavy retailers you'd add Playwright; see [ARCHITECTURE.md](ARCHITECTURE.md).

### Everything is slow
Free-tier latency plus a critique loop that can make many model calls per question. Reduce `MAX_CRITIQUE_ITERATIONS` to 3, use `glm-4-flash` (fastest free option), and ask narrower questions. Nothing here is CPU-bound on your Mac.

---

## 🔧 Tuning for a low-token test session

```env
MAX_CRITIQUE_ITERATIONS=3       # biggest single lever on token spend
HITL_ENABLED=false              # no interrupts while you iterate
AGENT_RECURSION_LIMIT=20        # fewer tool hops per specialist
MODEL_PROVIDER=zhipu
MODEL_NAME=glm-4-flash
```

Rough cost for the 15 prompts in PROMPTS.md at these settings: **free** on Zhipu's `glm-4-flash`, or a few cents on DeepSeek.
