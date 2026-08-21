# 💬 Karma Edge — Example Prompt Library

15 prompts, ordered from cheap to expensive. Paste them straight into `python -m app.main` (or the Streamlit UI). Each one lists the agents and tools it should exercise and what a good answer looks like — so you can tell "the model is weak" apart from "the code is broken."

**Before you start:**
```bash
python -m data.seed          # once
python -m app.main
you > /provider zhipu        # or deepseek / qwen / moonshot
```

Run **1–3 on `MODEL_PROVIDER=fake` first.** They exercise real SQL and real tools with a scripted model, so if the numbers come back sane, your install is fine and every later problem is a model problem.

> The demo warehouse has **deliberately planted saboteurs**. A correct system finds them *with the right owner attached*. Expected owners are listed below — that's your grading key.

---

## Tier 1 — Warm-up (single specialist, 1–3 tool calls, cheap)

### 1. Baseline margin scan
```
Which category is losing the most gross margin, and who owns it?
```
**Exercises:** `data_analyst` → `semantic_metric(gross_margin, [category])` → `accountability`
**Good answer:** dollar margin by category, the worst one named, one owning function. Should report **dollars, not just rate**.

### 2. Semantic layer introspection
```
What metrics can you calculate, and how is gross margin actually defined here?
```
**Exercises:** `list_metrics`
**Good answer:** the metric list with definitions and grains. If it invents metrics not in `semantic/metrics.yml`, your model is too weak for the rest of this list.

### 3. Policy retrieval, no SQL
```
What does our markdown policy say about discount depth, and at what point does ownership move to the buying team?
```
**Exercises:** `search_policy` (BM25 over `knowledge/markdown_policy.md`)
**Good answer:** quotes the discount cap and the **three-consecutive-months-of-deepening-discount → Buying** escalation rule. Must cite the doc, not paraphrase from thin air.

---

## Tier 2 — Real analysis (multi-tool, one or two specialists)

### 4. Stockout risk
```
Which SKUs are at stockout risk in the next 14 days, and what should we reorder today?
```
**Exercises:** `forecaster` → `inventory_health` + `reorder_plan`
**Good answer:** SKUs with weeks-of-cover under threshold, a reorder point and quantity per SKU, lead times acknowledged. Expected owner on findings: **supply_chain**.

### 5. Trapped capital
```
How much working capital is trapped in overstock right now, and which category is the worst offender?
```
**Exercises:** `inventory_health` (high-cover side) + `sql_query`
**Good answer:** a dollar figure for trapped capital, weeks of cover, small appliances flagged. Expected owner: **buying**.

### 6. Forecast with an error bar
```
Forecast the next 8 weeks of demand for our top-selling footwear SKU and tell me how much I should trust it.
```
**Exercises:** `forecast_series` (auto model selection by holdout MAPE)
**Good answer:** names the **selected model**, reports **holdout MAPE**, gives an 80% interval. A forecast with no error estimate is a failing answer — say so and switch models.

### 7. Price change simulation
```
If we cut price 10% on our worst-performing small appliance SKU, what happens to units, revenue, and margin dollars?
```
**Exercises:** `pricing_strategist` → `simulate_price_change` (log-log OLS elasticity)
**Good answer:** own elasticity with **observation count and R²**, then units up / margin dollars likely down. If it reports elasticity from fewer than 12 observations without caveating, the Critic should have caught it — check `critic_issues`.

### 8. Profit-optimal price
```
What's the profit-maximising price for that SKU given how much inventory we're sitting on?
```
**Exercises:** `optimize_price` via `simulate_price_change`
**Good answer:** a specific price, the expected profit delta, and the inventory-cover constraint that bounded the search.

---

## Tier 3 — The critique loop earns its keep (multi-specialist, expect loops)

### 9. Pricing problem or buying problem? *(the flagship prompt)*
```
Small appliance margin is collapsing and discounts keep getting deeper. Is this a pricing problem or a buying problem? Prove it, and put a name on it.
```
**Exercises:** `data_analyst` → `forecaster` → `pricing_strategist` → Critic (expect at least one **REVISE**) → `accountability`
**Good answer:** the discount trend month over month, cover level and trapped capital, elasticity showing demand *isn't* price-driven — concluding **buying**, per the markdown policy escalation rule. This is the prompt where you'll actually see the loop work; watch `critic_issues` in the transcript.

### 10. Seasonality trap (deliberately baits a confounder)
```
Q4 margin dropped 8% versus Q3. Why?
```
**Exercises:** `data_analyst` → Critic check **E (confounding)** → `forecaster`
**Good answer:** refuses to blame a function until seasonality and mix shift are separated from the underlying trend. If the first answer confidently blames Pricing and the Critic passes it, your model is too weak — try `deepseek-chat`.

### 11. Assortment error
```
Are we stocking any products in regions where they can't possibly sell? Quantify the damage.
```
**Exercises:** `sql_query` (product × region × sell-through) + `search_policy` (climate-fit rule in the assortment playbook)
**Good answer:** finds the snow boots in the warm region, cites the climate-fit rule, dollar-quantifies it. Expected owner: **buying**.

### 12. Cost to serve
```
What are the biggest inefficiencies in our supply chain right now, and how would you optimize cost to serve?
```
**Exercises:** `forecaster` + `data_analyst` + `search_policy` (vendor terms) + `accountability`
**Good answer:** lead-time volatility, expedite/split-shipment cost, cover mismatches — each with a dollar figure and an owner. Vague "improve efficiency" answers should be caught by Critic check **G (actionability)**.

---

## Tier 4 — Autonomous competitor agent (needs network; slowest)

### 13. Competitor discovery from scratch
```
I'm a mid-size US home goods retailer. Find out who my actual online competitors are in small kitchen appliances.
```
**Exercises:** `competitor_intel` → `discover_competitors` (keyless DuckDuckGo, or Tavily if keyed)
**Good answer:** real domains, aggregators and marketplaces filtered out, ranked by evidence. **Nothing is hardcoded** — this is live research. If it returns nothing, it should *say so*, not invent names.

### 14. Scrape and gap-analyse
```
Take the top competitor you just found, find their small appliance catalog, scrape their listed prices, and tell me where we're being undercut.
```
**Exercises:** `discover_catalog_urls` (robots → sitemap → keyword-scored URLs) → `scrape_competitor` (JSON-LD → LLM extraction → regex) → `match_to_internal` (fuzzy, confidence reported) → `price_gap_report`
**Good answer:** scraped product names and prices, fuzzy match confidence per row, our price vs theirs with % gap. **Acceptable answer:** "this site is JS-rendered / blocked us, so I have no prices" — honesty beats fabrication, and the agent is instructed accordingly.

### 15. Full accountability run, with the human gate
```
Give me a board-ready margin accountability report: the three biggest margin leaks, the dollar impact of each, who owns each one, and what they need to do this month.
```
**Exercises:** effectively the whole graph — every specialist, several critique iterations, `log_finding`, and the **HITL interrupt** if any finding lands at or above $250k
**Good answer:** three findings, each with a dollar figure traceable to a tool result, one named owner, one numeric action, and an honest confidence. Then:
```
you > /ledger
```
and see them persisted with their `finding_events` audit trail.

> Prompt 15 is the expensive one — many model calls. Run it last, and on your best available model.

---

## Reading the output

| You see | It means |
| --- | --- |
| `ROUTE: forecaster` | Supervisor picked a specialist. Multiple routes per question is normal. |
| `VERDICT: REVISE` + issues | The Critic broke the answer. **This is the system working**, not failing. |
| `A:` in issues | A dollar figure was hallucinated — no tool produced it. Most important check in the system. |
| `E:` in issues | A cause was claimed without ruling out seasonality or mix. |
| `⚠ shipped with unresolved issues` | 10 iterations hit the cap. Answer ships with caveats printed, by design. |
| `HUMAN APPROVAL REQUIRED` | A finding at or above $250k. Type `approve` or `reject`. |
| `KE-XXXXXXXX` | A ledger finding id. `/ledger` shows it. |

---

## If answers are disappointing

1. **Switch models:** `/provider deepseek`. Free 7B-class models route acceptably but critique shallowly.
2. **Narrow the question.** "Why is margin down?" is a research programme; "Why is small appliance margin down in the West region?" is a query.
3. **Check for a `429`.** Free tiers rate-limit per minute; the loop makes many calls. Wait, or switch providers.
4. **Reduce loops while iterating:** `MAX_CRITIQUE_ITERATIONS=3` in `.env`.
5. **Read the transcript, not just the answer.** The tool calls tell you whether the model failed to reason or a tool failed to return.
