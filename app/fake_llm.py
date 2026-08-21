################################################################################
# Karma Edge - app/fake_llm.py
#
# A deterministic, offline "analyst" chat model. It implements just enough of
# the LangChain BaseChatModel + bind_tools contract that the whole agent graph
# runs with zero API keys and zero network. The test suite uses it, and so does
# your first `make demo` before you ever sign up for a provider key.
#
# It is not pretending to be smart. It is pretending to be *predictable*.
################################################################################
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# keyword -> tool name. First match wins. Names MUST match the real @tool names.
_ROUTES: List[tuple] = [
    (r"stockout|out of stock|weeks of cover|coverage|overstock|inventory", "inventory_health"),
    (r"reorder|how many should we buy|buy quantity|order quantity", "reorder_plan"),
    (r"forecast|next quarter|project |predict", "forecast_series"),
    (r"elasticit|price change|price cut|markdown|discount|what if we", "simulate_price_change"),
    (r"competitor|rival|undercut|price gap", "price_gap_report"),
    (r"policy|playbook|vendor terms|guideline|handbook", "search_policy"),
    (r"log (a )?finding|accountab|whose fault|who owns", "log_finding"),
    (r"margin|pat|cashflow|p&l|pnl|revenue|profit|sales|category", "semantic_metric"),
]

# Scripted routing plan for the supervisor, so the offline graph still exercises
# more than one specialist before it finishes.
_SUPERVISOR_PLAN = ["data_analyst", "forecaster", "pricing_strategist", "accountability", "FINISH"]


def _first_tool(text: str, available: Sequence[str]) -> Optional[str]:
    low = text.lower()
    for pattern, name in _ROUTES:
        if re.search(pattern, low) and name in available:
            return name
    return available[0] if available else None


def _default_args(tool_name: str, question: str) -> Dict[str, Any]:
    m = re.search(r"SKU-?(\d{3,6})", question, re.I) or re.search(r"#(\d{3,6})", question)
    sku = f"SKU-{m.group(1)}" if m else "SKU-4521"
    return {
        "sql_query": {"sql": "SELECT category, ROUND(SUM(revenue - cogs), 2) AS gross_margin "
                             "FROM v_sales_margin GROUP BY category ORDER BY gross_margin"},
        "semantic_metric": {"metric": "gross_margin", "dimensions": ["category"]},
        "list_metrics": {},
        "forecast_series": {"metric": "revenue", "horizon": 8, "model": "auto"},
        "inventory_health": {},
        "reorder_plan": {"sku": sku},
        "simulate_price_change": {"sku": sku, "own_price_pct": -10.0, "competitor_price_pct": 0.0},
        "price_gap_report": {"limit": 10},
        "search_policy": {"query": question[:120], "k": 3},
        "read_ledger": {"status": "open", "limit": 10},
        "log_finding": {
            "title": "Escalating markdown on small appliances looks like a buy-quantity error",
            "dimension": "overstock", "dollar_impact": 412000.0, "confidence": 0.55,
            "evidence": "scripted offline run; see inventory_health and semantic_metric output",
            "entity_type": "sku", "entity_id": sku,
            "recommendation": "Cancel the open PO and claim vendor markdown support.",
        },
    }.get(tool_name, {})


class ScriptedChatModel(BaseChatModel):
    """Deterministic stand-in for a hosted chat model."""

    model_name: str = "scripted-analyst"
    bound_tools: List[Dict[str, Any]] = []
    forced_json: bool = False

    @property
    def _llm_type(self) -> str:  # pragma: no cover
        return "karma-scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        specs = [{"name": getattr(t, "name", None) or getattr(t, "__name__", str(t))} for t in tools]
        return self.__class__(model_name=self.model_name, bound_tools=specs, forced_json=self.forced_json)

    # -- core ---------------------------------------------------------------
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        available = [t["name"] for t in self.bound_tools]
        system_text = " ".join(str(m.content) for m in messages if m.type == "system")
        human_text = ""
        for m in messages:
            if m.type == "human":
                human_text = str(m.content)

        # --- supervisor role: emit a ROUTE: line -----------------------------
        if "ROUTE:" in system_text:
            visited = re.findall(r"already consulted: \[([^\]]*)\]", human_text)
            done = [v.strip(" '\"") for v in (visited[0].split(",") if visited else []) if v.strip()]
            nxt = next((a for a in _SUPERVISOR_PLAN if a not in done), "FINISH")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=f"ROUTE: {nxt}\nScripted routing plan step; offline deterministic mode."))])

        # --- critic role: emit a VERDICT block -------------------------------
        if "VERDICT:" in system_text:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="VERDICT: PASS\nCONFIDENCE: 0.4\nISSUES:\n"
                        "- D: offline scripted model, evidence depth not audited\n"
                        "NEXT:\n- publish"))])

        # --- ReAct specialist: call one tool, then answer --------------------
        already_called = {
            tc["name"] for m in messages if isinstance(m, AIMessage) for tc in (m.tool_calls or [])
        }
        tool_outputs = [str(m.content) for m in messages if isinstance(m, ToolMessage)]

        pick = _first_tool(human_text, available) if available else None
        if pick and pick not in already_called:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="",
                tool_calls=[{"name": pick, "args": _default_args(pick, human_text),
                             "id": f"call_{uuid.uuid4().hex[:8]}"}],
            ))])

        if self.forced_json:
            body = json.dumps({"answer": "Scripted offline answer.", "confidence": 0.4, "owner": "pricing"})
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=body))])

        evidence = "\n".join(f"- {o[:600]}" for o in tool_outputs)
        if not evidence and not available:
            # synthesis role: no tools bound, so echo the transcript it was given
            evidence = human_text[:4000]
        text = (
            "**Scripted offline answer** (MODEL_PROVIDER=fake - deterministic, not a real model).\n\n"
            f"{evidence or '- no tool evidence gathered'}\n\n"
            "Set MODEL_PROVIDER to zhipu / deepseek / qwen / moonshot / siliconflow / openrouter "
            "for real reasoning over the same tools."
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

