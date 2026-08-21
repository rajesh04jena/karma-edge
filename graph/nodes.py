################################################################################
# Karma Edge - graph/nodes.py
#
# Every specialist is a LangGraph ReAct sub-agent built from LangChain @tool
# functions. Tools are plain Python in this process. No MCP, no HTTP, no sidecar.
################################################################################
from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agents.prompts import (
    ACCOUNTABILITY_PROMPT,
    ANALYST_PROMPT,
    COMPETITOR_PROMPT,
    CRITIC_PROMPT,
    FORECASTER_PROMPT,
    PRICING_PROMPT,
    SUPERVISOR_PROMPT,
)
from app.config import settings
from app.llm import get_llm
from graph.state import KarmaState
from tools.competitor import (
    discover_catalog_urls_tool,
    discover_competitors_tool,
    price_gap_report,
    scrape_competitor,
)
from tools.elasticity import simulate_price_change
from tools.forecasting import forecast_series
from tools.inventory import inventory_health, reorder_plan
from tools.ledger import list_findings, log_finding, read_ledger
from tools.retrieval import search_policy
from tools.sql_tools import describe_schema, list_metrics, semantic_metric, sql_query

AGENTS = ("data_analyst", "forecaster", "pricing_strategist", "competitor_intel", "accountability")

TOOLSETS: Dict[str, List[Any]] = {
    "data_analyst": [sql_query, semantic_metric, list_metrics, search_policy],
    "forecaster": [forecast_series, inventory_health, reorder_plan, sql_query, semantic_metric],
    "pricing_strategist": [simulate_price_change, price_gap_report, sql_query, semantic_metric, search_policy],
    "competitor_intel": [
        discover_competitors_tool, discover_catalog_urls_tool, scrape_competitor,
        price_gap_report, sql_query,
    ],
    "accountability": [log_finding, read_ledger, search_policy, sql_query],
}


def _prompt_for(name: str) -> str:
    if name == "data_analyst":
        try:
            return ANALYST_PROMPT.format(schema=describe_schema())
        except Exception as exc:
            return ANALYST_PROMPT.format(schema=f"(schema unavailable: {exc}. Run python -m data.seed)")
    return {
        "forecaster": FORECASTER_PROMPT,
        "pricing_strategist": PRICING_PROMPT,
        "competitor_intel": COMPETITOR_PROMPT,
        "accountability": ACCOUNTABILITY_PROMPT,
    }[name]


_AGENT_CACHE: Dict[str, Any] = {}


def build_agent(name: str):
    """ReAct sub-agent: model + its own tools + its own doctrine."""
    key = f"{name}:{settings.model_provider}:{settings.model_name}"
    if key not in _AGENT_CACHE:
        _AGENT_CACHE[key] = create_react_agent(
            get_llm(), TOOLSETS[name], prompt=_prompt_for(name)
        )
    return _AGENT_CACHE[key]


def reset_agent_cache() -> None:
    """Called when the user swaps LLM provider mid-conversation."""
    _AGENT_CACHE.clear()


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------
def supervisor_node(state: KarmaState) -> Dict[str, Any]:
    visited = state.get("visited", [])
    iteration = state.get("iteration", 0)

    context = [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=state["question"])]
    if visited:
        context.append(HumanMessage(content=f"Specialists already consulted: {visited}."))
    if state.get("verdict") == "REVISE":
        context.append(HumanMessage(
            content="The Critic rejected the last draft. Issues:\n"
                    + "\n".join(state.get("critic_issues") or [])
                    + "\nRequired next step:\n" + "\n".join(state.get("critic_next") or [])
        ))
    if state.get("draft"):
        context.append(HumanMessage(content=f"Current draft analysis:\n{state['draft'][:2500]}"))

    try:
        raw = get_llm().invoke(context).content
        text = raw if isinstance(raw, str) else str(raw)
    except Exception as exc:
        return {"next_agent": "FINISH", "route_reason": f"router error: {exc}"}

    match = re.search(r"ROUTE:\s*([A-Za-z_]+)", text)
    dest = (match.group(1) if match else "FINISH").strip()
    if dest not in AGENTS + ("FINISH",):
        dest = "data_analyst" if not visited else "FINISH"
    if iteration == 0 and dest == "FINISH" and not visited:
        dest = "data_analyst"  # never finish before looking at the data once
    return {"next_agent": dest, "route_reason": text[:400]}


# ---------------------------------------------------------------------------
# Specialist wrapper
# ---------------------------------------------------------------------------
def _make_specialist(name: str):
    def node(state: KarmaState) -> Dict[str, Any]:
        task = state["question"]
        if state.get("verdict") == "REVISE":
            task += ("\n\nThe Critic rejected the previous attempt. Fix these:\n"
                     + "\n".join(state.get("critic_issues") or [])
                     + "\nDo this next:\n" + "\n".join(state.get("critic_next") or []))
        if state.get("draft"):
            task += f"\n\nWhat the team has established so far:\n{state['draft'][:3000]}"

        try:
            result = build_agent(name).invoke(
                {"messages": [HumanMessage(content=task)]},
                {"recursion_limit": settings.agent_recursion_limit},
            )
            msgs = result.get("messages", [])
            answer = next(
                (m.content for m in reversed(msgs) if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip()),
                "",
            )
            tool_calls = sum(1 for m in msgs if getattr(m, "tool_calls", None))
        except Exception as exc:
            answer, tool_calls = f"[{name} failed: {type(exc).__name__}: {exc}]", 0

        section = f"### {name} (tool calls: {tool_calls})\n{answer}"
        draft = (state.get("draft") or "")
        new_draft = (draft + "\n\n" + section).strip()

        out: Dict[str, Any] = {
            "draft": new_draft,
            "visited": [name],
            "messages": [AIMessage(content=section, name=name)],
        }
        if name == "accountability":
            out["findings"] = list_findings("open", 10)
            if settings.hitl_enabled:
                big = [f for f in out["findings"] if abs(f.get("dollar_impact", 0)) >= settings.hitl_dollar_threshold]
                if big:
                    out["hitl_required"] = True
                    out["hitl_reason"] = (
                        f"{len(big)} finding(s) at or above ${settings.hitl_dollar_threshold:,.0f} "
                        "require human approval before they leave this room."
                    )
        return out

    node.__name__ = f"{name}_node"
    return node


data_analyst_node = _make_specialist("data_analyst")
forecaster_node = _make_specialist("forecaster")
pricing_strategist_node = _make_specialist("pricing_strategist")
competitor_intel_node = _make_specialist("competitor_intel")
accountability_node = _make_specialist("accountability")


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------
def critic_node(state: KarmaState) -> Dict[str, Any]:
    iteration = state.get("iteration", 0) + 1
    try:
        raw = get_llm().invoke([
            SystemMessage(content=CRITIC_PROMPT),
            HumanMessage(content=f"USER QUESTION:\n{state['question']}\n\nANALYSIS TRANSCRIPT:\n{state.get('draft','')[:9000]}"),
        ]).content
        text = raw if isinstance(raw, str) else str(raw)
    except Exception as exc:
        return {"iteration": iteration, "verdict": "PASS", "critic_confidence": 0.0,
                "critic_issues": [f"critic unavailable: {exc}"], "critic_next": ["publish"]}

    verdict = "REVISE" if re.search(r"VERDICT:\s*REVISE", text, re.I) else "PASS"
    conf_m = re.search(r"CONFIDENCE:\s*([0-9.]+)", text)
    issues = re.findall(r"^\s*-\s*(.+)$", text.split("NEXT:")[0].split("ISSUES:")[-1], re.M) if "ISSUES:" in text else []
    nexts = re.findall(r"^\s*-\s*(.+)$", text.split("NEXT:")[-1], re.M) if "NEXT:" in text else []

    if iteration >= settings.max_critique_iterations:
        verdict = "PASS"  # ship it with the caveats attached rather than loop forever
        issues = issues + [f"iteration cap ({settings.max_critique_iterations}) reached; unresolved issues remain"]

    return {
        "iteration": iteration,
        "verdict": verdict,
        "critic_confidence": float(conf_m.group(1)) if conf_m else 0.5,
        "critic_issues": [i.strip() for i in issues if i.strip().lower() != "none"][:8],
        "critic_next": [n.strip() for n in nexts][:6],
        "messages": [AIMessage(content=f"### critic (iteration {iteration})\n{text}", name="critic")],
    }


# ---------------------------------------------------------------------------
# Human-in-the-loop gate + final synthesis
# ---------------------------------------------------------------------------
def hitl_node(state: KarmaState) -> Dict[str, Any]:
    """Interrupt point. The graph is compiled with interrupt_before=['hitl'] so
    execution actually stops here and waits for a real human decision."""
    decision = state.get("hitl_decision")
    return {"hitl_decision": decision or "approve",
            "messages": [AIMessage(content=f"### human gate\ndecision={decision or 'approve'}", name="hitl")]}


SYNTH_PROMPT = """You are the Karma Edge briefing writer. Produce the final answer.

Structure:
1. **The number** — the single most important quantified finding, in dollars.
2. **What happened** — 3-5 bullets, each with a number.
3. **Who owns it** — the named function, per finding.
4. **Do this** — specific actions with numbers.
5. **Confidence & caveats** — the Critic's unresolved issues, stated plainly.

Rules: no number that is not in the transcript. Be direct. No filler. No apologies.
"""


def synthesize_node(state: KarmaState) -> Dict[str, Any]:
    findings = state.get("findings") or []
    ledger = "\n".join(
        f"- {f['id']} [{f['dimension']}] {f['entity_id']} ${f['dollar_impact']:,.0f} "
        f"owner={f['owner_function']} :: {f['title']}"
        for f in findings
    ) or "(no ledger entries)"
    context = (
        f"QUESTION:\n{state['question']}\n\nTRANSCRIPT:\n{state.get('draft','')[:9000]}\n\n"
        f"LEDGER:\n{ledger}\n\nCRITIC CONFIDENCE: {state.get('critic_confidence')}\n"
        f"UNRESOLVED ISSUES: {state.get('critic_issues')}"
    )
    try:
        raw = get_llm().invoke([SystemMessage(content=SYNTH_PROMPT), HumanMessage(content=context)]).content
        final = raw if isinstance(raw, str) else str(raw)
    except Exception as exc:
        final = f"[synthesis failed: {exc}]\n\n{state.get('draft','')}"
    return {"final": final, "messages": [AIMessage(content=final, name="karma_edge")]}
