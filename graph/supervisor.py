################################################################################
# Karma Edge - graph/supervisor.py
#
# The control plane.
#
#   supervisor ──▶ specialist ──▶ critic ──┬─ REVISE ─▶ supervisor  (max N loops)
#        ▲                                 └─ PASS ───▶ hitl? ─▶ synthesize ─▶ END
#        └──────────────── FINISH ────────────────────────────────┘
#
# Checkpointed with MemorySaver so the HITL interrupt can actually resume.
################################################################################
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.config import settings
from graph.nodes import (
    AGENTS,
    accountability_node,
    competitor_intel_node,
    critic_node,
    data_analyst_node,
    forecaster_node,
    hitl_node,
    pricing_strategist_node,
    supervisor_node,
    synthesize_node,
)
from graph.state import KarmaState, new_state

NODE_FNS = {
    "data_analyst": data_analyst_node,
    "forecaster": forecaster_node,
    "pricing_strategist": pricing_strategist_node,
    "competitor_intel": competitor_intel_node,
    "accountability": accountability_node,
}


def route_from_supervisor(state: KarmaState) -> str:
    dest = state.get("next_agent", "FINISH")
    if dest == "FINISH":
        # nothing analysed yet? there is nothing to synthesise.
        return "synthesize" if state.get("draft") else "data_analyst"
    return dest


def route_from_critic(state: KarmaState) -> str:
    if state.get("verdict") == "REVISE" and state.get("iteration", 0) < settings.max_critique_iterations:
        return "supervisor"
    if state.get("hitl_required") and settings.hitl_enabled:
        return "hitl"
    return "synthesize"


def build_graph(checkpointer: Optional[Any] = None):
    g = StateGraph(KarmaState)
    g.add_node("supervisor", supervisor_node)
    for name, fn in NODE_FNS.items():
        g.add_node(name, fn)
    g.add_node("critic", critic_node)
    g.add_node("hitl", hitl_node)
    g.add_node("synthesize", synthesize_node)

    g.set_entry_point("supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {**{a: a for a in AGENTS}, "synthesize": "synthesize", "data_analyst": "data_analyst"},
    )
    for name in NODE_FNS:
        g.add_edge(name, "critic")
    g.add_conditional_edges(
        "critic", route_from_critic,
        {"supervisor": "supervisor", "hitl": "hitl", "synthesize": "synthesize"},
    )
    g.add_edge("hitl", "synthesize")
    g.add_edge("synthesize", END)

    return g.compile(
        checkpointer=checkpointer or MemorySaver(),
        interrupt_before=["hitl"] if settings.hitl_enabled else None,
    )


def ask(question: str, thread_id: Optional[str] = None, graph: Optional[Any] = None) -> Dict[str, Any]:
    """One-shot convenience entry point. Returns the final state dict."""
    graph = graph or build_graph()
    run_id = str(uuid.uuid4())[:8]
    cfg = {"configurable": {"thread_id": thread_id or run_id}, "recursion_limit": settings.graph_recursion_limit}
    result = graph.invoke(new_state(question, run_id), cfg)

    # If we stopped at the human gate, auto-approve for non-interactive callers.
    snapshot = graph.get_state(cfg)
    if snapshot.next and "hitl" in snapshot.next:
        graph.update_state(cfg, {"hitl_decision": "approve"})
        result = graph.invoke(None, cfg)
    return result
