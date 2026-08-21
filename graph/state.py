################################################################################
# Karma Edge - graph/state.py
#
# One typed state object flows through the whole graph. Messages accumulate via
# LangGraph's add_messages reducer; everything else is last-write-wins.
################################################################################
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class KarmaState(TypedDict, total=False):
    # conversation
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    run_id: str

    # routing
    next_agent: str
    route_reason: str
    visited: Annotated[List[str], operator.add]

    # critique loop
    iteration: int
    verdict: str            # PASS | REVISE | ""
    critic_confidence: float
    critic_issues: List[str]
    critic_next: List[str]

    # outputs
    draft: str
    final: str
    findings: Annotated[List[Dict[str, Any]], operator.add]

    # human-in-the-loop
    hitl_required: bool
    hitl_reason: str
    hitl_decision: Optional[str]   # approve | reject | None


def new_state(question: str, run_id: str) -> KarmaState:
    return {
        "messages": [],
        "question": question,
        "run_id": run_id,
        "next_agent": "",
        "visited": [],
        "iteration": 0,
        "verdict": "",
        "critic_confidence": 0.0,
        "critic_issues": [],
        "critic_next": [],
        "draft": "",
        "final": "",
        "findings": [],
        "hitl_required": False,
        "hitl_reason": "",
        "hitl_decision": None,
    }
