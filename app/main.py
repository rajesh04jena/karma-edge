################################################################################
# Karma Edge - app/main.py
#
# Terminal chatbot. Runtime provider switching included, because your free tier
# will rate-limit you halfway through a demo and you should be able to hop.
#
#   python -m app.main                    # interactive chat
#   python -m app.main providers          # list providers + key readiness
#   python -m app.main ask "question"     # one-shot
#   python -m app.main ledger             # dump the accountability ledger
################################################################################
from __future__ import annotations

import json
import sys
import uuid
from typing import Any, Dict

from app.config import settings
from app.llm import PROVIDERS, provider_status, set_provider

BANNER = r"""
 _  __                          _____    _
| |/ /__ _ _ _ _ __  __ _   ___| ____|__| | __ _  ___
| ' </ _` | '_| '  \/ _` | / -_) _| / _` |/ _` |/ -_)
|_|\_\__,_|_| |_|_|_\__,_| \___|___|\__,_|\__, |\___|
                                          |___/
 margin accountability, with names attached to the numbers
"""

HELP = """
commands:
  /providers                 list model providers and whether their key is set
  /provider <name> [model]   switch provider at runtime (e.g. /provider deepseek)
  /ledger                    show open findings
  /reset                     start a new conversation thread
  /help                      this
  /quit                      leave
anything else is treated as a question for the agent team.
"""


def _print_providers() -> None:
    for p in provider_status():
        flag = "READY" if p["ready"] else "no key"
        print(f"  [{flag:>5}] {p['provider']:<12} {p['label']}")
        print(f"           default={p['default_model']}  env={p['api_key_env']}")
        print(f"           models={', '.join(p['models'])}")
        print(f"           {p['notes']}")
        if p["signup"]:
            print(f"           signup: {p['signup']}")


def _run(graph, question: str, thread_id: str) -> Dict[str, Any]:
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.graph_recursion_limit}
    from graph.state import new_state

    state = graph.invoke(new_state(question, thread_id), cfg)

    snap = graph.get_state(cfg)
    if snap.next and "hitl" in snap.next:
        reason = snap.values.get("hitl_reason", "A high-impact finding needs sign-off.")
        print(f"\n*** HUMAN GATE ***\n{reason}")
        for f in snap.values.get("findings", [])[:5]:
            print(f"  {f['id']} ${f['dollar_impact']:,.0f} owner={f['owner_function']} :: {f['title']}")
        ans = input("approve / reject > ").strip().lower()
        graph.update_state(cfg, {"hitl_decision": "reject" if ans.startswith("r") else "approve"})
        state = graph.invoke(None, cfg)
    return state


def chat() -> None:
    print(BANNER)
    print(f"provider={settings.model_provider} model={settings.model_name or '(default)'}")
    if settings.model_provider == "fake":
        print("running on the offline scripted model. `/provider deepseek` (etc.) for a real one.")
    print(HELP)

    from graph.supervisor import build_graph

    graph = build_graph()
    thread_id = str(uuid.uuid4())[:8]

    while True:
        try:
            line = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return
        if not line:
            continue
        if line in ("/quit", "/exit"):
            print("bye.")
            return
        if line == "/help":
            print(HELP)
            continue
        if line == "/providers":
            _print_providers()
            continue
        if line == "/reset":
            thread_id = str(uuid.uuid4())[:8]
            print(f"new thread {thread_id}")
            continue
        if line == "/ledger":
            from tools.ledger import read_ledger

            print(read_ledger.invoke({"status": "all", "limit": 25}))
            continue
        if line.startswith("/provider"):
            parts = line.split()
            if len(parts) < 2:
                print(f"usage: /provider <{'|'.join(PROVIDERS)}> [model]")
                continue
            try:
                info = set_provider(parts[1], parts[2] if len(parts) > 2 else None)
                graph = build_graph()  # rebuild so nodes pick up the new model
                print(f"switched -> {info['label']} ({info['model']})")
            except Exception as exc:
                print(f"could not switch: {exc}")
            continue

        try:
            state = _run(graph, line, thread_id)
        except Exception as exc:
            print(f"\n[run failed] {type(exc).__name__}: {exc}")
            continue

        print("\n" + "=" * 78)
        print(state.get("final") or state.get("draft") or "(no answer)")
        print("=" * 78)
        print(f"agents={state.get('visited')} critique_iterations={state.get('iteration')} "
              f"critic_confidence={state.get('critic_confidence')}")
        if state.get("critic_issues"):
            print("open critic issues: " + "; ".join(state["critic_issues"]))


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        return chat()
    cmd = argv[0]
    if cmd == "providers":
        return _print_providers()
    if cmd == "ledger":
        from tools.ledger import list_findings

        print(json.dumps(list_findings("all", 50), indent=2))
        return
    if cmd == "ask":
        from graph.supervisor import ask

        state = ask(" ".join(argv[1:]))
        print(state.get("final") or state.get("draft"))
        return
    print(__doc__ or HELP)


if __name__ == "__main__":
    main()
