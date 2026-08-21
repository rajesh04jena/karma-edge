################################################################################
# Karma Edge - app/ui.py
#
# Optional Streamlit chat UI with a provider picker in the sidebar.
#   streamlit run app/ui.py
################################################################################
from __future__ import annotations

import uuid

import streamlit as st

from app.config import settings
from app.llm import PROVIDERS, provider_status, set_provider

st.set_page_config(page_title="Karma Edge", page_icon="•", layout="wide")


@st.cache_resource
def _graph(provider: str, model: str):
    from graph.supervisor import build_graph

    return build_graph()


st.sidebar.title("Karma Edge")
st.sidebar.caption("margin accountability, with names attached to the numbers")

status = {p["provider"]: p for p in provider_status()}
labels = [f"{k} {'✓' if status[k]['ready'] else '(no key)'}" for k in PROVIDERS]
choice = st.sidebar.selectbox("Model provider", labels,
                              index=list(PROVIDERS).index(settings.model_provider))
provider = choice.split()[0]
model = st.sidebar.selectbox("Model", PROVIDERS[provider].models)
st.sidebar.caption(PROVIDERS[provider].notes)

if st.sidebar.button("Apply provider", use_container_width=True):
    try:
        info = set_provider(provider, model)
        st.cache_resource.clear()
        st.sidebar.success(f"switched to {info['label']}")
    except Exception as exc:
        st.sidebar.error(str(exc))

st.sidebar.divider()
if st.sidebar.button("Show ledger", use_container_width=True):
    from tools.ledger import list_findings

    st.sidebar.dataframe(list_findings("all", 50))

if "thread" not in st.session_state:
    st.session_state.thread = str(uuid.uuid4())[:8]
if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

if prompt := st.chat_input("Where is margin leaking, and whose fault is it?"):
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    from graph.state import new_state

    graph = _graph(settings.model_provider, settings.model_name)
    cfg = {"configurable": {"thread_id": st.session_state.thread},
           "recursion_limit": settings.graph_recursion_limit}
    with st.chat_message("assistant"):
        with st.status("agents working...", expanded=False) as box:
            state = graph.invoke(new_state(prompt, st.session_state.thread), cfg)
            snap = graph.get_state(cfg)
            if snap.next and "hitl" in snap.next:
                box.update(label="human gate: auto-approving in UI mode")
                graph.update_state(cfg, {"hitl_decision": "approve"})
                state = graph.invoke(None, cfg)
            box.update(label=f"agents: {state.get('visited')}", state="complete")
        answer = state.get("final") or state.get("draft") or "(no answer)"
        st.markdown(answer)
        with st.expander("critique trail"):
            st.write({"iterations": state.get("iteration"),
                      "confidence": state.get("critic_confidence"),
                      "issues": state.get("critic_issues")})
    st.session_state.history.append(("assistant", answer))
