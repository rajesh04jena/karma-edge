################################################################################
# Karma Edge - app/llm.py
#
# One factory for every supported chat model. All providers below expose an
# OpenAI-compatible endpoint, which means LangChain's ChatOpenAI can talk to
# them AND use native tool calling (function calling) with zero custom code.
#
# No Ollama. No local weights. Everything is a hosted API so this runs on an
# ancient MacBook with 8GB of RAM and a fan that sounds like a jet engine.
################################################################################
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.config import settings


@dataclass(frozen=True)
class ProviderSpec:
    key: str
    label: str
    base_url: str
    default_model: str
    api_key_env: str
    models: List[str]
    notes: str
    signup: str


# NOTE: model ids and base URLs change. If a call 404s on the model id, run
# `python -m app.main providers` and check the provider's console for current ids.
PROVIDERS: Dict[str, ProviderSpec] = {
    "zhipu": ProviderSpec(
        key="zhipu",
        label="Zhipu AI / GLM (BigModel)",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-flash",
        api_key_env="ZHIPU_API_KEY",
        models=["glm-4-flash", "glm-4-flashx", "glm-4-air", "glm-4-plus"],
        notes="glm-4-flash is free. Best default: solid tool calling, generous limits.",
        signup="https://open.bigmodel.cn/",
    ),
    "deepseek": ProviderSpec(
        key="deepseek",
        label="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        models=["deepseek-chat", "deepseek-reasoner"],
        notes="Cheapest strong reasoner. Signup credits; tool calling on deepseek-chat.",
        signup="https://platform.deepseek.com/",
    ),
    "qwen": ProviderSpec(
        key="qwen",
        label="Alibaba Qwen (DashScope, international endpoint)",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        api_key_env="DASHSCOPE_API_KEY",
        models=["qwen-turbo", "qwen-plus", "qwen-max", "qwen3-235b-a22b"],
        notes="Free trial quota per model. Use the -intl base URL outside mainland China.",
        signup="https://bailian.console.alibabacloud.com/",
    ),
    "moonshot": ProviderSpec(
        key="moonshot",
        label="Moonshot AI / Kimi",
        base_url="https://api.moonshot.cn/v1",
        default_model="moonshot-v1-8k",
        api_key_env="MOONSHOT_API_KEY",
        models=["moonshot-v1-8k", "moonshot-v1-32k", "kimi-k2-0711-preview"],
        notes="Trial credits. Long-context variants are handy for PDF-ish prompts.",
        signup="https://platform.moonshot.cn/",
    ),
    "siliconflow": ProviderSpec(
        key="siliconflow",
        label="SiliconFlow (hosts Qwen / GLM / DeepSeek)",
        base_url="https://api.siliconflow.cn/v1",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        api_key_env="SILICONFLOW_API_KEY",
        models=[
            "Qwen/Qwen2.5-7B-Instruct",
            "THUDM/glm-4-9b-chat",
            "deepseek-ai/DeepSeek-V3",
        ],
        notes="Several models are free tier. One key, many Chinese open models.",
        signup="https://cloud.siliconflow.cn/",
    ),
    "openrouter": ProviderSpec(
        key="openrouter",
        label="OpenRouter (aggregator, has :free variants)",
        base_url="https://openrouter.ai/api/v1",
        default_model="deepseek/deepseek-chat-v3-0324:free",
        api_key_env="OPENROUTER_API_KEY",
        models=[
            "deepseek/deepseek-chat-v3-0324:free",
            "qwen/qwen-2.5-72b-instruct:free",
            "z-ai/glm-4.5-air:free",
        ],
        notes="One key, many free models. Free variants are rate limited and rotate.",
        signup="https://openrouter.ai/",
    ),
    "fake": ProviderSpec(
        key="fake",
        label="Fake / scripted (offline)",
        base_url="",
        default_model="scripted-analyst",
        api_key_env="",
        models=["scripted-analyst"],
        notes="Deterministic stub. No keys, no network. Used by the test suite.",
        signup="",
    ),
}


def provider_status() -> List[Dict[str, Any]]:
    """What the /providers endpoint and the `/provider` chat command render."""
    out = []
    for spec in PROVIDERS.values():
        ready = True if not spec.api_key_env else bool(os.environ.get(spec.api_key_env, "").strip())
        out.append(
            {
                "provider": spec.key,
                "label": spec.label,
                "default_model": spec.default_model,
                "models": spec.models,
                "api_key_env": spec.api_key_env or "(none needed)",
                "ready": ready,
                "notes": spec.notes,
                "signup": spec.signup,
            }
        )
    return out


_CACHE: Dict[tuple, Any] = {}


def get_llm(provider: Optional[str] = None, model: Optional[str] = None, temperature: Optional[float] = None):
    """Return a LangChain chat model for `provider`. Lazy + cached.

    A missing key for provider A never breaks provider B, because nothing is
    constructed until it is actually asked for.
    """
    provider = (provider or settings.model_provider or "fake").lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}. Options: {', '.join(PROVIDERS)}")
    spec = PROVIDERS[provider]
    model = model or settings.model_name or spec.default_model
    temp = settings.temperature if temperature is None else temperature

    cache_key = (provider, model, temp)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    if provider == "fake":
        from app.fake_llm import ScriptedChatModel

        llm = ScriptedChatModel(model_name=model)
    else:
        api_key = os.environ.get(spec.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{spec.api_key_env} is not set. Get a key at {spec.signup} and put it in .env, "
                f"or run with MODEL_PROVIDER=fake for an offline dry run."
            )
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=spec.base_url,
            temperature=temp,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

    _CACHE[cache_key] = llm
    return llm


def set_provider(provider: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Switch provider at RUNTIME, mid-conversation. Used by the chatbot's
    `/provider` command and the Streamlit sidebar picker.

    Mutates the process-wide settings, then clears the agent cache so every
    ReAct sub-agent is rebuilt against the new model on its next turn.
    """
    provider = (provider or "").lower().strip()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}. Options: {', '.join(PROVIDERS)}")
    spec = PROVIDERS[provider]
    if provider != "fake" and not os.environ.get(spec.api_key_env, "").strip():
        raise RuntimeError(f"{spec.api_key_env} is not set. Get a key at {spec.signup}.")

    settings.model_provider = provider
    settings.model_name = model or spec.default_model

    from graph.nodes import reset_agent_cache  # local import avoids a cycle

    reset_agent_cache()
    return {"provider": provider, "model": settings.model_name, "label": spec.label}

