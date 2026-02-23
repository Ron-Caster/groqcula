"""
Model Configuration — Groqcula Deep Agent
Simplified model config for the Deep Agents SDK.
Uses langchain-groq's ChatGroq with fallback chains.

The Deep Agents SDK handles planning, routing, and execution internally —
we only need to provide the model and tools.
"""

import os
from langchain_groq import ChatGroq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


# ── Model Registry ─────────────────────────────────────────────────────────
# Primary and fallback models for the deep agent.
# The Deep Agent SDK uses a single model for all stages, so we pick the best
# general-purpose model and configure fallbacks.

MODEL_REGISTRY = {
    "primary": "openai/gpt-oss-120b",
    "fallbacks": [
        "openai/gpt-oss-20b",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "moonshotai/kimi-k2-instruct-0905",
        "qwen/qwen3-32b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ],
    "temperature": 0,
}


def get_chat_model(api_key: str = "") -> ChatGroq:
    """
    Get a ChatGroq model with automatic fallbacks for the Deep Agent.
    Returns a single LLM object with fallback chain configured.
    """
    key = api_key or GROQ_API_KEY
    config = MODEL_REGISTRY

    primary = ChatGroq(
        model=config["primary"],
        api_key=key,
        temperature=config["temperature"],
        max_retries=0,
        request_timeout=10,
    )

    if config["fallbacks"]:
        fallback_llms = [
            ChatGroq(
                model=fb,
                api_key=key,
                temperature=config["temperature"],
                max_retries=0,
                request_timeout=10,
            )
            for fb in config["fallbacks"]
        ]
        return primary.with_fallbacks(fallback_llms)

    return primary


def get_primary_model_name() -> str:
    """Get the primary model name for display/logging."""
    return MODEL_REGISTRY["primary"]
