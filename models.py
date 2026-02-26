import os
from langchain_groq import ChatGroq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


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
    key = api_key or GROQ_API_KEY
    config = MODEL_REGISTRY

    primary = ChatGroq(
        model=config["primary"],
        api_key=key,
        temperature=config["temperature"],
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
    return MODEL_REGISTRY["primary"]
