"""
Multi-Model Configuration — Agent Template
Each agent stage uses the optimal model for cost, speed, and quality.
Includes optional prompt injection guard and output safety checking.

Model assignments:
┌─────────────────┬──────────────────────────────────────┬──────────┐
│ Role            │ Model                                │ RPM / TPM│
├─────────────────┼──────────────────────────────────────┼──────────┤
│ Router          │ llama-3.1-8b-instant                 │ 30 / 6K  │
│ Planner         │ llama-3.1-8b-instant                 │ 30 / 6K  │
│ Executor        │ groq/compound                        │ 30 / 70K │
│ Prompt Guard    │ llama-prompt-guard-2-86m             │ 30 / 15K │
│ Output Safety   │ openai/gpt-oss-safeguard-20b         │ 30 / 8K  │
└─────────────────┴──────────────────────────────────────┴──────────┘
"""

import os
import re
from langchain_groq import ChatGroq
from langchain_core.callbacks import BaseCallbackHandler

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── Feature Toggles ────────────────────────────────────────────────────────
# Set to True to enable safety layers. Both are off by default for speed.
ENABLE_ROUTER = True           # Query classification (task/greeting/unrelated)
ENABLE_PLANNER = True          # Action planning before execution
ENABLE_PROMPT_GUARD = False    # Prompt injection detection (adds ~1-2s latency)
ENABLE_OUTPUT_SAFETY = False   # Output safety check (adds ~2s latency)


# ── Model Registry ─────────────────────────────────────────────────────────
# Each role has a primary model and a list of fallbacks (tried in order).
# TODO: Adjust models, temperatures, and fallbacks for your use case.

MODEL_REGISTRY = {
    "router": {
        "primary": "llama-3.1-8b-instant",
        "fallbacks": [
            "groq/compound",
            "groq/compound-mini",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-safeguard-20b",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "moonshotai/kimi-k2-instruct-0905",
            "qwen/qwen3-32b",
            "llama-3.3-70b-versatile"
        ],
        "temperature": 0,
        "description": "1-word classification. Speed > quality.",
    },
    "planner": {
        "primary": "llama-3.1-8b-instant",
        "fallbacks": [
            "groq/compound",
            "groq/compound-mini",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-safeguard-20b",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "moonshotai/kimi-k2-instruct-0905",
            "qwen/qwen3-32b",
            "llama-3.3-70b-versatile",
        ],
        "temperature": 0,
        "description": "Short 2-3 step plan. Speed > quality.",
    },
    "executor": {
        "primary": "openai/gpt-oss-120b",
        "fallbacks": [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-safeguard-20b",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "moonshotai/kimi-k2-instruct-0905",
            "qwen/qwen3-32b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ],
        "temperature": 0,
        "description": "Tool calling and report generation. Quality > speed.",
    },
    "prompt_guard": {
        "primary": "meta-llama/llama-prompt-guard-2-86m",
        "fallbacks": ["meta-llama/llama-prompt-guard-2-22m"],
        "temperature": 0,
        "description": "Detect prompt injection / jailbreak attempts on user input.",
    },
    "output_safety": {
        "primary": "openai/gpt-oss-safeguard-20b",
        "fallbacks": ["openai/gpt-oss-20b"],
        "temperature": 0,
        "description": "Check agent output is safe and appropriate before displaying.",
    },
}


# ── LLM Factory ─────────────────────────────────────────────────────────────

# Groq error code → (emoji, short label)
_GROQ_ERRORS = {
    206: ("ℹ️", "Partial Content"),
    400: ("🚫", "Bad Request"),
    401: ("🔑", "Unauthorized — check API key"),
    403: ("🚷", "Forbidden — permission denied"),
    404: ("❓", "Model Not Found"),
    413: ("📦", "Request Too Large"),
    422: ("⚠️", "Unprocessable — semantic error"),
    424: ("🔗", "Failed Dependency"),
    429: ("⏳", "Rate Limited"),
    498: ("📈", "Flex Tier Capacity Exceeded"),
    499: ("🚫", "Request Cancelled"),
    500: ("💥", "Internal Server Error — retry later"),
    502: ("🌐", "Bad Gateway — upstream error"),
    503: ("🔧", "Service Unavailable — maintenance/overload"),
}


class _FallbackLogger(BaseCallbackHandler):
    """Tracks Groq errors and logs which fallback model succeeds."""
    def __init__(self):
        self.last_model = None
        self._had_error = False
        self._current_model = "unknown"

    def on_chat_model_start(self, serialized, messages, **kwargs):
        """Track which model is currently being attempted."""
        self._current_model = serialized.get("kwargs", {}).get("model", "unknown")

    def on_llm_error(self, error, **kwargs):
        self._had_error = True
        err_str = str(error)
        match = re.search(r"Error code: (\d+)", err_str)
        if match:
            code = int(match.group(1))
            emoji, label = _GROQ_ERRORS.get(code, ("❌", f"Error {code}"))
            model_match = re.search(r"model\s+`?(\S+?)`?\s+in", err_str)
            model = model_match.group(1) if model_match else self._current_model
            print(f"{emoji} {label} [{model}]")
        else:
            print(f"❌ {err_str[:80]} [{self._current_model}]")

    def on_llm_end(self, response, **kwargs):
        model = "unknown"
        if hasattr(response, "llm_output") and response.llm_output:
            model = response.llm_output.get("model_name", "unknown")
        self.last_model = model
        if self._had_error:
            print(f"↪️  Switched to [{model}]")
            self._had_error = False


_fallback_logger = _FallbackLogger()


def get_llm(role: str, api_key: str = "") -> ChatGroq:
    """
    Get the LLM for a specific role, with automatic fallbacks.
    max_retries=0: 429 errors propagate instantly to fallback chain.
    request_=10 seconds: connection errors fail fast instead of waiting 60s+.
    """
    key = api_key or GROQ_API_KEY
    if role not in MODEL_REGISTRY:
        raise ValueError(f"Unknown role: {role}. Available: {list(MODEL_REGISTRY.keys())}")

    config = MODEL_REGISTRY[role]

    primary = ChatGroq(
        model=config["primary"],
        api_key=key,
        temperature=config["temperature"],
        max_retries=0,
        request_timeout=10,
        callbacks=[_fallback_logger],
    )

    if config["fallbacks"]:
        fallback_llms = [
            ChatGroq(
                model=fb,
                api_key=key,
                temperature=config["temperature"],
                max_retries=0,
                request_timeout=10,
                callbacks=[_fallback_logger],
            )
            for fb in config["fallbacks"]
        ]
        return primary.with_fallbacks(fallback_llms)

    return primary


def get_model_name(role: str) -> str:
    """Get the primary model name for a role (for display/logging)."""
    return MODEL_REGISTRY[role]["primary"]


# ── Prompt Safety ───────────────────────────────────────────────────────────

async def check_prompt_safety(text: str, api_key: str = "") -> dict:
    """
    Check if user input contains prompt injection / jailbreak attempts.
    Uses Llama Prompt Guard (86M) — a purpose-built classification model.

    Returns:
        {"safe": True/False, "score": float, "label": str}
    """
    THRESHOLD = 0.5

    try:
        llm = get_llm("prompt_guard", api_key)
        response = await llm.ainvoke(text)
        raw = response.content.strip()

        try:
            score = float(raw)
            is_safe = score < THRESHOLD
            label = f"score={score:.4f} ({'SAFE' if is_safe else 'INJECTION'})"
        except ValueError:
            label = raw.upper()
            is_safe = "JAILBREAK" not in label and "INJECTION" not in label
            score = 0.0 if is_safe else 1.0

        return {"safe": is_safe, "score": score, "label": label}
    except Exception as e:
        return {"safe": True, "score": 0.0, "label": f"GUARD_ERROR: {e}"}


# ── Output Safety Check ────────────────────────────────────────────────────

OUTPUT_SAFETY_PROMPT = """You are a safety classifier. Evaluate if the following AI agent output is safe and appropriate.
Check for: leaked API keys, personal data exposure, harmful content, off-topic responses.
Respond with ONLY one word: SAFE or UNSAFE."""


async def check_output_safety(text: str, api_key: str = "") -> dict:
    """
    Check if agent output is safe to display to the user.

    Returns:
        {"safe": True/False, "label": "SAFE"/"UNSAFE"}
    """
    try:
        llm = get_llm("output_safety", api_key)
        response = await llm.ainvoke([
            {"role": "system", "content": OUTPUT_SAFETY_PROMPT},
            {"role": "user", "content": text},
        ])
        label = response.content.strip().upper()
        return {"safe": "SAFE" in label, "label": label}
    except Exception as e:
        return {"safe": True, "label": f"SAFETY_ERROR: {e}"}


# ── Cost Estimator (informational) ─────────────────────────────────────────

COST_PER_1M = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "meta-llama/llama-prompt-guard-2-86m": {"input": 0.04, "output": 0.04},
    "meta-llama/llama-prompt-guard-2-22m": {"input": 0.03, "output": 0.03},
    "openai/gpt-oss-safeguard-20b": {"input": 0.075, "output": 0.30},
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.11, "output": 0.34},
    "meta-llama/llama-4-maverick-17b-128e-instruct": {"input": 0.20, "output": 0.60},
}


def estimate_query_cost() -> str:
    """Estimate cost per query across all model calls."""
    estimates = {
        "prompt_guard": {"input": 50, "output": 5},
        "router": {"input": 200, "output": 5},
        "planner": {"input": 400, "output": 100},
        "executor": {"input": 1500, "output": 500},
        "output_safety": {"input": 600, "output": 5},
    }

    total = 0.0
    lines = []
    for role, tokens in estimates.items():
        model = MODEL_REGISTRY[role]["primary"]
        cost_info = COST_PER_1M.get(model, {"input": 0, "output": 0})
        cost = (tokens["input"] * cost_info["input"] + tokens["output"] * cost_info["output"]) / 1_000_000
        total += cost
        lines.append(f"  {role}: ~${cost:.6f} ({model})")

    return f"Estimated cost per query: ~${total:.5f}\n" + "\n".join(lines)
