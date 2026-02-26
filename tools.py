import os
from datetime import datetime
from groq import Groq

# ── Groq Agentic Tools ──────────────────────────────────────────────────────

def get_groq_client() -> Groq:
    return Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

def web_assistant(instruction: str) -> str:
    """Use this tool for ANY web-related task: searching the web or visiting specific websites.
    Pass a detailed natural language instruction of what you want to achieve on the web.
    For example: 'Search for recent news about OpenAI' or 'Read the article at https://example.com'"""
    client = get_groq_client()
    completion = client.chat.completions.create(
        model="groq/compound",
        messages=[{"role": "user", "content": instruction}],
        temperature=0,
        max_completion_tokens=4096,
        top_p=1,
        stream=False,
        compound_custom={"tools": {"enabled_tools": ["web_search", "visit_website"]}}
    )
    return completion.choices[0].message.content or ""

# ── Current Time ────────────────────────────────────────────────────────────

def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()

# ── Tool Registry ──────────────────────────────────────────────────────────
# Deep Agents accept plain Python functions — no @tool decorator needed.

ALL_TOOLS = [
    web_assistant,
    get_current_time,
]
