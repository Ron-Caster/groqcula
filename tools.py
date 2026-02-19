"""
LangChain Tools — Agent Template
Define your agent's tools here. Each tool should be a function decorated with @tool.
The agent will use these tools to perform actions during the EXECUTE stage.
"""

from langchain_core.tools import tool
from ddgs import DDGS


# ── CUSTOMIZE: Define your tools below ──────────────────────────────────────

@tool
def web_search(query: str) -> list[dict]:
    """Search the web using DuckDuckGo. Returns a list of results with title, url, and snippet."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except Exception as e:
        return [{"error": str(e)}]


@tool
def web_search_news(query: str) -> list[dict]:
    """Search for recent news articles using DuckDuckGo News. Returns a list of news results with title, url, snippet, date, and source."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=5))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("body", ""),
                "date": r.get("date", ""),
                "source": r.get("source", ""),
            }
            for r in results
        ]
    except Exception as e:
        return [{"error": str(e)}]


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().isoformat()



# ── Tool Registry ──────────────────────────────────────────────────────────
# Add all your tools to this list. The agent will have access to all of them.

ALL_TOOLS = [
    web_search,
    web_search_news,
    get_current_time,
]
