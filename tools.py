"""
LangChain Tools — Agent Template
Define your agent's tools here. Each tool should be a function decorated with @tool.
The agent will use these tools to perform actions during the EXECUTE stage.

TODO: Replace the example tools below with your own domain-specific tools.
"""

from langchain_core.tools import tool


# ── CUSTOMIZE: Define your tools below ──────────────────────────────────────

@tool
def example_search(query: str) -> dict:
    """Search for information matching the given query.
    Returns a dict with results. Replace this with your own search logic."""
    # TODO: Replace with your actual search implementation
    return {
        "query": query,
        "results": [
            {"title": "Example Result 1", "summary": f"This is a placeholder result for '{query}'."},
            {"title": "Example Result 2", "summary": "Replace this tool with your real search logic."},
        ],
        "total": 2,
    }


@tool
def example_lookup(item_id: str) -> dict:
    """Look up detailed information for a specific item by its ID.
    Returns a dict with item details. Replace this with your own lookup logic."""
    # TODO: Replace with your actual lookup implementation
    return {
        "item_id": item_id,
        "name": f"Item {item_id}",
        "status": "active",
        "details": "This is placeholder data. Replace this tool with your real lookup logic.",
    }


# ── Tool Registry ──────────────────────────────────────────────────────────
# Add all your tools to this list. The agent will have access to all of them.

ALL_TOOLS = [
    example_search,
    example_lookup,
]
