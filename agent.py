"""
Groqcula — Deep Agent Builder
Uses the LangChain Deep Agents SDK to build a powerful agent with:
- Built-in planning (write_todos)
- Context management (virtual filesystem)
- Subagent delegation (task tool)
- Automatic tool calling
- Long-term memory (CompositeBackend with StoreBackend)
- Chat history compression (SummarizationMiddleware)

Replaces the manual LangGraph pipeline (guard → router → plan → execute → safety)
with a single create_deep_agent() call that handles all of this internally.
"""

import os
import sys
import asyncio
import uuid
from datetime import datetime

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from models import get_chat_model, get_primary_model_name, GROQ_API_KEY
from tools import ALL_TOOLS


# ── System Prompt ───────────────────────────────────────────────────────────
# Built as a function so we can inject the current date/time dynamically.
# Planning, routing, and execution are handled by the SDK's built-in
# middleware (TodoListMiddleware, SubAgentMiddleware, SummarizationMiddleware)

def get_system_prompt() -> str:
    """Generate the system prompt with the current date/time injected."""
    now = datetime.now()
    return f"""\
You are an AI Agent 🤖 — a helpful, multi-capable assistant powered by Groq.

## Current Date & Time
Today is {now.strftime('%A, %B %d, %Y')} and the current time is {now.strftime('%I:%M %p %Z')} (IST).
Always use this as your reference for "today", "upcoming", "recent", etc.
When searching for events or news, only return results that are AFTER today's date.
Discard any results from the past.

## Your Capabilities
You have access to the following tools:
- **search** — search the web using DuckDuckGo. Use topic='general' for web pages or topic='news' for recent news articles.
- **get_current_time** — get the current date and time
- **scrape_webpage** — visit any URL and extract its full text content. Use this to read articles, event pages, documentation, etc.

You also have filesystem tools (read_file, write_file, edit_file, ls) for managing context.

## How to Work
1. For complex requests, break the task into steps using the todo list.
2. Use the most specific tool for each step.
3. Present results in clean, well-formatted markdown.
4. Be concise and professional.
5. Use tool outputs as the source of truth — do not fabricate data.
6. When searching for events, include the current year ({now.year}) in your search queries.

## Long-Term Memory
You have persistent memory stored under `/memories/`. Use it to remember things across conversations:
- `/memories/preferences.txt` — User preferences and settings
- `/memories/context/` — Long-term context about the user
- `/memories/knowledge/` — Facts and information learned over time
- `/memories/research/` — Research notes and findings

When the user tells you their preferences or asks you to remember something, save it to `/memories/`.
At the start of conversations, check `/memories/` to recall what you know about the user.
Files under `/memories/` persist across all conversations.
Files elsewhere (like `/notes.txt` or `/draft.md`) are ephemeral and lost when the session ends.

## Greetings
If the user greets you or asks what you can do, introduce yourself and list your capabilities.

## Out of Scope
If asked to do something outside your capabilities (code execution, image generation, email, etc.),
politely explain what you can do instead.
"""

# ── Research Subagent ──────────────────────────────────────────────────────
# A specialized subagent for in-depth research tasks.

RESEARCH_SUBAGENT = {
    "name": "researcher",
    "description": "Used for in-depth research on complex topics. Delegates web searching and news gathering to produce comprehensive reports.",
    "system_prompt": (
        "You are an expert researcher. Your job is to conduct thorough research "
        "using the available search tools and produce a polished, well-structured "
        "report with sources cited. Use web_search for general queries and "
        "web_search_news for recent events."
    ),
    "tools": ALL_TOOLS,
}


# ── Shared store + checkpointer (module-level singletons) ─────────────────
# These must persist across the lifetime of the process so that:
# - MemorySaver retains conversation threads
# - InMemoryStore retains /memories/ files across threads

_store = InMemoryStore()
_checkpointer = MemorySaver()


def build_deep_agent(api_key: str = ""):
    """
    Build a Deep Agent using the Deep Agents SDK.

    Memory architecture (CompositeBackend):
    ┌────────────────────┐
    │   Deep Agent       │
    │                    │
    │  /memories/* ──────┼──► StoreBackend  (persistent across threads)
    │  everything else ──┼──► StateBackend  (ephemeral, single thread)
    └────────────────────┘

    Chat history compression is handled by the built-in SummarizationMiddleware
    which condenses older messages to stay within context limits.
    """
    model = get_chat_model(api_key)

    def make_backend(runtime):
        return CompositeBackend(
            default=StateBackend(runtime),       # Ephemeral (per-thread)
            routes={
                "/memories/": StoreBackend(runtime)  # Persistent (cross-thread)
            }
        )

    agent = create_deep_agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=get_system_prompt(),
        subagents=[RESEARCH_SUBAGENT],
        backend=make_backend,
        store=_store,
        checkpointer=_checkpointer,
    )

    return agent


# ── Interactive CLI ─────────────────────────────────────────────────────────

def main():
    """Interactive chat loop — keeps conversation going until you quit."""
    if not GROQ_API_KEY:
        print("ERROR: Set GROQ_API_KEY environment variable.")
        sys.exit(1)

    # Each session gets a unique thread_id for conversation memory
    thread_id = str(uuid.uuid4())

    print(f"\n{'='*60}")
    print(f"  🧛 Groqcula Deep Agent")
    print(f"  Model: {get_primary_model_name()}")
    print(f"  Session: {thread_id[:8]}")
    print(f"  Memory: /memories/ (persistent) + ephemeral state")
    print(f"  Type 'exit' or 'quit' to leave. Ctrl+C also works.")
    print(f"{'='*60}\n")

    agent = build_deep_agent(GROQ_API_KEY)
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("👋 Bye!")
            break

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            final_msg = result["messages"][-1].content
            print(f"\n🤖 {final_msg}\n")
        except Exception as e:
            print(f"\n⚠️ Error: {e}\n")


if __name__ == "__main__":
    main()
