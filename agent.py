"""
Groqcula — Deep Agent Builder
Uses the LangChain Deep Agents SDK to build a powerful agent with:
- Built-in planning (write_todos)
- Context management (virtual filesystem)
- Subagent delegation (task tool)
- Automatic tool calling
- Persistent memory via PostgreSQL (checkpointer + store)
- Long-term memory (CompositeBackend with StoreBackend)
- Chat history compression (SummarizationMiddleware)

Memory architecture:
┌──────────────────────────────────────────────────────────┐
│  Short-term memory (PostgresSaver checkpointer)          │
│  → Conversation threads persisted to Postgres            │
│  → Survives restarts, resumable by thread_id             │
│                                                          │
│  Long-term memory (CompositeBackend)                     │
│  → /memories/* → PostgresStore (persistent cross-thread) │
│  → everything else → StateBackend (ephemeral per-thread) │
└──────────────────────────────────────────────────────────┘

Set DATABASE_URL env var to connect to Postgres:
  export DATABASE_URL="postgresql://user:pass@localhost:5432/groqcula"

Falls back to in-memory storage if DATABASE_URL is not set (dev mode).
"""

import os
import sys
import asyncio
import uuid
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from models import get_chat_model, get_primary_model_name, GROQ_API_KEY
from tools import ALL_TOOLS


# ── Database Config ────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "")


# ── System Prompt ───────────────────────────────────────────────────────────

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
Files under `/memories/` persist across all conversations and survive restarts.
Files elsewhere (like `/notes.txt` or `/draft.md`) are ephemeral and lost when the session ends.

## Greetings
If the user greets you or asks what you can do, introduce yourself and list your capabilities.

## Out of Scope
If asked to do something outside your capabilities (code execution, image generation, email, etc.),
politely explain what you can do instead.
"""

# ── Research Subagent ──────────────────────────────────────────────────────

RESEARCH_SUBAGENT = {
    "name": "researcher",
    "description": "Used for in-depth research on complex topics. Delegates web searching and news gathering to produce comprehensive reports.",
    "system_prompt": (
        "You are an expert researcher. Your job is to conduct thorough research "
        "using the available search tools and produce a polished, well-structured "
        "report with sources cited. Use search(topic='general') for general queries and "
        "search(topic='news') for recent events."
    ),
    "tools": ALL_TOOLS,
}


# ── Persistence Layer ──────────────────────────────────────────────────────

def _create_postgres_persistence():
    """
    Create PostgreSQL-backed checkpointer and store.
    Both connect to the same database.
    Runs setup/migrations on first use.
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore

    # Create checkpointer (short-term: conversation threads)
    checkpointer_ctx = PostgresSaver.from_conn_string(DATABASE_URL)
    checkpointer = checkpointer_ctx.__enter__()
    checkpointer.setup()  # Run migrations

    # Create store (long-term: /memories/ files)
    store_ctx = PostgresStore.from_conn_string(DATABASE_URL)
    store = store_ctx.__enter__()
    store.setup()  # Run migrations

    return checkpointer, store


def _create_memory_persistence():
    """
    Create in-memory checkpointer and store (dev/fallback mode).
    Data is lost on restart.
    """
    return MemorySaver(), InMemoryStore()


def get_persistence():
    """
    Get the appropriate persistence layer based on DATABASE_URL.
    Returns (checkpointer, store) tuple.
    """
    if DATABASE_URL:
        try:
            checkpointer, store = _create_postgres_persistence()
            print(f"  💾 Storage: PostgreSQL ({DATABASE_URL[:40]}...)")
            return checkpointer, store
        except Exception as e:
            print(f"  ⚠️ Postgres connection failed: {e}")
            print(f"  💾 Falling back to in-memory storage")
            return _create_memory_persistence()
    else:
        print(f"  💾 Storage: In-memory (set DATABASE_URL for persistence)")
        return _create_memory_persistence()


# ── Agent Builder ──────────────────────────────────────────────────────────

def build_deep_agent(api_key: str, checkpointer, store):
    """
    Build a Deep Agent using the Deep Agents SDK.

    Memory architecture (CompositeBackend):
    ┌────────────────────────────────────────────────────────┐
    │   Deep Agent                                           │
    │                                                        │
    │  /memories/* ──► StoreBackend  (Postgres / persistent) │
    │  everything  ──► StateBackend  (ephemeral, per-thread) │
    │                                                        │
    │  Checkpointer ── PostgresSaver (conversation threads)  │
    │  SummarizationMiddleware (auto-compresses chat history)│
    └────────────────────────────────────────────────────────┘
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
        store=store,
        checkpointer=checkpointer,
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

    # Initialize persistence (Postgres or in-memory fallback)
    checkpointer, store = get_persistence()

    print(f"  Memory: /memories/ (persistent) + ephemeral state")
    print(f"  Type 'exit' or 'quit' to leave. Ctrl+C also works.")
    print(f"{'='*60}\n")

    agent = build_deep_agent(GROQ_API_KEY, checkpointer, store)
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
