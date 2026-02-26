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

import deepagents.graph
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
You have access to the following built-in tools:
- **web_assistant** — search the web or visit specific websites using natural language instructions
- **get_current_time** — get the current local date and time

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
        "using the available `web_assistant` tool and produce "
        "a polished, well-structured report with sources cited."
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

    Chat history compression is forced to trigger at 4000 tokens to 
    ensure it never hits Groq's 8000 Tokens-Per-Minute limit.
    """
    model = get_chat_model(api_key)

    # Monkey patch the built-in summarization defaults to force it to compress memory
    # BEFORE we hit Groq's tiny 8000 TPM limit. (Normally it triggers at ~100k+ tokens
    # based on the model's actual context window, which kills Groq free tier).
    original_compute = deepagents.graph._compute_summarization_defaults
    def patched_compute(model):
        res = original_compute(model)
        # Deep Agents summarizer sees these as internal thresholds (not API limits)
        res['trigger'] = ("tokens", 4000)   # When context hits ~4000 tokens, compress it!
        res['keep'] = ("tokens", 2000)      # Keep the most recent 2000 tokens
        return res
    deepagents.graph._compute_summarization_defaults = patched_compute

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
        if user_input.lower() == "clear":
            # Generate a new thread ID to start a fresh memory state
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            print(f"🧹 History cleared! Started fresh session: {thread_id[:8]}\n")
            continue

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            final_msg = result["messages"][-1].content
            print(f"\n🤖 {final_msg}\n")
        except Exception as e:
            error_str = str(e)
            print(f"\n⚠️ Error: {error_str}\n")
            if "rate_limit_exceeded" in error_str or "413" in error_str:
                print("💡 Groq Free Tier has an 8,000 Tokens-Per-Minute limit.")
                print("💡 Since the checkpointer remembers history, your conversation is getting too long.")
                print("💡 Type 'clear' to reset your chat history and free up tokens!\n")


if __name__ == "__main__":
    main()
