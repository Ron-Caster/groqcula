"""
AI Agent Template — LangGraph Multi-Stage Agent
Pipeline: PROMPT_GUARD → ROUTER → PLAN → EXECUTE → SAFETY_CHECK
Multi-model system: each stage uses the optimal model via models.py.

TODO: Customize the prompts, routing categories, and execution logic for your domain.
"""

import json as _json
import os
import sys
import asyncio
import operator
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from models import (
    get_llm,
    get_model_name,
    check_prompt_safety,
    check_output_safety,
    GROQ_API_KEY,
    ENABLE_ROUTER,
    ENABLE_PLANNER,
    ENABLE_PROMPT_GUARD,
    ENABLE_OUTPUT_SAFETY,
    _fallback_logger,
)


# ── CUSTOMIZE: Prompts ──────────────────────────────────────────────────────

# TODO: Update the routing categories for your domain.
ROUTER_PROMPT = """You are a query classifier for an AI agent. Given the user's message, respond with EXACTLY one word.

The agent has the following capabilities:
- Web search (search the internet for any information)
- News search (find recent news articles on any topic)
- Get current date and time

Categories:
- "task" — ANY request that can be answered using web search, news search, or time lookup. This includes questions about ANY topic, current events, people, companies, technology, science, etc.
- "greeting" — ONLY simple greetings or questions about the bot itself. Examples: "hi", "hello", "who are you?", "what can you do?".
- "unrelated" — ONLY requests that require capabilities the agent does NOT have, such as: writing code, editing files, doing math calculations, generating images, or sending emails.

RULES:
- When in doubt, ALWAYS choose "task".
- If the question can be answered by searching the web or looking up news, it is ALWAYS "task".
- Almost everything is "task" — the agent can search for information on virtually any topic.

Respond with ONLY one word: task, greeting, or unrelated."""

# TODO: Update with your domain's available tools and planning instructions.
PLANNER_PROMPT = """You are a task planner for an AI agent.
Given the user's question, produce a short, numbered plan of actions.

Available tools:
- `web_search` — search the web for general information using DuckDuckGo.
- `web_search_news` — search for recent news articles using DuckDuckGo News. Use this for any news-related queries.
- `get_current_time` — get the current date and time.

RULES:
1. Pick the MOST SPECIFIC tool for the question.
2. Each step must reference one of the tools listed above.
3. Keep the plan to 2-3 steps maximum.

Output ONLY the numbered plan. Do NOT execute or call any tools."""

# TODO: Update with your domain's output format and presentation rules.
EXECUTOR_PROMPT = """You are an AI assistant that executes plans using the available tools.
Your job is to call the right tools and present the results clearly and concisely.

RULES:
1. Follow the plan step by step.
2. Use the tool outputs as the source of truth — do not fabricate data.
3. Present results in a clean, well-formatted markdown response.
4. Be concise and professional.

Start by calling the appropriate tool as described in the plan."""

# TODO: Customize the greeting and rejection messages for your agent.
GREETING_RESPONSE = """Hello! I'm an **AI Agent** 🤖

I can help you with various tasks. Here's what I can do:

- 🔍 **Web Search** — search the internet for any information
- 📰 **News Search** — find recent news articles on any topic
- 🕐 **Current Time** — get the current date and time

Ask me a question to get started!"""

UNRELATED_RESPONSE = """I'm an **AI Agent** 🤖

That's outside what I can do. I'm best at:

- 🔍 **Web Search** — searching the internet
- 📰 **News Search** — finding recent news
- 🕐 **Current Time** — getting the date/time

Try asking me to search for something!"""

INJECTION_RESPONSE = """⚠️ Your message was flagged by our safety system.

Please rephrase your question as a valid request."""

UNSAFE_OUTPUT_RESPONSE = """⚠️ The response was flagged by our safety system and cannot be displayed.

Please try rephrasing your question."""


# ── State ───────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    plan: str
    query_type: str


# ── Graph Builder ───────────────────────────────────────────────────────────

def build_agent_graph(api_key: str, tools: list):
    """Build a LangGraph StateGraph with guard → router → plan → execute → safety → END.
    Uses different models for each stage via models.py."""

    # ── Prompt Guard node ──────────────────────────────────────────────
    async def prompt_guard_node(state: AgentState) -> dict:
        """Check user input for prompt injection / jailbreak attempts."""
        user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        last_msg = user_messages[-1].content if user_messages else ""

        result = await check_prompt_safety(last_msg, api_key)
        label = result["label"]

        if result["safe"]:
            print(f"🛡️  GUARD: safe ({label})")
            return {"query_type": ""}
        else:
            print(f"🛡️  GUARD: BLOCKED ({label})")
            return {"query_type": "injection"}

    # ── Guard decision ─────────────────────────────────────────────────
    def guard_decision(state: AgentState) -> Literal["router", "injection"]:
        return "injection" if state.get("query_type") == "injection" else "router"

    # ── Injection node ─────────────────────────────────────────────────
    async def injection_node(state: AgentState) -> dict:
        """Respond when prompt injection is detected."""
        return {"messages": [AIMessage(content=INJECTION_RESPONSE)], "plan": ""}

    # ── Router node ────────────────────────────────────────────────────
    async def router_node(state: AgentState) -> dict:
        """Classify user query as task, greeting, or unrelated."""
        user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        last_msg = user_messages[-1].content if user_messages else ""

        router_llm = get_llm("router", api_key)
        response = await router_llm.ainvoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=last_msg),
        ])

        query_type = response.content.strip().lower()
        if query_type not in ("task", "greeting", "unrelated"):
            query_type = "task"

        print(f"🔀 ROUTER [{get_model_name('router')}]: {query_type}")
        return {"query_type": query_type}

    # ── Route decision ─────────────────────────────────────────────────
    def route_decision(state: AgentState) -> Literal["plan", "greet", "unrelated"]:
        qt = state.get("query_type", "task")
        if qt == "greeting":
            return "greet"
        elif qt == "unrelated":
            return "unrelated"
        return "plan"

    # ── Greeting node ──────────────────────────────────────────────────
    async def greet_node(state: AgentState) -> dict:
        return {"messages": [AIMessage(content=GREETING_RESPONSE)], "plan": ""}

    # ── Unrelated node ─────────────────────────────────────────────────
    async def unrelated_node(state: AgentState) -> dict:
        return {"messages": [AIMessage(content=UNRELATED_RESPONSE)], "plan": ""}

    # ── Plan node ──────────────────────────────────────────────────────
    async def plan_node(state: AgentState) -> dict:
        """Generate an action plan before executing."""
        user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        last_user_msg = user_messages[-1].content if user_messages else "Help me."

        planner_llm = get_llm("planner", api_key)
        plan_response = await planner_llm.ainvoke([
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=last_user_msg),
        ])

        plan_text = plan_response.content
        print(f"\n📋 PLAN [{get_model_name('planner')}]:\n{plan_text}\n")
        return {"plan": plan_text}

    # ── Execute node ───────────────────────────────────────────────────
    executor_llm = get_llm("executor", api_key)
    react_agent = create_react_agent(executor_llm, tools)

    async def execute_node(state: AgentState) -> dict:
        """Execute using a ReAct agent with tools. Adapts to planner on/off."""
        plan_text = state.get("plan", "")

        if plan_text:
            # Planner was active — follow the plan
            exec_messages = [
                SystemMessage(content=EXECUTOR_PROMPT),
                HumanMessage(content=f"**PLAN TO FOLLOW:**\n{plan_text}\n\nExecute this plan now."),
            ]
        else:
            # No planner — pass user message directly (general-purpose chatbot mode)
            user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
            last_user_msg = user_messages[-1].content if user_messages else "Hello"
            exec_messages = [
                SystemMessage(content=EXECUTOR_PROMPT),
                HumanMessage(content=last_user_msg),
            ]

        result = await react_agent.ainvoke({"messages": exec_messages})
        final_content = result["messages"][-1].content

        actual_model = _fallback_logger.last_model or get_model_name('executor')
        print(f"📝 EXECUTOR [{actual_model}]: done")
        return {
            "messages": [AIMessage(content=final_content)],
        }

    # ── Output Safety node ─────────────────────────────────────────────
    async def safety_check_node(state: AgentState) -> dict:
        """Check if the executor output is safe to display."""
        last_msg = state["messages"][-1].content if state["messages"] else ""

        result = await check_output_safety(last_msg, api_key)
        print(f"🔒 SAFETY [{get_model_name('output_safety')}]: {result['label']}")

        if not result["safe"]:
            return {"messages": [AIMessage(content=UNSAFE_OUTPUT_RESPONSE)]}
        return {}

    # ── Wire the graph ─────────────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("execute", execute_node)

    # Determine the first node after START (guard → router → plan → execute)
    if ENABLE_PROMPT_GUARD:
        graph.add_node("guard", prompt_guard_node)
        graph.add_node("injection", injection_node)
        graph.add_edge(START, "guard")
        first_after_guard = "router" if ENABLE_ROUTER else ("plan" if ENABLE_PLANNER else "execute")
        graph.add_conditional_edges("guard", guard_decision, {"router": first_after_guard, "injection": "injection"})
        graph.add_edge("injection", END)
    else:
        first_node = "router" if ENABLE_ROUTER else ("plan" if ENABLE_PLANNER else "execute")
        graph.add_edge(START, first_node)

    # Router stage (optional)
    if ENABLE_ROUTER:
        graph.add_node("router", router_node)
        graph.add_node("greet", greet_node)
        graph.add_node("unrelated", unrelated_node)
        next_after_router = "plan" if ENABLE_PLANNER else "execute"
        graph.add_conditional_edges("router", route_decision, {"plan": next_after_router, "greet": "greet", "unrelated": "unrelated"})
        graph.add_edge("greet", END)
        graph.add_edge("unrelated", END)

    # Planner stage (optional)
    if ENABLE_PLANNER:
        graph.add_node("plan", plan_node)
        graph.add_edge("plan", "execute")
    elif ENABLE_ROUTER:
        pass  # router already edges to execute
    # else: START already edges to execute

    # Output safety (optional)
    if ENABLE_OUTPUT_SAFETY:
        graph.add_node("safety_check", safety_check_node)
        graph.add_edge("execute", "safety_check")
        graph.add_edge("safety_check", END)
    else:
        graph.add_edge("execute", END)

    return graph.compile()


# ── Run Functions ───────────────────────────────────────────────────────────

async def run_agent(user_query: str, tools: list):
    """Run the agent with direct LangChain tools (no MCP server needed)."""
    if not GROQ_API_KEY:
        print("ERROR: Set GROQ_API_KEY environment variable.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  AI Agent — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    graph = build_agent_graph(GROQ_API_KEY, tools)
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=user_query)], "plan": "", "query_type": ""}
    )

    final_msg = result["messages"][-1].content
    print(final_msg)
    return result


def main():
    """Entry point — run a single query from CLI."""
    from tools import ALL_TOOLS

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello, what can you do?"
    asyncio.run(run_agent(query, ALL_TOOLS))


if __name__ == "__main__":
    main()
