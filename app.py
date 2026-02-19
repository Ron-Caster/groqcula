"""
Streamlit Chat App — AI Agent Template
Interactive chat UI powered by the LangGraph agent.
Uses direct LangChain tools (no MCP server needed).

TODO: Customize the page title, branding, and example questions for your domain.
"""

import os
import sys
import asyncio
import streamlit as st
from datetime import datetime

from langchain_core.messages import HumanMessage

from agent import build_agent_graph
from models import GROQ_API_KEY
from tools import ALL_TOOLS


def _get_secret(key: str, default: str = "") -> str:
    """Safely read from st.secrets (returns default if no secrets.toml)."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


# ── Page Config ─────────────────────────────────────────────────────────────
# TODO: Update page_title and page_icon for your agent.
st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="wide",
)

st.markdown("# 🤖 AI Agent")
st.caption("Multi-stage AI agent — powered by LangGraph + Groq")

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    groq_key = st.text_input(
        "Groq API Key (optional)",
        value="",
        type="password",
        help="Only needed if not set via env var or secrets.toml",
    )
    _has_key = bool(groq_key or _get_secret("GROQ_API_KEY") or GROQ_API_KEY)
    st.caption("🔑 API Key: " + ("✅ configured" if _has_key else "❌ not set"))
    st.divider()
    # TODO: Update these example questions for your domain.
    st.markdown(
        "**How to use:**\n"
        "Just type a question or request below!\n\n"
        "**Example questions:**\n"
        "- Search for something\n"
        "- Look up item ABC-123\n"
        "- What can you do?\n"
    )


# ── Session State ───────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Display Chat History ────────────────────────────────────────────────────
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        if entry["role"] == "assistant" and entry.get("plan"):
            with st.expander("📋 Agent Plan", expanded=False):
                st.markdown(entry["plan"])
        st.markdown(entry["content"])

# ── Chat Input ──────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask me anything …")

if user_input:
    # Resolve API key: sidebar override → secrets.toml → env var
    api_key = groq_key or _get_secret("GROQ_API_KEY") or GROQ_API_KEY
    if not api_key:
        st.error("Please set GROQ_API_KEY in secrets.toml, env var, or enter it in the sidebar.")
        st.stop()

    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("🔍 Thinking …"):
            async def _run_query():
                graph = build_agent_graph(api_key, ALL_TOOLS)

                result = await graph.ainvoke(
                    {"messages": [HumanMessage(content=user_input)], "plan": "", "query_type": ""}
                )

                plan = result.get("plan", "")
                final = result["messages"][-1].content
                return plan, final

            try:
                plan_text, report = asyncio.run(_run_query())

                # Show plan in an expander
                if plan_text:
                    with st.expander("📋 Agent Plan", expanded=False):
                        st.markdown(plan_text)

                # Show the response
                st.markdown(report)

                # Store for history persistence
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": report, "plan": plan_text}
                )

            except Exception as e:
                error_msg = f"⚠️ Error: {e}"
                import traceback
                st.error(error_msg)
                with st.expander("🔍 Error Details", expanded=False):
                    st.code(traceback.format_exc())
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg, "plan": ""}
                )
