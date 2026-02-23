"""
Tools — Groqcula Deep Agent
Define your agent's tools here. Each tool is a plain Python function.
The Deep Agent will automatically convert them into tool-callable functions.
"""

import re
from datetime import datetime
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup


# ── Search ──────────────────────────────────────────────────────────────────

def search(query: str, topic: str = "general") -> list[dict]:
    """Search the web using DuckDuckGo. Set topic to 'general' for web pages or
    'news' for recent news articles. Returns a list of results."""
    try:
        with DDGS() as ddgs:
            if topic == "news":
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
            else:
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


# ── Current Time ────────────────────────────────────────────────────────────

def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()


# ── Web Scraper ─────────────────────────────────────────────────────────────

def scrape_webpage(url: str) -> dict:
    """Scrape a webpage and extract its text content. Give it any URL and it will
    fetch the page, strip out scripts/styles/nav, and return the clean text.
    Use this to read full articles, event pages, documentation, etc."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove truly unwanted elements like scripts and styles
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        # Try to find the most likely content container first,
        # but don't aggressively delete headers/footers because poorly-formed
        # HTML can cause the entire page to be nested inside them!
        main = soup.find("article") or soup.find("main") or soup.find("div", {"role": "main"})
        
        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Truncate continuous newlines and spaces
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Get page title
        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        # Get meta description
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag["content"]

        # Cap the text to avoid blowing up the context window
        # Groq free tier has a strict 8000 TPM limit, so we keep this small (3000 chars = ~750 tokens)
        max_chars = 3000
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... [truncated — {len(text)} total characters]"

        return {
            "url": url,
            "title": title,
            "meta_description": meta_desc,
            "content": text,
            "content_length": len(text),
        }

    except requests.exceptions.Timeout:
        return {"url": url, "error": "Request timed out after 15 seconds"}
    except requests.exceptions.HTTPError as e:
        return {"url": url, "error": f"HTTP error: {e.response.status_code}"}
    except Exception as e:
        return {"url": url, "error": str(e)}


# ── Tool Registry ──────────────────────────────────────────────────────────
# Deep Agents accept plain Python functions — no @tool decorator needed.

ALL_TOOLS = [
    search,
    get_current_time,
    scrape_webpage,
]
