"""Web tools: web_search and web_fetch."""

import html
import json
import logging
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from nanobot.agent.tools.base import Tool

# Setup logger
logger = logging.getLogger("nanobot.web_tools")

# Shared constants
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"
MAX_REDIRECTS = 5  # Limit redirects to prevent DoS attacks

# Public SearXNG instances (fallback list)
SEARXNG_INSTANCES = [
    "https://searx.be",
    "https://search.bus-hit.me",
    "https://search.rowie.at",
    "https://searx.fmac.xyz",
    "https://search.sapti.me",
]

# Search usage tracking file
USAGE_FILE = Path.home() / ".nanobot" / "search_usage.json"


def _load_usage() -> dict:
    """Load search usage data from file."""
    if USAGE_FILE.exists():
        try:
            with open(USAGE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"brave": {"month": "", "count": 0}}


def _save_usage(data: dict) -> None:
    """Save search usage data to file."""
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USAGE_FILE, "w") as f:
        json.dump(data, f)


def _get_brave_usage_this_month() -> int:
    """Get Brave search count for current month."""
    data = _load_usage()
    current_month = datetime.now().strftime("%Y-%m")
    if data.get("brave", {}).get("month") == current_month:
        return data["brave"]["count"]
    return 0


def _increment_brave_usage() -> None:
    """Increment Brave search count for current month."""
    data = _load_usage()
    current_month = datetime.now().strftime("%Y-%m")
    if data.get("brave", {}).get("month") != current_month:
        data["brave"] = {"month": current_month, "count": 0}
    data["brave"]["count"] += 1
    _save_usage(data)
    logger.info(f"Brave search usage: {data['brave']['count']} this month")


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<script[\s\S]*?</script>', '', text, flags=re.I)
    text = re.sub(r'<style[\s\S]*?</style>', '', text, flags=re.I)
    text = re.sub(r'<[^>]+>', '', text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL: must be http(s) with valid domain."""
    try:
        p = urlparse(url)
        if p.scheme not in ('http', 'https'):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


class SearXNGSearchTool(Tool):
    """Search the web using SearXNG (free, no API key required)."""
    
    name = "web_search"
    description = "Search the web using SearXNG. Returns titles, URLs, and snippets. No API key required."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10}
        },
        "required": ["query"]
    }
    
    def __init__(self, instance_url: str | None = None, max_results: int = 5):
        self.instance_url = instance_url or os.environ.get("SEARXNG_URL", "")
        self.max_results = max_results
        # Shuffle instances for load balancing if using default list
        self._instances = SEARXNG_INSTANCES.copy()
        random.shuffle(self._instances)
    
    async def _search_instance(self, client: httpx.AsyncClient, instance: str, query: str, n: int) -> list[dict]:
        """Search a single SearXNG instance."""
        try:
            r = await client.get(
                f"{instance}/search",
                params={"q": query, "format": "json", "engines": "google,bing,duckduckgo"},
                headers={"User-Agent": USER_AGENT},
                timeout=15.0
            )
            r.raise_for_status()
            data = r.json()
            return data.get("results", [])[:n]
        except Exception:
            return []
    
    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        n = min(max(count or self.max_results, 1), 10)
        
        async with httpx.AsyncClient() as client:
            results = []
            
            # Try configured instance first
            if self.instance_url:
                results = await self._search_instance(client, self.instance_url, query, n)
            
            # Fallback to public instances
            if not results:
                for instance in self._instances[:3]:  # Try up to 3 instances
                    results = await self._search_instance(client, instance, query, n)
                    if results:
                        break
            
            if not results:
                return f"No results for: {query}"
            
            lines = [f"Results for: {query}\n"]
            for i, item in enumerate(results[:n], 1):
                lines.append(f"{i}. {item.get('title', 'No title')}\n   {item.get('url', '')}")
                if content := item.get("content"):
                    lines.append(f"   {content[:200]}")
            return "\n".join(lines)


class WebSearchTool(Tool):
    """Search the web. Routes to Brave Search (if configured with quota) or SearXNG (free fallback)."""
    
    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10}
        },
        "required": ["query"]
    }
    
    def __init__(self, api_key: str | None = None, max_results: int = 5, searxng_url: str | None = None,
                 brave_monthly_limit: int = 2000):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        self.searxng_url = searxng_url or os.environ.get("SEARXNG_URL", "")
        self.max_results = max_results
        self.brave_monthly_limit = int(os.environ.get("BRAVE_MONTHLY_LIMIT", brave_monthly_limit))
        self._searxng = SearXNGSearchTool(self.searxng_url, max_results)
    
    def _can_use_brave(self) -> bool:
        """Check if Brave Search can be used (has key and quota available)."""
        if not self.api_key:
            return False
        current_usage = _get_brave_usage_this_month()
        return current_usage < self.brave_monthly_limit
    
    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        n = min(max(count or self.max_results, 1), 10)
        
        # Decision: Brave (if available with quota) or SearXNG
        if self._can_use_brave():
            logger.info(f"[web_search] Using Brave Search (quota: {_get_brave_usage_this_month()}/{self.brave_monthly_limit})")
            try:
                result = await self._brave_search(query, n)
                _increment_brave_usage()
                return result
            except Exception as e:
                logger.warning(f"[web_search] Brave failed: {e}, falling back to SearXNG")
                # Fall through to SearXNG
        else:
            reason = "no API key" if not self.api_key else f"quota exceeded ({_get_brave_usage_this_month()}/{self.brave_monthly_limit})"
            logger.info(f"[web_search] Using SearXNG (Brave: {reason})")
        
        # Use SearXNG
        try:
            return await self._searxng.execute(query, count, **kwargs)
        except Exception as e:
            return f"Error: Search failed - {e}"
    
    async def _brave_search(self, query: str, n: int) -> str:
        """Execute search via Brave Search API."""
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": n},
                headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
                timeout=10.0
            )
            r.raise_for_status()
        
        results = r.json().get("web", {}).get("results", [])
        if not results:
            return f"No results for: {query}"
        
        lines = [f"Results for: {query}\n"]
        for i, item in enumerate(results[:n], 1):
            lines.append(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}")
            if desc := item.get("description"):
                lines.append(f"   {desc}")
        return "\n".join(lines)


class WebFetchTool(Tool):
    """Fetch and extract content from a URL using Readability."""
    
    name = "web_fetch"
    description = "Fetch URL and extract readable content (HTML → markdown/text)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100}
        },
        "required": ["url"]
    }
    
    def __init__(self, max_chars: int = 50000):
        self.max_chars = max_chars
    
    async def execute(self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any) -> str:
        from readability import Document

        max_chars = maxChars or self.max_chars

        # Validate URL before fetching
        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            return json.dumps({"error": f"URL validation failed: {error_msg}", "url": url})

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=MAX_REDIRECTS,
                timeout=30.0
            ) as client:
                r = await client.get(url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()
            
            ctype = r.headers.get("content-type", "")
            
            # JSON
            if "application/json" in ctype:
                text, extractor = json.dumps(r.json(), indent=2), "json"
            # HTML
            elif "text/html" in ctype or r.text[:256].lower().startswith(("<!doctype", "<html")):
                doc = Document(r.text)
                content = self._to_markdown(doc.summary()) if extractMode == "markdown" else _strip_tags(doc.summary())
                text = f"# {doc.title()}\n\n{content}" if doc.title() else content
                extractor = "readability"
            else:
                text, extractor = r.text, "raw"
            
            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]
            
            return json.dumps({"url": url, "finalUrl": str(r.url), "status": r.status_code,
                              "extractor": extractor, "truncated": truncated, "length": len(text), "text": text})
        except Exception as e:
            return json.dumps({"error": str(e), "url": url})
    
    def _to_markdown(self, html: str) -> str:
        """Convert HTML to markdown."""
        # Convert links, headings, lists before stripping tags
        text = re.sub(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
                      lambda m: f'[{_strip_tags(m[2])}]({m[1]})', html, flags=re.I)
        text = re.sub(r'<h([1-6])[^>]*>([\s\S]*?)</h\1>',
                      lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n', text, flags=re.I)
        text = re.sub(r'<li[^>]*>([\s\S]*?)</li>', lambda m: f'\n- {_strip_tags(m[1])}', text, flags=re.I)
        text = re.sub(r'</(p|div|section|article)>', '\n\n', text, flags=re.I)
        text = re.sub(r'<(br|hr)\s*/?>', '\n', text, flags=re.I)
        return _normalize(_strip_tags(text))
