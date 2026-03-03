"""Web tools: web_search and web_fetch."""

import html
import json
import os
import random
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from nanobot.agent.tools.base import Tool

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
    """Search the web. Uses SearXNG (free) or Brave Search API (if configured)."""
    
    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets. Uses SearXNG by default (no API key needed)."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10}
        },
        "required": ["query"]
    }
    
    def __init__(self, api_key: str | None = None, max_results: int = 5, searxng_url: str | None = None):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        self.searxng_url = searxng_url or os.environ.get("SEARXNG_URL", "")
        self.max_results = max_results
        # SearXNG fallback instances
        self._searxng = SearXNGSearchTool(self.searxng_url, max_results)
    
    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        # Prefer SearXNG (free, no API key needed)
        # Only use Brave if explicitly configured and SearXNG fails
        try:
            return await self._searxng.execute(query, count, **kwargs)
        except Exception as e:
            # Fallback to Brave API if configured
            if self.api_key:
                return await self._brave_search(query, count)
            return f"Error: Search failed - {e}"
    
    async def _brave_search(self, query: str, count: int | None = None) -> str:
        """Fallback to Brave Search API."""
        try:
            n = min(max(count or self.max_results, 1), 10)
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
        except Exception as e:
            return f"Error: {e}"


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
