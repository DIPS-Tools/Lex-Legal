"""
Tavily web-search fallback through the Model Context Protocol (MCP).

Why MCP instead of the Tavily Python client?
============================================

The application no longer calls ``TavilyClient.search`` directly. It acts as
an MCP client and calls a tool exposed by Tavily's remote MCP server. MCP keeps
the application independent from one provider-specific Python SDK:

```text
LangGraph node
    |
    |  Streamable HTTP transport
    v
Tavily MCP server
    |
    |  tools/list  -> discover available tools
    |  tools/call  -> invoke tavily_search
    v
Tavily Search API
```

The protocol sequence implemented below is:

```text
1. Open Streamable HTTP transport
2. Create MCP ClientSession
3. Perform the MCP initialize handshake
4. Request tools/list
5. Verify tavily_search is available
6. Send tools/call with structured arguments
7. Parse MCP content blocks
8. Normalize results for reranking and citations
9. Close the session and HTTP transport automatically
```

This module uses the official ``mcp`` Python SDK directly. The project's
currently locked ``langchain-mcp-adapters`` and ``mcp`` versions are not API
compatible, while the direct SDK transport is available and makes each MCP
step explicit for learning.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult

from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


load_dotenv()

DEFAULT_TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"
DEFAULT_MAX_RESULTS = 5
DEFAULT_SEARCH_DEPTH = "advanced"
TAVILY_SEARCH_TOOL_NAMES = ("tavily_search", "tavily-search")


class TavilyMCPError(RuntimeError):
    """Raised when the MCP server cannot provide usable search results."""


def _configured_domains() -> list[str]:
    """
    Read an optional comma-separated authoritative-domain allowlist.

    For this EU-law project, a useful production value is:

    ``TAVILY_INCLUDE_DOMAINS=eur-lex.europa.eu``

    It is not hard-coded because the same query workflow may later answer
    questions requiring other official sources.
    """
    raw_domains = os.getenv("TAVILY_INCLUDE_DOMAINS", "")
    return [
        domain.strip()
        for domain in raw_domains.split(",")
        if domain.strip()
    ]


def _mcp_connection_settings() -> tuple[str, dict[str, str], float]:
    """
    Validate configuration and construct secret-safe connection settings.

    The API key is sent in an Authorization header rather than placed in the
    URL. URLs are commonly logged by proxies and HTTP clients, while request
    authorization headers are normally redacted.
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise TavilyMCPError(
            "TAVILY_API_KEY is required for Tavily MCP search."
        )

    endpoint = os.getenv(
        "TAVILY_MCP_URL",
        DEFAULT_TAVILY_MCP_URL,
    ).strip()
    if not endpoint:
        raise TavilyMCPError("TAVILY_MCP_URL must not be empty.")

    try:
        timeout = float(os.getenv("TAVILY_MCP_TIMEOUT", "30"))
    except ValueError as exc:
        raise TavilyMCPError(
            "TAVILY_MCP_TIMEOUT must be a number."
        ) from exc
    if timeout <= 0:
        raise TavilyMCPError(
            "TAVILY_MCP_TIMEOUT must be greater than zero."
        )

    return (
        endpoint,
        {"Authorization": f"Bearer {api_key}"},
        timeout,
    )


def _select_search_tool(tool_names: Sequence[str]) -> str:
    """
    Select Tavily's search tool after MCP discovery.

    Current Tavily servers expose ``tavily_search``. Supporting the historical
    hyphenated spelling keeps this client tolerant of older deployments.
    """
    available_names = set(tool_names)
    for preferred_name in TAVILY_SEARCH_TOOL_NAMES:
        if preferred_name in available_names:
            return preferred_name
    raise TavilyMCPError(
        "Tavily MCP server does not expose a supported search tool. "
        f"Available tools: {sorted(available_names)}"
    )


def _valid_web_url(value: Any) -> bool:
    """Return whether a value is an absolute HTTP(S) URL."""
    if not isinstance(value, str):
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _normalize_result_items(
    result_items: Any,
) -> list[dict[str, Any]]:
    """
    Convert structured Tavily records into the query pipeline's common shape.

    Web records use ``text`` rather than Tavily's ``content`` so the existing
    cross-encoder reranker can score local and network evidence uniformly.
    URLs are required because generated legal answers must be able to cite the
    external evidence.
    """
    if not isinstance(result_items, list):
        return []

    documents: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for item in result_items:
        if not isinstance(item, Mapping):
            continue

        url = item.get("url")
        if not _valid_web_url(url):
            continue
        normalized_url = str(url).strip()
        if normalized_url in seen_urls:
            continue

        text = item.get("content") or item.get("raw_content")
        if not isinstance(text, str) or not text.strip():
            continue

        score = item.get("score")
        normalized_score: float | None = None
        if (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score))
        ):
            normalized_score = float(score)

        document: dict[str, Any] = {
            "chunk_id": "",
            "parent_id": "",
            "text": text.strip(),
            "title": str(item.get("title") or normalized_url).strip(),
            "url": normalized_url,
            "source": "web",
            "retrieval_method": "tavily_mcp",
        }
        if normalized_score is not None:
            document["retrieval_score"] = normalized_score
        if item.get("published_date"):
            document["published_date"] = str(item["published_date"])

        seen_urls.add(normalized_url)
        documents.append(document)
    return documents


_FORMATTED_RESULT_RE = re.compile(
    r"(?:^|\n)Title:\s*(?P<title>.*?)\n"
    r"URL:\s*(?P<url>\S+)\n"
    r"Content:\s*(?P<content>.*?)"
    r"(?=\nTitle:\s*|\nImages:\s*|\Z)",
    flags=re.DOTALL,
)


def _parse_formatted_tavily_text(text: str) -> list[dict[str, Any]]:
    """
    Parse the human-readable content returned by Tavily's MCP server.

    Tavily currently formats each hit as ``Title``, ``URL``, and ``Content``
    lines. MCP permits servers to return text instead of structured JSON, so a
    client must understand the advertised content representation.
    """
    items = [
        {
            "title": match.group("title").strip(),
            "url": match.group("url").strip(),
            "content": match.group("content").strip(),
        }
        for match in _FORMATTED_RESULT_RE.finditer(text)
    ]
    return _normalize_result_items(items)


def normalize_mcp_search_result(
    result: CallToolResult,
) -> list[dict[str, Any]]:
    """
    Extract normalized documents from an MCP ``tools/call`` response.

    MCP supports both structured content and typed content blocks. Structured
    results are preferred; JSON text and Tavily's formatted text are supported
    as fallbacks.
    """
    if result.isError:
        error_text = " ".join(
            content.text
            for content in result.content
            if getattr(content, "type", None) == "text"
        ).strip()
        raise TavilyMCPError(
            f"Tavily MCP tool returned an error: "
            f"{error_text or 'unknown MCP tool error'}"
        )

    structured_content = result.structuredContent
    if isinstance(structured_content, Mapping):
        documents = _normalize_result_items(
            structured_content.get("results")
        )
        if documents:
            return documents

    text_blocks = [
        content.text
        for content in result.content
        if getattr(content, "type", None) == "text"
        and isinstance(getattr(content, "text", None), str)
    ]
    for text in text_blocks:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping):
            documents = _normalize_result_items(payload.get("results"))
            if documents:
                return documents

        documents = _parse_formatted_tavily_text(text)
        if documents:
            return documents

    raise TavilyMCPError(
        "Tavily MCP search returned no citation-ready results."
    )


async def call_tavily_search_mcp(
    query: str,
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    search_depth: str = DEFAULT_SEARCH_DEPTH,
    include_domains: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Discover and call Tavily search through an MCP Streamable HTTP session.

    ``ClientSession.initialize`` performs the protocol handshake.
    ``list_tools`` avoids assuming that a remote server still exposes a
    particular capability. ``call_tool`` sends a typed MCP ``tools/call``
    request after discovery.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The Tavily MCP search query must not be empty.")
    if not 5 <= max_results <= 20:
        raise ValueError("max_results must be between 5 and 20.")
    if search_depth not in {"basic", "advanced", "fast", "ultra-fast"}:
        raise ValueError(f"Unsupported Tavily search_depth: {search_depth!r}")

    endpoint, headers, timeout = _mcp_connection_settings()
    arguments: dict[str, Any] = {
        "query": query.strip(),
        # Tavily's current MCP schema supports "general" here. "law" is not a
        # valid topic; legal focus should be expressed through query wording and
        # optional authoritative-domain filtering.
        "topic": "general",
        "search_depth": search_depth,
        "max_results": max_results,
        "include_raw_content": False,
        "include_images": False,
    }
    normalized_domains = [
        domain.strip()
        for domain in (include_domains or [])
        if isinstance(domain, str) and domain.strip()
    ]
    if normalized_domains:
        arguments["include_domains"] = normalized_domains

    logger.info(
        f"Connecting to Tavily MCP server: "
        f"endpoint={endpoint}, max_results={max_results}, "
        f"search_depth={search_depth}, "
        f"include_domains={normalized_domains}"
    )

    # Both context managers are important:
    # - streamablehttp_client owns the HTTP/SSE transport streams;
    # - ClientSession owns MCP request IDs, initialization, and messages.
    async with streamablehttp_client(
        endpoint,
        headers=headers,
        timeout=timeout,
        sse_read_timeout=timeout,
    ) as (read_stream, write_stream, _):
        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:
            await session.initialize()
            listed_tools = await session.list_tools()
            tool_name = _select_search_tool(
                [tool.name for tool in listed_tools.tools]
            )
            logger.info(
                f"Tavily MCP tools discovered: "
                f"{[tool.name for tool in listed_tools.tools]}; "
                f"selected={tool_name}"
            )
            result = await session.call_tool(
                name=tool_name,
                arguments=arguments,
            )

    return normalize_mcp_search_result(result)


async def node_network_search(
    state: QueryGraphState,
) -> dict[str, Any]:
    """
    Run one MCP web-search fallback and return a partial graph-state update.

    The rewritten query is preferred because earlier query-planning nodes may
    have resolved references from conversation history. The fallback flag is
    always set on success, preventing the query graph from repeatedly searching
    the network.
    """
    function_name = sys._getframe().f_code.co_name
    session_id = state.get("session_id", "unknown")
    query = (
        state.get("rewritten_query")
        or state.get("original_query")
        or ""
    ).strip()

    logger.info(
        f">>> [{function_name}] Started. "
        f"session_id={session_id}, query={query!r}"
    )
    add_running_task(
        session_id,
        function_name,
        state.get("is_stream", False),
    )

    try:
        documents = await call_tavily_search_mcp(
            query,
            include_domains=_configured_domains(),
        )
    except Exception:
        logger.exception(
            f"Tavily MCP network search failed. "
            f"session_id={session_id}"
        )
        raise

    add_done_task(
        session_id,
        function_name,
        state.get("is_stream", False),
    )
    logger.info(
        f">>> [{function_name}] Finished. "
        f"session_id={session_id}, documents={len(documents)}"
    )
    # Do not use ``return state.update(...)``: dict.update mutates the input and
    # returns None. LangGraph nodes should return a partial update dictionary.
    return {
        "network_searched_docs": documents,
        "network_fallback_used": True,
    }


async def test_tavily_mcp_search() -> None:
    """Manual integration test that calls the configured remote MCP server."""
    state = {
        "session_id": "test_tavily_mcp",
        "original_query": (
            "What is the current consolidated text of GDPR Article 6?"
        ),
        "rewritten_query": "",
        "is_stream": False,
    }
    result = await node_network_search(state)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_tavily_mcp_search())
