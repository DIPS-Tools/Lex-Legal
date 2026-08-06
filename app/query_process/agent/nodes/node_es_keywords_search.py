"""
Elasticsearch BM25 retrieval for the legal RAG query workflow.

Workflow
========

```text
rewritten_query + extracted legal key terms
        |
        v
YAML-configured Elasticsearch multi_match
        |
        | law/article headings, citations, key_terms, legal text
        v
BM25-ranked ESRecordInfo results
        |
        v
normalized keyword hits
        |
        v
state["keywords_searched_chunks"]
```

This node performs lexical retrieval only. It does not classify intent and it
does not compare BM25 scores with Milvus cosine scores. Hybrid score fusion is
rank-based and occurs later in ``node_rrf``.
"""

from __future__ import annotations

import sys
from typing import Any

from app.clients.es_utils import es_client_manager
from app.config.es_config import es_config
from app.core.logger import logger
from app.entities.es_record import ESSearchResult
from app.repositories.es.es_repository import ESRepository
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


KEYWORD_SEARCH_LIMIT = 10


def _build_bm25_query(state: QueryGraphState) -> str:
    """Combine the standalone query with unique LLM-extracted legal terms."""
    query = str(
        state.get("rewritten_query")
        or state.get("original_query")
        or ""
    ).strip()
    if not query:
        raise ValueError("Elasticsearch keyword search requires a query.")

    parts = [query]
    seen = {query.casefold()}
    for value in state.get("query_key_terms", []) or []:
        if not isinstance(value, str):
            continue
        term = " ".join(value.split())
        if not term or term.casefold() in seen:
            continue
        seen.add(term.casefold())
        parts.append(term)
    return " ".join(parts)


def _normalize_keyword_hits(
    search_results: list[ESSearchResult],
) -> list[dict[str, Any]]:
    """Convert typed Elasticsearch results to the shared retrieval shape."""
    normalized: list[dict[str, Any]] = []
    for result in search_results:
        document = result.record.to_document()
        score = float(result.score or 0.0)
        title = (
            document.get("article_title")
            or document.get("chapter_title")
            or document.get("law_title")
            or ""
        )
        chunk_id = document["chunk_id"]
        normalized.append(
            {
                "id": chunk_id,
                "distance": score,
                "score": score,
                "retrieval_source": "bm25",
                "entity": {
                    **document,
                    "content": document["text"],
                    "title": title,
                },
            }
        )
    return normalized


async def node_es_keywords_search(
    state: QueryGraphState,
) -> dict[str, list[dict[str, Any]]]:
    """Search legal text and metadata using Elasticsearch BM25 ranking."""
    # Step 1: Validate query state and register progress. The node writes only
    # the keyword branch key, making hybrid parallel execution safe.
    function_name = sys._getframe().f_code.co_name
    session_id = str(state.get("session_id") or "unknown")
    is_stream = bool(state.get("is_stream", False))
    bm25_query = _build_bm25_query(state)
    logger.info(
        f">>> [{function_name}] started: "
        f"session_id={session_id}, query={bm25_query}"
    )
    add_running_task(session_id, function_name, is_stream)

    try:
        # Step 2: Initialize the shared asynchronous Elasticsearch connection.
        # Client creation is lazy and does not create a new pool per request.
        es_client_manager.init()
        es_repository = ESRepository(
            client=es_client_manager.get_client(),
            index_name=es_config.index_name,
        )

        # Step 3: Execute one multi-field BM25 query. Field boosts and the list
        # of searchable legal metadata come from configs/meta_config.yml.
        search_results = await es_repository.search(
            bm25_query,
            limit=KEYWORD_SEARCH_LIMIT,
        )

        # Step 4: Convert Pydantic ES records to a common retrieval shape while
        # retaining authoritative text, legal metadata, and original BM25 rank.
        keyword_hits = _normalize_keyword_hits(search_results)

        # Step 5: Log result counts rather than complete legal text, avoiding
        # oversized production logs while retaining useful diagnostics.
        logger.info(
            f"BM25 legal search completed: session_id={session_id}, "
            f"results={len(keyword_hits)}"
        )

        # Step 6: Complete branch tracking and return only this branch's state
        # key so the semantic branch can update state concurrently.
        add_done_task(session_id, function_name, is_stream)
        return {"keywords_searched_chunks": keyword_hits}
    except Exception:
        logger.exception(
            f">>> [{function_name}] failed: session_id={session_id}"
        )
        raise
