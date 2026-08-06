"""
Dense semantic retrieval for the legal RAG query workflow.

Workflow
========

```text
rewritten_query (or original_query)
        |
        v
TEI / BGE query embedding
        |
        v
Milvus cosine search over legal child vectors
        |
        | chunk IDs + vector scores + filter metadata
        v
Elasticsearch multi-get by chunk ID
        |
        | authoritative legal text and headings
        v
normalized semantic hits
        |
        v
state["semantic_searched_chunks"]
```

Milvus intentionally does not duplicate authoritative legal text. Every vector
hit is therefore enriched from Elasticsearch before it reaches RRF or rerank.
The node returns only its branch-specific state key so it can run in parallel
with ``node_es_keywords_search`` during hybrid retrieval.
"""

from __future__ import annotations

import sys
from typing import Any

from app.clients.embedding_utils import embedding_client_manager
from app.clients.es_utils import es_client_manager
from app.clients.milvus_utils import milvus_client_manager
from app.config.embedding_config import embedding_config
from app.config.es_config import es_config
from app.config.meta_config import meta_config
from app.config.milvus_config import milvus_config
from app.core.logger import logger
from app.repositories.es.es_repository import ESRepository
from app.repositories.milvus.milvus_repository import MilvusRepository
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


SEMANTIC_SEARCH_LIMIT = 10

ACTIVE_CHILD_FILTER = (
    'record_type in ["child", "parent_child"] '
    'and status in ["in_force", "amended"]'
)


def _query_text(state: QueryGraphState) -> str:
    """Return the standalone query preferred by every retrieval branch."""
    query = str(
        state.get("rewritten_query")
        or state.get("original_query")
        or ""
    ).strip()
    if not query:
        raise ValueError("Semantic search requires a non-empty query.")
    return query


def _validate_vector_contract() -> None:
    """Fail early when the query model and Milvus index dimensions differ."""
    vector_field = meta_config.get_milvus_field("dense_vector")
    if vector_field.dimension != embedding_config.dimension:
        raise ValueError(
            "Embedding/Milvus dimension mismatch: "
            f"embedding={embedding_config.dimension}, "
            f"milvus={vector_field.dimension}."
        )


def _normalize_semantic_hits(
    milvus_hits: list[dict[str, Any]],
    es_records: dict[str, Any],
) -> list[dict[str, Any]]:
    """Combine ranked Milvus metadata with authoritative ES documents."""
    normalized: list[dict[str, Any]] = []
    for hit in milvus_hits:
        chunk_id = str(hit.get("chunk_id") or "").strip()
        record = es_records.get(chunk_id)
        if record is None:
            logger.warning(
                f"Skipping semantic hit missing from Elasticsearch: {chunk_id}"
            )
            continue

        document = record.to_document()
        score = float(hit["score"])
        title = (
            document.get("article_title")
            or document.get("chapter_title")
            or document.get("law_title")
            or ""
        )
        entity = {
            **document,
            # These aliases preserve compatibility with the current reranker,
            # while ``text`` remains the canonical authoritative field.
            "content": document["text"],
            "title": title,
        }
        normalized.append(
            {
                "id": chunk_id,
                "distance": score,
                "score": score,
                "retrieval_source": "semantic",
                "entity": entity,
            }
        )
    return normalized


async def node_embedding_semantic_search(
    state: QueryGraphState,
) -> dict[str, list[dict[str, Any]]]:
    """
    Embed one query, search Milvus, and enrich hits from Elasticsearch.

    The six numbered comments describe the complete node lifecycle. External
    clients are process-level managers and are initialized lazily, so repeated
    requests reuse their connection pools.
    """
    # Step 1: Validate graph inputs and register branch progress. This branch
    # may execute concurrently with Elasticsearch BM25 search.
    function_name = sys._getframe().f_code.co_name
    session_id = str(state.get("session_id") or "unknown")
    is_stream = bool(state.get("is_stream", False))
    query = _query_text(state)
    logger.info(
        f">>> [{function_name}] started: session_id={session_id}, query={query}"
    )
    add_running_task(session_id, function_name, is_stream)

    try:
        # Step 2: Validate the model/index vector contract, initialize the TEI
        # adapter, and generate exactly one BGE query embedding asynchronously.
        _validate_vector_contract()
        embedding_client_manager.init()
        embedding_client = embedding_client_manager.get_client()
        # embedd query
        query_vector = await embedding_client.aembed_query(query)

        # Step 3: Initialize Milvus and run cosine search only over searchable,
        # active child records. Parent records have no vectors by design.
        milvus_client_manager.init()
        milvus_repository = MilvusRepository(
            client=milvus_client_manager.get_client(),
            collection_name=milvus_config.chunks_collection,
        )
        milvus_hits = await milvus_repository.search(
            query_vector,
            limit=SEMANTIC_SEARCH_LIMIT,
            filter_expression=ACTIVE_CHILD_FILTER,
        )

        # Step 4: Fetch the corresponding authoritative text records from ES.
        # A single mget preserves efficiency and avoids one request per hit.
        es_client_manager.init()
        es_repository = ESRepository(
            client=es_client_manager.get_client(),
            index_name=es_config.index_name,
        )
        es_records = await es_repository.get_by_chunk_ids(
            [hit["chunk_id"] for hit in milvus_hits]
        )

        # Step 5: Normalize both storage representations into the common hit
        # shape expected by RRF and reranking, preserving Milvus rank order.
        semantic_hits = _normalize_semantic_hits(milvus_hits, es_records)

        # Step 6: Mark this parallel branch complete and return only its own
        # state key to avoid concurrent writes with the BM25 branch.
        add_done_task(session_id, function_name, is_stream)
        logger.info(
            f">>> [{function_name}] finished: session_id={session_id}, "
            f"milvus_hits={len(milvus_hits)}, enriched_hits={len(semantic_hits)}"
        )
        return {"semantic_searched_chunks": semantic_hits}
    except Exception:
        logger.exception(
            f">>> [{function_name}] failed: session_id={session_id}"
        )
        raise
