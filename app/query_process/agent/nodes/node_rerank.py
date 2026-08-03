"""
Cross-encoder reranking for local and network legal evidence.

Workflow
========

```text
rrf_chunks (hybrid or single-search local evidence) --.
                                                       +--> canonical documents
network_searched_docs (optional fallback evidence) ---'          |
                                                                  v
                                                    (query, document) pairs
                                                                  |
                                                                  v
                                               BGE cross-encoder scores
                                                                  |
                                                                  v
                                                sort + bounded gap cutoff
                                                                  |
                                                                  v
                                                     state["reranked_docs"]
```

The reranker runs after RRF for hybrid retrieval. For semantic-only or
BM25-only retrieval, ``node_search_join`` places that branch directly in
``rrf_chunks``, so this node has one stable local-input field in every route.

``FlagReranker.compute_score`` is synchronous and CPU/GPU intensive. The node
runs it in a worker thread so FastAPI's event loop can continue serving Mongo,
Elasticsearch, Milvus, and SSE operations. A lock prevents concurrent requests
from loading or executing the same local model simultaneously.
"""

from __future__ import annotations

import asyncio
import math
import sys
from collections.abc import Sequence
from numbers import Real
from typing import Any

from app.clients.reranker_utils import get_reranker_model
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


# Score no more than this many final records in the answer context. RRF already
# limits hybrid local candidates, while network fallback may add more records.
RERANK_MAX_TOP_K = 10

# Keep at least three records when available before applying a score-gap cutoff.
# Legal answers often need neighboring provisions or more than one authority.
RERANK_MIN_TOP_K = 3

# After the minimum result count, stop when neighboring normalized scores show
# either a large absolute drop or a large drop relative to the higher score.
RERANK_GAP_ABS = 0.25
RERANK_GAP_RATIO = 0.35

# A single in-process FlagReranker instance can be large. Serializing access
# avoids duplicate lazy loading and excessive CPU/GPU memory pressure.
_reranker_lock = asyncio.Lock()


def _required_query(state: QueryGraphState) -> str:
    """Return the standalone query used to score every candidate passage."""
    query = str(
        state.get("rewritten_query")
        or state.get("original_query")
        or ""
    ).strip()
    if not query:
        raise ValueError("Reranking requires a non-empty query.")
    return query


def _local_document(chunk: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten one normalized RRF/local hit into answer evidence."""
    entity = chunk.get("entity")
    if not isinstance(entity, dict):
        logger.warning("Skipping local rerank candidate without an entity.")
        return None

    text = str(entity.get("text") or entity.get("content") or "").strip()
    if not text:
        logger.warning(
            f"Skipping local rerank candidate with empty text: "
            f"{chunk.get('id', '<unknown>')}"
        )
        return None

    chunk_id = str(
        chunk.get("id")
        or entity.get("chunk_id")
        or chunk.get("chunk_id")
        or ""
    ).strip()
    if not chunk_id:
        raise ValueError("Local rerank candidate is missing chunk_id.")

    # Preserve legal metadata from Elasticsearch while replacing retrieval
    # aliases with the flat schema consumed by answer generation and citations.
    document = {
        **entity,
        "chunk_id": chunk_id,
        "text": text,
        "title": str(
            entity.get("title")
            or entity.get("article_title")
            or entity.get("chapter_title")
            or entity.get("law_title")
            or ""
        ).strip(),
        "source": "local",
        "url": str(entity.get("url") or "").strip(),
    }
    if "rrf_score" in chunk:
        document["rrf_score"] = chunk["rrf_score"]
    retrieval_score = chunk.get("score", chunk.get("distance"))
    if retrieval_score is not None:
        document["retrieval_score"] = retrieval_score
    return document


def _network_document(document: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and copy one Tavily/MCP result for cross-encoder scoring."""
    text = str(
        document.get("text")
        or document.get("snippet")
        or document.get("content")
        or ""
    ).strip()
    if not text:
        logger.warning("Skipping network rerank candidate with empty text.")
        return None
    return {
        **document,
        "chunk_id": str(document.get("chunk_id") or ""),
        "text": text,
        "title": str(document.get("title") or "").strip(),
        "source": "web",
        "url": str(document.get("url") or "").strip(),
    }


def step_1_merge_local_rrf_and_network_search(state: QueryGraphState) -> list[dict[str, Any]]:
    """
    Normalize and deduplicate local and optional network evidence.

    Local records use ``chunk_id`` as their identity. Web records use URL,
    falling back to title plus text when a provider omits the URL. Preserving
    the first occurrence keeps the existing RRF/Tavily order before reranking.
    """
    local_chunks = state.get("rrf_chunks", [])
    network_documents = state.get("network_searched_docs", [])
    if not isinstance(local_chunks, list):
        raise TypeError("rrf_chunks must be a list.")
    if not isinstance(network_documents, list):
        raise TypeError("network_searched_docs must be a list.")

    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    # Step 1: Flatten local RRF records and preserve authoritative ES metadata.
    for chunk in local_chunks:
        if not isinstance(chunk, dict):
            raise TypeError("Every rrf_chunks item must be a dictionary.")
        document = _local_document(chunk)
        if document is None:
            continue
        identity = ("local", document["chunk_id"])
        if identity not in seen:
            seen.add(identity)
            merged.append(document)

    # Step 2: Normalize network records and prevent duplicate URLs from being
    # scored repeatedly after the network fallback.
    for raw_document in network_documents:
        if not isinstance(raw_document, dict):
            raise TypeError(
                "Every network_searched_docs item must be a dictionary."
            )
        document = _network_document(raw_document)
        if document is None:
            continue
        web_identity = document["url"] or (
            f"{document['title']}\n{document['text']}"
        )
        identity = ("web", web_identity)
        if identity not in seen:
            seen.add(identity)
            merged.append(document)

    logger.info(
        f"Prepared reranker candidates: local={len(local_chunks)}, "
        f"network={len(network_documents)}, unique_usable={len(merged)}"
    )
    return merged


def _normalize_scores(raw_scores: Any, expected_count: int) -> list[float]:
    """Convert FlagEmbedding output to a validated list of finite floats."""
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    if expected_count == 1 and isinstance(raw_scores, Real):
        raw_scores = [raw_scores]

    if not isinstance(raw_scores, (list, tuple)):
        raise TypeError("Reranker scores must be a list, tuple, or scalar.")
    if len(raw_scores) != expected_count:
        raise ValueError(
            "Reranker score count does not match candidate count: "
            f"expected={expected_count}, received={len(raw_scores)}."
        )

    scores: list[float] = []
    for raw_score in raw_scores:
        if (
            not isinstance(raw_score, Real)
            or isinstance(raw_score, bool)
        ):
            raise TypeError("Every reranker score must be numeric.")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("Reranker returned a non-finite score.")
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                "Normalized reranker score must be between 0 and 1."
            )
        scores.append(score)
    return scores


async def step_2_rerank_doc_list(
    documents: Sequence[dict[str, Any]],
    state: QueryGraphState,
) -> list[dict[str, Any]]:
    """Score every query-document pair and return descending copies."""
    if not documents:
        return []
    query = _required_query(state)
    pairs = [[query, document["text"]] for document in documents]

    # Loading and inference are both synchronous in FlagEmbedding. Execute
    # them together in a worker thread while serializing access to the model.
    def compute_scores() -> Any:
        reranker = get_reranker_model()
        return reranker.compute_score(pairs, normalize=True)

    async with _reranker_lock:
        raw_scores = await asyncio.to_thread(compute_scores)
    scores = _normalize_scores(raw_scores, len(documents))

    # Copy each candidate before adding the cross-encoder score. Retrieval
    # lists in state remain unchanged for diagnostics and reproducibility.
    scored_documents = [
        {
            **document,
            "score": score,
        }
        for document, score in zip(documents, scores, strict=True)
    ]
    scored_documents.sort(key=lambda document: document["score"], reverse=True)
    return scored_documents


def step_3_topk_and_gap(
    reranked_documents: Sequence[dict[str, Any]],
    *,
    max_top_k: int = RERANK_MAX_TOP_K,
    min_top_k: int = RERANK_MIN_TOP_K,
    gap_abs: float = RERANK_GAP_ABS,
    gap_ratio: float = RERANK_GAP_RATIO,
) -> list[dict[str, Any]]:
    """Keep a bounded top set and stop after a meaningful score cliff."""
    if max_top_k <= 0:
        raise ValueError("max_top_k must be greater than zero.")
    if min_top_k <= 0 or min_top_k > max_top_k:
        raise ValueError("min_top_k must be between 1 and max_top_k.")
    if gap_abs < 0 or gap_ratio < 0:
        raise ValueError("Gap thresholds must not be negative.")

    retained_count = min(max_top_k, len(reranked_documents))
    if retained_count > min_top_k:
        # Begin comparisons only after the minimum evidence count. For example,
        # index=2 compares the third and fourth results when min_top_k is 3.
        for index in range(min_top_k - 1, retained_count - 1):
            current_score = float(reranked_documents[index]["score"])
            next_score = float(reranked_documents[index + 1]["score"])
            absolute_gap = current_score - next_score
            relative_gap = absolute_gap / max(abs(current_score), 1e-12)
            if absolute_gap >= gap_abs or relative_gap >= gap_ratio:
                retained_count = index + 1
                break

    # Return fresh dictionaries so callers cannot mutate the scored input list.
    return [dict(document) for document in reranked_documents[:retained_count]]


async def node_rerank(
    state: QueryGraphState,
) -> dict[str, list[dict[str, Any]]]:
    """Rerank available legal evidence and return the final candidate set."""
    # Step 1: Validate session metadata and register expensive model work.
    function_name = sys._getframe().f_code.co_name
    session_id = str(state.get("session_id") or "unknown")
    is_stream = bool(state.get("is_stream", False))
    add_running_task(session_id, function_name, is_stream)

    try:
        # Step 2: Flatten local RRF hits and optional network results into one
        # deduplicated schema containing text, title, source, URL, and metadata.
        candidates = step_1_merge_local_rrf_and_network_search(state)

        # Step 3: Cross-encode every query/document pair asynchronously and
        # sort candidates by normalized relevance score from highest to lowest.
        scored_candidates = await step_2_rerank_doc_list(candidates, state)

        # Step 4: Limit answer context and remove the low-scoring tail after a
        # clear score cliff, while preserving minimum legal context when present.
        final_documents = step_3_topk_and_gap(scored_candidates)

        # Step 5: Log compact diagnostics. Full legal passages are intentionally
        # omitted because they can be long and may contain sensitive user data.
        logger.info(
            f"Reranking completed: session_id={session_id}, "
            f"candidates={len(candidates)}, retained={len(final_documents)}, "
            f"scores={[document['score'] for document in final_documents]}"
        )

        # Step 6: Complete progress tracking and update only this node's field.
        add_done_task(session_id, function_name, is_stream)
        return {"reranked_docs": final_documents}
    except Exception:
        logger.exception(f"Reranking failed: session_id={session_id}")
        raise


if __name__ == "__main__":
    async def _manual_test() -> None:
        example_state: QueryGraphState = {  # type: ignore[typeddict-item]
            "session_id": "test_rerank_session",
            "original_query": "When may personal data be processed lawfully?",
            "rewritten_query": "When may personal data be processed lawfully?",
            "is_stream": False,
            "rrf_chunks": [
                {
                    "id": "gdpr:article_6:paragraph_1",
                    "rrf_score": 0.032,
                    "entity": {
                        "chunk_id": "gdpr:article_6:paragraph_1",
                        "law_title": "General Data Protection Regulation",
                        "article_title": "Lawfulness of processing",
                        "text": "Processing shall be lawful only if a legal basis applies.",
                    },
                }
            ],
            "network_searched_docs": [],
        }
        print(await node_rerank(example_state))

    asyncio.run(_manual_test())
