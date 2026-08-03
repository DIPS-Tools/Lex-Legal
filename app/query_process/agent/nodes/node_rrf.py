"""
Reciprocal-rank fusion for semantic and Elasticsearch retrieval results.

Workflow
========

```text
semantic_searched_chunks ----.
                              +--> weighted RRF --> rrf_chunks
keywords_searched_chunks ----'
```

RRF uses result positions rather than raw scores. This is important because a
Milvus cosine score and an Elasticsearch BM25 score are not directly
comparable. A chunk found by both systems receives contributions from both
ranked lists and therefore moves upward in the fused ranking.

Formula
-------

For each occurrence of a chunk in a ranked source:

```text
source contribution = source_weight / (rank_constant + rank)
final RRF score      = sum of all source contributions
```

With ``rank_constant=60``, a first-place result contributes ``1 / 61`` and a
second-place result contributes ``1 / 62`` when the source weight is ``1.0``.
The constant reduces excessive differences between neighboring ranks. A chunk
ranked second by both systems will usually outrank a chunk ranked first by only
one system because it receives two contributions.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


# A commonly used RRF constant. A larger value makes neighboring positions
# contribute more similarly; a smaller value rewards the first positions more.
RRF_RANK_CONSTANT = 60

# Only the highest fused candidates proceed to the more expensive cross-encoder
# reranker. Retrieval nodes currently fetch up to twenty candidates per source.
RRF_TOP_K = 10

# Equal weights treat semantic and lexical rankings as equally important. These
# values can later be tuned from evaluation data without changing the formula.
SEMANTIC_WEIGHT = 1.0
BM25_WEIGHT = 1.0


def _get_chunk_id(chunk: dict[str, Any]) -> str:
    """
    Read the stable shared chunk ID from the normalized retrieval shape.

    Search records currently expose the ID at the top level as ``id`` and
    inside ``entity`` as ``chunk_id``. The additional top-level ``chunk_id``
    fallback keeps this helper usable with repository-native records. RRF must
    use this stable ID—not document text—because the same legal chunk returned
    by Milvus and Elasticsearch must accumulate into one score.
    """
    # Read entity defensively because malformed input may contain a non-dict
    # value. Validation below produces a clear error instead of AttributeError.
    entity = chunk.get("entity")
    entity_chunk_id = entity.get("chunk_id") if isinstance(entity, dict) else None

    # Prefer the normalized top-level ID, then support the two compatible
    # storage shapes used elsewhere in the project.
    chunk_id = chunk.get("id") or chunk.get("chunk_id") or entity_chunk_id
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("RRF input is missing a non-empty chunk_id.")
    return chunk_id.strip()


def step_3_reciprocal_rank_fusion(
    source_with_weight: Sequence[tuple[Sequence[dict[str, Any]], float]],
    top_k: int = RRF_TOP_K,
    rank_constant: int = RRF_RANK_CONSTANT,
) -> list[dict[str, Any]]:
    """
    Fuse independent ranked lists using weighted reciprocal rank.

    ``source_with_weight`` contains pairs such as
    ``(semantic_results, 1.0)`` and ``(bm25_results, 1.0)``. Each source list
    must already be ordered from most relevant to least relevant. The function
    does not inspect or normalize each source's raw ``score`` field.
    """
    # Step 1: Validate algorithm controls before processing any records. Zero
    # top_k would make the fusion silently useless, while a negative constant
    # could create zero or negative denominators for early ranks.
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if rank_constant < 0:
        raise ValueError("rank_constant must not be negative.")

    # Step 2: Keep one accumulated score and one representative record for
    # every shared chunk_id. The representative preserves text and metadata
    # required by the downstream cross-encoder reranker.
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, dict[str, Any]] = {}

    # Step 3: Process each retrieval source independently. Source order does
    # not affect scores, although it determines which equivalent record is
    # retained in chunks_by_id via setdefault().
    for source, weight in source_with_weight:
        if weight < 0:
            raise ValueError("RRF source weights must not be negative.")

        # A malformed source could repeat the same chunk. Count it only once
        # per source so duplicate rows cannot artificially inflate its score.
        seen_in_source: set[str] = set()
        for rank, chunk in enumerate(source, start=1):
            chunk_id = _get_chunk_id(chunk)
            if chunk_id in seen_in_source:
                continue
            seen_in_source.add(chunk_id)

            # Step 4: Add this source's rank contribution. If the other source
            # returns the same chunk_id, its contribution is added here too.
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                weight / (rank_constant + rank)
            )

            # Preserve the first complete normalized hit. Both search nodes
            # provide authoritative ES text, so either representation is valid.
            chunks_by_id.setdefault(chunk_id, chunk)

    # Step 5: Sort IDs by the accumulated cross-source score and apply top_k.
    # Python's stable sort preserves first-seen order when scores are equal.
    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )[:top_k]

    # Step 6: Copy each retained record and expose rrf_score for diagnostics.
    # Dictionary unpacking avoids mutating either original retrieval list.
    return [
        {
            **chunks_by_id[chunk_id],
            "rrf_score": scores[chunk_id],
        }
        for chunk_id in ranked_ids
    ]


def node_rrf(state: QueryGraphState) -> dict[str, list[dict[str, Any]]]:
    """Fuse canonical semantic and BM25 result lists for reranking."""
    # Step 1: Register node progress and read the session configuration.
    function_name = sys._getframe().f_code.co_name
    session_id = str(state.get("session_id") or "unknown")
    is_stream = bool(state.get("is_stream", False))
    add_running_task(session_id, function_name, is_stream)

    try:
        # Step 2: Read canonical branch outputs directly. No embedding_chunks
        # or hyde_embedding_chunks compatibility aliases are required.
        semantic_chunks = state.get("semantic_searched_chunks", [])
        keyword_chunks = state.get("keywords_searched_chunks", [])
        if not isinstance(semantic_chunks, list):
            raise TypeError("semantic_searched_chunks must be a list.")
        if not isinstance(keyword_chunks, list):
            raise TypeError("keywords_searched_chunks must be a list.")

        # Step 3: Associate each ranked source with its configurable weight.
        sources = [
            (semantic_chunks, SEMANTIC_WEIGHT),
            (keyword_chunks, BM25_WEIGHT),
        ]

        # Step 4: Fuse ranks. Shared chunk IDs accumulate contributions from
        # both retrieval systems without comparing their raw relevance scores.
        fused_chunks = step_3_reciprocal_rank_fusion(
            sources,
            top_k=RRF_TOP_K,
        )

        # Step 5: Log counts and fused IDs without dumping complete legal text.
        logger.info(
            f"RRF completed: session_id={session_id}, "
            f"semantic={len(semantic_chunks)}, bm25={len(keyword_chunks)}, "
            f"fused={len(fused_chunks)}, "
            f"chunk_ids={[_get_chunk_id(chunk) for chunk in fused_chunks]}"
        )

        # Step 6: Complete task tracking and return only this node's state key.
        add_done_task(session_id, function_name, is_stream)
        return {"rrf_chunks": fused_chunks}
    except Exception:
        logger.exception(f"RRF failed: session_id={session_id}")
        raise


if __name__ == "__main__":
    example_state: QueryGraphState = {  # type: ignore[typeddict-item]
        "session_id": "test_rrf_session",
        "is_stream": False,
        "semantic_searched_chunks": [
            {"id": "chunk-a", "entity": {"chunk_id": "chunk-a"}},
            {"id": "chunk-b", "entity": {"chunk_id": "chunk-b"}},
        ],
        "keywords_searched_chunks": [
            {"id": "chunk-b", "entity": {"chunk_id": "chunk-b"}},
            {"id": "chunk-c", "entity": {"chunk_id": "chunk-c"}},
        ],
    }
    print(node_rrf(example_state))
