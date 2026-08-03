"""
Synchronization barrier for local legal-document retrieval branches.

Workflow
========

```text
semantic_search ---------.
                          |
es_keywords_search ------+--> node_search_join
                          |          |
hybrid: both branches ---'          |-- single branch -> node_rerank
                                     `-- both branches -> node_rrf
```

The join does not compare BM25 and cosine scores because those score spaces are
not directly comparable. For hybrid retrieval, RRF reads the canonical branch
fields directly. For one strategy, the join places that ranked list in
``rrf_chunks`` so the existing reranker receives a consistent input field.
"""

from __future__ import annotations

import sys
from typing import Any

from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


VALID_SEARCH_TYPES = {
    "semantic_search",
    "es_keywords_search",
    "hybrid_search",
}


def _require_hit_list(state: QueryGraphState, field_name: str) -> list:
    """Validate one branch result without treating an empty search as an error."""
    value = state.get(field_name, [])
    if not isinstance(value, list):
        raise TypeError(f"state[{field_name!r}] must be a list.")
    return value


def node_search_join(state: QueryGraphState) -> dict[str, Any]:
    """Synchronize search branches and prepare the downstream ranking input."""
    # Step 1: Register this synchronization node and validate the selected
    # strategy before examining branch-specific results.
    function_name = sys._getframe().f_code.co_name
    session_id = str(state.get("session_id") or "unknown")
    is_stream = bool(state.get("is_stream", False))
    search_type = state.get("search_type")
    if search_type not in VALID_SEARCH_TYPES:
        raise ValueError(f"Unsupported search_type: {search_type!r}")
    add_running_task(session_id, function_name, is_stream)

    # Step 2: Read both canonical result fields. LangGraph reaches this node
    # only after every branch selected in the current graph step has finished.
    semantic_hits = _require_hit_list(state, "semantic_searched_chunks")
    keyword_hits = _require_hit_list(state, "keywords_searched_chunks")

    # Step 3: Validate branch availability for the chosen strategy. An empty
    # list is valid; it means the search completed but found no matching rows.
    if search_type == "semantic_search":
        selected_hits = semantic_hits
    elif search_type == "es_keywords_search":
        selected_hits = keyword_hits
    else:
        selected_hits = []

    # Step 4: Hybrid lists already live in their canonical state fields, so no
    # aliases are needed. A single branch still goes directly to reranking.
    if search_type == "hybrid_search":
        update: dict[str, Any] = {}
    else:
        update = {
            "rrf_chunks": selected_hits,
        }

    # Step 5: Log counts and routing information without logging full legal
    # passages. The graph's conditional edge chooses RRF or rerank next.
    logger.info(
        f"Local search branches joined: session_id={session_id}, "
        f"search_type={search_type}, semantic_count={len(semantic_hits)}, "
        f"keyword_count={len(keyword_hits)}, next="
        f"{'node_rrf' if search_type == 'hybrid_search' else 'node_rerank'}"
    )

    # Step 6: Mark synchronization complete and return only fields owned by
    # this node. Hybrid branch results remain in their canonical state fields.
    add_done_task(session_id, function_name, is_stream)
    return update
