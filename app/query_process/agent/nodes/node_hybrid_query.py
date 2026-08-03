from __future__ import annotations

from app.query_process.agent.state import QueryGraphState


VALID_SEARCH_TYPES = {
    "semantic_search",
    "es_keywords_search",
    "hybrid_search",
}


def node_hybrid_query(state: QueryGraphState) -> dict[str, str]:
    """
    Validate the selected retrieval strategy.

    Despite its historical name, this node is the search planner and must not
    perform Elasticsearch or Milvus I/O. A later implementation may use rules
    or an LLM to select the strategy and rewrite the query.
    """
    search_type = state.get("search_type", "hybrid_search")
    if search_type not in VALID_SEARCH_TYPES:
        raise ValueError(f"Unsupported search_type: {search_type!r}")
    return {"search_type": search_type}
