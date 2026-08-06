"""
LangGraph workflow for legal RAG queries.

Workflow overview
=================

The graph separates *planning* from *execution*. ``node_hybrid_query`` keeps
its existing name for compatibility, but its responsibility is to choose one
of three search strategies; it must not perform Elasticsearch or Milvus I/O.

```text
START
  |
  v
node_intent_confirm
  |-- answer already available ----------------------> node_answer_output
  |
  |-- ordinary legal retrieval ----------------------> node_hybrid_query
  |
  `-- policy compliance
          |
          v
      node_policy_understanding
          |
          v
      node_compliance_retrieval_plan
          |
          v
      node_hybrid_query  (search-strategy selection)
          |
          |-- semantic_search --> node_embedding_semantic_search --.
          |                                                       |
          |-- es_keywords_search -> node_es_keywords_search -------+--> node_search_join
          |                                                       |
          `-- hybrid_search -----> both nodes in parallel --------'
                                                                    |
                              .-------------------------------------'
                              |
                              |-- one local search --> node_rerank
                              |
                              `-- hybrid search --> node_rrf --> node_rerank
                                                                  |
                                                                  v
                                                node_retrieval_quality_check
                                                   |             |
                                      sufficient --'             `-- insufficient
                                           |                           |
                                           v                           v
                                  node_answer_output          node_network_search
                                           ^                           |
                                           |                           v
                                           `------ quality check <- node_rerank

After one network fallback, insufficient evidence routes to answer output
instead of searching repeatedly. The answer node can then explain that
authoritative evidence was insufficient rather than entering an infinite loop.
```

Equivalent Mermaid diagram
===========================

```mermaid
flowchart TD
    START([START]) --> Intent[node_intent_confirm]
    Intent -->|direct answer| Answer[node_answer_output]
    Intent -->|ordinary legal question| Strategy[node_hybrid_query]
    Intent -->|policy compliance| Policy[node_policy_understanding]
    Policy --> Plan[node_compliance_retrieval_plan]
    Plan --> Strategy

    Strategy -->|semantic_search| Semantic[node_embedding_semantic_search]
    Strategy -->|es_keywords_search| Keyword[node_es_keywords_search]
    Strategy -->|hybrid_search| Semantic
    Strategy -->|hybrid_search| Keyword

    Semantic --> Join[node_search_join]
    Keyword --> Join
    Join -->|single search| Rerank[node_rerank]
    Join -->|hybrid search| RRF[node_rrf]
    RRF --> Rerank

    Rerank --> Quality[node_retrieval_quality_check]
    Quality -->|sufficient| Answer
    Quality -->|insufficient; fallback unused| Network[node_network_search]
    Network --> Rerank
    Quality -->|insufficient; fallback used| Answer
    Answer --> END([END])
```

Parallel-state rule
===================

For hybrid search, the semantic and Elasticsearch nodes run in the same graph
step. They must return partial updates to different keys:

```text
node_embedding_semantic_search -> semantic_searched_chunks
node_es_keywords_search        -> keywords_searched_chunks
```

Returning the entire state from both parallel branches would cause concurrent
updates to the same keys. ``node_search_join`` acts as the synchronization
barrier before RRF.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_compliance_retrieval_plan import node_compliance_retrieval_plan
from app.query_process.agent.nodes.node_embedding_semantic_search import node_embedding_semantic_search
from app.query_process.agent.nodes.node_es_keywords_search import node_es_keywords_search
from app.query_process.agent.nodes.node_hybrid_query import node_hybrid_query
from app.query_process.agent.nodes.node_intent_confirm import node_intent_confirm
from app.query_process.agent.nodes.node_network_search import node_network_search
from app.query_process.agent.nodes.node_policy_understanding import node_policy_understanding
from app.query_process.agent.nodes.node_retrieval_quality_check import node_retrieval_quality_check
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_join import node_search_join

from app.query_process.agent.state import (
    QueryGraphState,
    create_query_default_state,
)

load_dotenv()

SearchType = Literal[
    "semantic_search",
    "es_keywords_search",
    "hybrid_search",
]

NodeCallable = Callable[[QueryGraphState], Any]

SEARCH_NODE_NAMES = {
    "semantic_search": "node_embedding_semantic_search",
    "es_keywords_search": "node_es_keywords_search",
}


def route_after_intent_confirm(state: QueryGraphState) -> str:
    """
    Route direct responses away from retrieval.

    ``node_intent_confirm`` should put text in ``state["answer"]`` only for
    cases that do not need legal evidence, such as greetings, unsupported
    requests, or a clarification question. Substantive legal questions should
    continue to the search-strategy node.
    """
    if state.get("answer"):
        return "node_answer_output"
    if state.get("task_type") == "policy_compliance":
        return "node_policy_understanding"
    return "node_hybrid_query"


def route_after_search_strategy(
    state: QueryGraphState,
) -> str | Sequence[str]:
    """
    Start the selected local retrieval branch or both hybrid branches.

    Returning a two-item sequence is LangGraph fan-out: Milvus semantic search
    and Elasticsearch BM25 search execute in parallel.
    """
    search_type = state.get("search_type")
    if search_type == "hybrid_search":
        return (
            SEARCH_NODE_NAMES["semantic_search"],
            SEARCH_NODE_NAMES["es_keywords_search"],
        )
    if search_type in SEARCH_NODE_NAMES:
        return SEARCH_NODE_NAMES[search_type]
    raise ValueError(
        f"Unsupported search_type {search_type!r}. Expected one of: "
        f"{', '.join(sorted((*SEARCH_NODE_NAMES, 'hybrid_search')))}."
    )


def route_after_search_join(state: QueryGraphState) -> str:
    """
    Apply RRF only when two independent ranked lists were requested.

    A single semantic or BM25 result list already has an internal ranking, so
    it goes directly to the cross-encoder reranker.
    """
    if state.get("search_type") == "hybrid_search":
        return "node_rrf"
    return "node_rerank"


def route_after_retrieval_quality_check(
    state: QueryGraphState,
) -> str:
    """
    Use web search once when local evidence remains too weak.

    After the fallback has been attempted, routing to answer output prevents an
    infinite network-search loop. The answer node should use
    ``retrieval_failure_reason`` to avoid presenting unsupported claims.
    """
    if state.get("retrieval_is_sufficient"):
        return "node_answer_output"
    if not state.get("network_fallback_used", False):
        return "node_network_search"
    return "node_answer_output"




def build_query_workflow(
    node_overrides: Mapping[str, NodeCallable] | None = None,
) -> StateGraph:
    """
    Build and return the uncompiled query workflow.

    ``node_overrides`` exists for isolated routing tests. Production callers
    normally use the compiled module-level ``query_app``. Keeping graph
    construction in a function makes every route testable without loading
    embedding, database, reranker, network, or LLM services.
    """
    nodes = {
        "node_intent_confirm": node_intent_confirm,
        "node_policy_understanding": node_policy_understanding,
        "node_compliance_retrieval_plan": node_compliance_retrieval_plan,
        "node_hybrid_query": node_hybrid_query,
        "node_embedding_semantic_search": node_embedding_semantic_search,
        "node_es_keywords_search": node_es_keywords_search,
        "node_search_join": node_search_join,
        "node_rrf": node_rrf,
        "node_rerank": node_rerank,
        "node_retrieval_quality_check": node_retrieval_quality_check,
        "node_network_search": node_network_search,
        "node_answer_output": node_answer_output,
    }

    if node_overrides:
        unknown_nodes = sorted(set(node_overrides) - set(nodes))
        if unknown_nodes:
            raise ValueError(
                "Unknown query graph node overrides: "
                + ", ".join(unknown_nodes)
            )
        nodes.update(node_overrides)

    workflow = StateGraph(QueryGraphState)
    for node_name, node_function in nodes.items():
        workflow.add_node(node_name, node_function)

    # Entry and intent routing.
    workflow.add_edge(START, "node_intent_confirm")
    workflow.add_conditional_edges(
        "node_intent_confirm",
        route_after_intent_confirm,
        {
            "node_answer_output": "node_answer_output",
            "node_policy_understanding": "node_policy_understanding",
            "node_hybrid_query": "node_hybrid_query",
        },
    )

    # Policy assessments first produce independently validated rules and then
    # convert them into the query fields consumed by the existing search path.
    workflow.add_edge("node_policy_understanding","node_compliance_retrieval_plan")
    workflow.add_edge("node_compliance_retrieval_plan", "node_hybrid_query")

    # Search-strategy fan-out. A hybrid decision returns both node names.
    workflow.add_conditional_edges("node_hybrid_query", route_after_search_strategy,
        {
            "node_embedding_semantic_search": "node_embedding_semantic_search",
            "node_es_keywords_search": "node_es_keywords_search",
        },
    )

    # Both search branches converge here. During hybrid fan-out, LangGraph
    # schedules this join after both branches in the current graph step finish.
    workflow.add_edge(
        "node_embedding_semantic_search",
        "node_search_join",
    )
    workflow.add_edge(
        "node_es_keywords_search",
        "node_search_join",
    )

    # Only hybrid retrieval needs rank fusion.
    workflow.add_conditional_edges(
        "node_search_join",
        route_after_search_join,
        {
            "node_rrf": "node_rrf",
            "node_rerank": "node_rerank",
        },
    )
    workflow.add_edge("node_rrf", "node_rerank")

    # Reranker scores provide a common quality signal after BM25, vectors, RRF,
    # or a network fallback. Raw BM25 and cosine scores are not comparable.
    workflow.add_edge(
        "node_rerank",
        "node_retrieval_quality_check",
    )
    workflow.add_conditional_edges(
        "node_retrieval_quality_check",
        route_after_retrieval_quality_check,
        {
            "node_answer_output": "node_answer_output",
            "node_network_search": "node_network_search",
        },
    )

    # Network results are reranked with the same query. The network node must
    # set network_fallback_used=True so the quality gate cannot loop forever.
    workflow.add_edge("node_network_search", "node_rerank")
    workflow.add_edge("node_answer_output", END)
    return workflow


# Import servers and tests invoke this compiled graph.
query_app = build_query_workflow().compile()

async def _manual_test() -> None:
    """
    Run one legal question through the complete compiled query graph.

    This integration test calls the configured MongoDB, LLM, TEI embedding,
    Milvus, Elasticsearch, reranker, and answer-generation services. When the
    local retrieval quality is insufficient, it may also call the configured
    MCP web-search service. Unlike a node-by-node test, invoking ``query_app``
    verifies the real conditional routing, hybrid fan-out, join, RRF, rerank,
    quality gate, optional network fallback, and final answer node together.
    """
    # Step 1: Build a complete initial state for a substantive legal question.
    # Explicit hybrid search guarantees that both local retrieval branches are
    # selected unless the intent node determines that no retrieval is needed.
    mock_state: QueryGraphState = create_query_default_state(
        session_id="test_session_001",
        original_query=(
            # "May a news article be quoted in part when the publisher and "
            # "author are acknowledged?"
            "compliance the policy : The document may be downloaded once and retained for 30 days for academic research only."
        ),
        search_type="hybrid_search",
        is_stream=False,
    )

    # Step 2: Invoke the compiled graph rather than calling nodes manually.
    # LangGraph now owns branch selection, parallel execution, synchronization,
    # fallback-loop prevention, and termination at END.
    final_state = await query_app.ainvoke(mock_state)

    # Step 3: Confirm that the graph reached its required final output. An empty
    # answer normally indicates a routing or answer-generation failure.
    if not final_state.get("answer", "").strip():
        raise RuntimeError(
            "The complete query graph finished without generating an answer."
        )

    # Step 4: Print compact stage counts before the complete JSON state. These
    # values make it easy to verify which retrieval and fallback paths ran.
    print("\n" + "=" * 100)
    print("FULL QUERY GRAPH INTEGRATION TEST COMPLETED")
    print(f"Search strategy: {final_state['search_type']}")
    print(f"Semantic hits: {len(final_state['semantic_searched_chunks'])}")
    print(f"BM25 hits: {len(final_state['keywords_searched_chunks'])}")
    print(f"RRF candidates: {len(final_state['rrf_chunks'])}")

    print(f"Reranked documents: {len(final_state['reranked_docs'])}")

    print(f"Network fallback used: {final_state['network_fallback_used']}")

    print(f"Retrieval sufficient: {final_state['retrieval_is_sufficient']}")

    if final_state.get("retrieval_failure_reason"):
        print(
            "Retrieval failure reason: "
            f"{final_state['retrieval_failure_reason']}"
        )
    print("=" * 100)

    # Step 5: Print the complete final state for manual inspection of rewritten
    # query, key terms, retrieved evidence, reranker scores, and final answer.
    print(json.dumps(final_state, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(_manual_test())
