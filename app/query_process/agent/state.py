import copy
from typing import Any, Literal

from typing_extensions import TypedDict


class RequestedOutputs(TypedDict):
    """User-visible outputs; ODRL analysis always includes JSON-LD."""

    odrl_analysis: bool
    odrl_jsonld_document: bool
    compliance_assessment: bool


class ODRLState(TypedDict):
    """Canonical ODRL extraction state owned by the ODRL workflow."""

    source_text: str
    validated_policy_rules: list[dict[str, Any]]
    odrl_jsonld_document: dict[str, Any] | None

    endpoint_rules: list[dict[str, Any]]
    independent_ai_rules: list[dict[str, Any]]
    comparison: dict[str, Any]
    extracted_policy_context: dict[str, Any]
    extraction_errors: list[str]

    extraction_id: str
    reconciliation_method: str
    reconciliation_error: str
    audit_log_path: str
    endpoint_latency_ms: int
    llm_latency_ms: int

    serialization_status: Literal[
        "not_requested",
        "not_implemented",
        "generated",
        "validation_failed",
    ]
    serialization_errors: list[str]

    validation_status: Literal[
        "not_run",
        "valid",
        "repaired",
        "invalid",
        "error",
    ]
    validation_result: dict[str, Any]
    initial_validation_result: dict[str, Any]
    original_odrl_jsonld_document: dict[str, Any] | None
    validation_repair_attempted: bool
    validation_errors: list[str]


def create_default_odrl_state() -> ODRLState:
    """Return a fresh ODRL state without shared nested mutable values."""
    return {
        "source_text": "",
        "validated_policy_rules": [],
        "odrl_jsonld_document": None,
        "endpoint_rules": [],
        "independent_ai_rules": [],
        "comparison": {},
        "extracted_policy_context": {},
        "extraction_errors": [],
        "extraction_id": "",
        "reconciliation_method": "",
        "reconciliation_error": "",
        "audit_log_path": "",
        "endpoint_latency_ms": 0,
        "llm_latency_ms": 0,
        "serialization_status": "not_requested",
        "serialization_errors": [],
        "validation_status": "not_run",
        "validation_result": {},
        "initial_validation_result": {},
        "original_odrl_jsonld_document": None,
        "validation_repair_attempted": False,
        "validation_errors": [],
    }


class QueryGraphState(TypedDict):
    """
    Shared state passed through the legal-document query workflow.

    Parallel retrieval nodes must return partial dictionaries and update only
    their own result field. For example, the semantic node writes
    ``semantic_searched_chunks`` while the Elasticsearch node writes
    ``keywords_searched_chunks``. This prevents concurrent LangGraph branches
    from attempting to update the same state key.
    """
    session_id: str
    query_run_id: str
    original_query: str

    # Intent classification and optional policy-understanding results.
    task_type: Literal[
        "legal_question",
        "odrl_processing",
        "policy_compliance",
    ]
    requested_outputs: RequestedOutputs
    # pure the policy state, removing question description
    policy_text: str
    odrl_state: ODRLState
    
    odrl_extracted_rules: list[dict]
    llm_extracted_rules: list[dict]
    policy_extraction_comparison: dict
    validated_policy_rules: list[dict]

    policy_context: dict
    policy_understanding_errors: list[str]
    odrl_audit_log_path: str
    compliance_retrieval_queries: list[dict]

    # Search planning and retrieval results.
    search_type: Literal[
        "semantic_search",
        "es_keywords_search",
        "hybrid_search",
    ]
    semantic_searched_chunks: list
    keywords_searched_chunks: list
    network_searched_docs: list

    # Fusion, reranking, and fallback decisions.
    rrf_chunks: list
    reranked_docs: list
    retrieval_is_sufficient: bool
    retrieval_failure_reason: str
    # • network_fallback_used records whether the workflow has already tried an external web search.
    #
    # - False: Web search has not run, so weak local results can trigger it.
    # - True: Web search has already run, so the graph must proceed to answer output even if evidence remains weak
    #
    network_fallback_used: bool

    # Answer generation and presentation.
    answer: str
    image_urls: list[str]

    # Supporting query information.
    query_key_terms: list[str]
    rewritten_query: str
    history: list
    is_stream: bool


# ========================
# default state
# ========================
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "query_run_id": "",
    "original_query": "",
    "task_type": "legal_question",
    "requested_outputs": {
        "odrl_analysis": False,
        "odrl_jsonld_document": False,
        "compliance_assessment": False,
    },
    "policy_text": "",
    "odrl_state": create_default_odrl_state(),
    "odrl_extracted_rules": [],
    "llm_extracted_rules": [],
    "policy_extraction_comparison": {},
    "validated_policy_rules": [],
    "policy_context": {},
    "policy_understanding_errors": [],
    "odrl_audit_log_path": "",
    "compliance_retrieval_queries": [],
    "semantic_searched_chunks": [],
    "keywords_searched_chunks": [],
    "network_searched_docs": [],
    "search_type": "hybrid_search",
    "rrf_chunks": [],
    "reranked_docs": [],
    "retrieval_is_sufficient": False,
    "retrieval_failure_reason": "",
    "network_fallback_used": False,
    "answer": "",
    "image_urls": [],
    "query_key_terms": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False
}


# ========================
# create a default state
# ========================
def create_query_default_state(**overrides) -> QueryGraphState:
    """
    创建查询流程的默认状态，支持覆盖字段
    """
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state
