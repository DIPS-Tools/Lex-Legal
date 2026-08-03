"""
Evidence-quality gate between cross-encoder reranking and answer generation.

Workflow
========

```text
state["reranked_docs"]
        |
        v
validate list and usable legal text
        |
        v
validate normalized scores, source, and descending order
        |
        v
compare top score with its source-aware threshold
        |
        |-- sufficient ----------------------> node_answer_output
        |
        `-- insufficient + fallback unused -> node_network_search
                    |
                    `-- rerank -> quality check again
                                |
                                `-- still insufficient -> node_answer_output
```

This node does not perform network search itself. It returns a quality decision
and explanation; ``query_main_graph.route_after_retrieval_quality_check`` owns
the fallback routing and ensures that network search is attempted at most once.

The thresholds apply only because ``node_rerank`` requests ``normalize=True``
and validates scores in the range ``[0, 1]``. Curated local law records use a
lower threshold than open-web snippets. Both are initial operational values
that should eventually be calibrated with labelled legal-query examples.
"""

from __future__ import annotations

import math
import sys
from numbers import Real
from typing import Any, TypedDict

from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


# Curated local records already come from the project's controlled legal
# corpus. A lower cross-encoder threshold allows broad compliance questions to
# use genuinely relevant provisions such as research-data rules even when the
# question and statutory wording differ substantially.
MIN_LOCAL_RERANK_SCORE = 0.15

# Open-web snippets are less controlled and may be incomplete, outdated, or
# secondary commentary. Keep the stricter threshold before allowing them to
# support an answer. Both values are operational starting points and should be
# calibrated against a labelled legal-query evaluation set.
MIN_WEB_RERANK_SCORE = 0.35

# Backward-compatible alias for callers and tests that used the original name.
MIN_RERANK_SCORE = MIN_LOCAL_RERANK_SCORE


class QualityAssessment(TypedDict):
    """Internal result used to keep node routing decisions explicit."""

    is_sufficient: bool
    reason: str
    usable_count: int
    top_score: float | None
    top_source: str | None
    required_score: float | None


def _usable_documents(state: QueryGraphState) -> list[dict[str, Any]]:
    """
    Return reranked records that contain non-empty evidence text.

    The function does not modify state or silently coerce malformed records.
    Empty text cannot support a legal answer even if a stale score is present.
    """
    documents = state.get("reranked_docs", [])
    if not isinstance(documents, list):
        raise TypeError("state['reranked_docs'] must be a list.")

    return [
        document
        for document in documents
        if (
            isinstance(document, dict)
            and isinstance(document.get("text"), str)
            and document["text"].strip()
        )
    ]


def _normalized_score(document: dict[str, Any]) -> float | None:
    """Return one valid normalized reranker score, otherwise ``None``."""
    score = document.get("score")
    if not isinstance(score, Real) or isinstance(score, bool):
        return None
    normalized_score = float(score)
    if not math.isfinite(normalized_score):
        return None
    if not 0.0 <= normalized_score <= 1.0:
        return None
    return normalized_score


def assess_retrieval_quality(state: QueryGraphState) -> QualityAssessment:
    """Validate reranker output and determine whether evidence is sufficient."""
    # Step 1: Select records containing actual evidence text. No usable record
    # means the answer model would have no authoritative context.
    documents = _usable_documents(state)
    if not documents:
        return {
            "is_sufficient": False,
            "reason": "No reranked document contains usable evidence text.",
            "usable_count": 0,
            "top_score": None,
            "top_source": None,
            "required_score": None,
        }

    # Step 2: Validate every retained cross-encoder score. The reranker should
    # already guarantee normalized finite values, so malformed scores indicate
    # a broken upstream contract and must fail closed.
    scores = [_normalized_score(document) for document in documents]
    if any(score is None for score in scores):
        return {
            "is_sufficient": False,
            "reason": (
                "At least one reranked document has an invalid normalized "
                "relevance score."
            ),
            "usable_count": len(documents),
            "top_score": None,
            "top_source": None,
            "required_score": None,
        }
    validated_scores = [float(score) for score in scores if score is not None]

    # Step 3: Confirm descending order. Answer generation consumes documents in
    # list order, so accepting an unsorted list could place weak evidence first.
    if any(current_score < next_score for current_score, next_score in zip(
            validated_scores,
            validated_scores[1:],
        )
    ):
        return {
            "is_sufficient": False,
            "reason": "Reranked documents are not ordered by descending score.",
            "usable_count": len(documents),
            "top_score": validated_scores[0],
            "top_source": None,
            "required_score": None,
        }

    # Step 4: Read the highest normalized score. The preceding ordering check
    # guarantees that the first usable document is the strongest candidate.
    top_score = validated_scores[0]
    top_source = str(documents[0].get("source") or "").strip().lower()
    if top_source == "local":
        required_score = MIN_LOCAL_RERANK_SCORE
    elif top_source == "web":
        required_score = MIN_WEB_RERANK_SCORE
    else:
        return {
            "is_sufficient": False,
            "reason": (
                "The highest-ranked document has no recognized evidence "
                "source classification."
            ),
            "usable_count": len(documents),
            "top_score": top_score,
            "top_source": top_source or None,
            "required_score": None,
        }

    # Step 5: Compare only the cross-encoder score with the source-aware gate.
    # BM25, cosine, Tavily, and RRF scores remain diagnostic metadata only.
    is_sufficient = top_score >= required_score
    reason = (
        ""
        if is_sufficient
        else (
            f"Top {top_source} reranker score {top_score:.4f} is below the "
            f"current {top_source} threshold {required_score:.4f}."
        )
    )

    # Step 6: Return metrics used for logs and the graph-state decision. The
    # public node maps these internal names to the stable state field names.
    return {
        "is_sufficient": is_sufficient,
        "reason": reason,
        "usable_count": len(documents),
        "top_score": top_score,
        "top_source": top_source,
        "required_score": required_score,
    }


def node_retrieval_quality_check(
    state: QueryGraphState,
) -> dict[str, Any]:
    """Set the legal-evidence quality decision used by graph routing."""
    # Step 1: Register this lightweight validation node in task progress.
    function_name = sys._getframe().f_code.co_name
    session_id = str(state.get("session_id") or "unknown")
    is_stream = bool(state.get("is_stream", False))
    add_running_task(session_id, function_name, is_stream)

    try:
        # Step 2: Evaluate usable evidence, normalized scores, ordering, and the
        # configured threshold without mutating reranked documents.
        assessment = assess_retrieval_quality(state)

        # Step 3: Read fallback state only for diagnostics. The graph router—not
        # this node—decides whether network search should execute next.
        network_fallback_used = bool(
            state.get("network_fallback_used", False)
        )

        # Step 4: Log a warning for rejected evidence and an informational event
        # for accepted evidence, including only compact quality metrics.
        message = (
            f"Retrieval quality evaluated: session_id={session_id}, "
            f"usable={assessment['usable_count']}, "
            f"top_score={assessment['top_score']}, "
            f"top_source={assessment['top_source']}, "
            f"required_score={assessment['required_score']}, "
            f"sufficient={assessment['is_sufficient']}, "
            f"network_fallback_used={network_fallback_used}, "
            f"reason={assessment['reason'] or 'accepted'}"
        )
        if assessment["is_sufficient"]:
            logger.info(message)
        else:
            logger.warning(message)

        # Step 5: Complete task tracking even when evidence is insufficient;
        # insufficient evidence is a valid routing result, not a node failure.
        add_done_task(session_id, function_name, is_stream)

        # Step 6: Return only fields owned by this node. The failure reason is
        # later used by node_answer_output if the one web fallback also fails.
        return {
            "retrieval_is_sufficient": assessment["is_sufficient"],
            "retrieval_failure_reason": assessment["reason"],
        }
    except Exception:
        logger.exception(
            f"Retrieval quality check failed unexpectedly: "
            f"session_id={session_id}"
        )
        raise
