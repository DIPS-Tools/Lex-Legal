"""
Classify query intent, rewrite the query, and extract legal search terms.

Workflow
========

```text
QueryGraphState
    |
    |-- read session_id and original_query
    v
MongoDB: load recent messages
    |
    |-- format bounded conversation context
    v
LLM in JSON mode
    |
    |-- classify intent
    |-- resolve references from history
    |-- rewrite a standalone query
    |-- extract legal concepts and law names
    v
Validate and normalize the JSON response
    |
    |-- legal_retrieval ------> leave answer empty -> retrieval workflow
    |-- clarification_needed -> set answer --------> answer output
    `-- general_conversation -> set answer --------> answer output
    |
    v
MongoDB: save the current user message with query metadata
```

Elasticsearch is deliberately not used to decide whether a question is legal.
An index miss can mean that the wording differs from the indexed text, not that
the question is unrelated to law. Elasticsearch and Milvus therefore remain
retrieval services used after this intent node.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage

from app.clients.mongo_utils import (
    get_recent_messages,
    save_chat_message,
)
from app.core.logger import logger
from app.llm.llm_utils import get_llm_client
from app.prompt.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


IntentName = Literal[
    "legal_retrieval",
    "clarification_needed",
    "general_conversation",
]

VALID_INTENTS: set[str] = {
    "legal_retrieval",
    "clarification_needed",
    "general_conversation",
}
MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_CHARACTERS = 8_000
MAX_QUERY_KEY_TERMS = 12


class IntentAnalysis(TypedDict):
    """Normalized information returned by the intent-analysis step."""

    intent: IntentName
    query_key_terms: list[str]
    rewritten_query: str
    direct_answer: str


def _format_history_for_prompt(
    history_chats: Sequence[Mapping[str, Any]],
) -> str:
    """
    Convert MongoDB history records into compact, readable prompt context.

    MongoDB returns records in chronological order. Only fields useful for
    resolving references are included; internal IDs and image URLs would add
    tokens without helping intent classification. A character bound prevents
    unexpectedly large historical messages from dominating the LLM prompt.
    """
    if not history_chats:
        return "No previous conversation is available."

    newest_first_lines: list[str] = []
    used_characters = 0
    for chat in reversed(history_chats):
        role = str(chat.get("role") or "unknown").strip()
        text = str(chat.get("text") or "").strip()
        if not text:
            continue

        rewritten_query = str(chat.get("rewritten_query") or "").strip()
        query_key_terms = _normalize_query_key_terms(
            chat.get("query_key_terms", [])
        )
        line = f"Role: {role}\nMessage: {text}"
        if rewritten_query:
            line += f"\nEarlier rewritten query: {rewritten_query}"
        if query_key_terms:
            line += f"\nEarlier legal key terms: {', '.join(query_key_terms)}"

        # Keep the newest useful context when the configured history contains
        # a very long message. Mongo already limits this input to ten records.
        remaining = MAX_HISTORY_CHARACTERS - used_characters
        if remaining <= 0:
            break
        newest_first_lines.append(line[:remaining])
        used_characters += min(len(line), remaining)

    newest_first_lines.reverse()
    return (
        "\n\n".join(newest_first_lines)
        or "No previous conversation is available."
    )


def _extract_json_object(content: Any) -> dict[str, Any]:
    """
    Parse one JSON object from an LLM response.

    JSON mode should return plain JSON, but the fence handling keeps the node
    compatible with OpenAI-compatible providers that still wrap the object in
    a Markdown code block. Invalid or non-object output raises a clear error
    instead of silently sending corrupt state to later graph nodes.
    """
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Intent-analysis LLM returned empty content.")

    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Intent-analysis LLM response must be a JSON object.")
    return parsed


def _normalize_query_key_terms(values: Any) -> list[str]:
    """Return unique, non-empty key terms while preserving model order."""
    if not isinstance(values, (list, tuple)):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        term = " ".join(value.split())
        comparison_key = term.casefold()
        if not term or comparison_key in seen:
            continue
        seen.add(comparison_key)
        normalized.append(term)
        if len(normalized) >= MAX_QUERY_KEY_TERMS:
            break
    return normalized


def _normalize_intent_analysis(
    raw_result: Mapping[str, Any],
    original_query: str,
) -> IntentAnalysis:
    """
    Validate model fields and apply conservative routing defaults.

    Unknown intent labels default to ``legal_retrieval``. This fail-open
    choice is safer for a legal RAG assistant than inventing a direct answer:
    later retrieval-quality checks can still reject weak evidence and use the
    configured network fallback.
    """
    raw_intent = str(raw_result.get("intent") or "").strip()
    intent: IntentName = (
        raw_intent  # type: ignore[assignment]
        if raw_intent in VALID_INTENTS
        else "legal_retrieval"
    )
    rewritten_query = str(
        raw_result.get("rewritten_query") or original_query
    ).strip()
    if not rewritten_query:
        rewritten_query = original_query

    direct_answer = str(raw_result.get("direct_answer") or "").strip()
    if intent == "legal_retrieval":
        # An empty answer is the graph's routing signal to continue to search.
        direct_answer = ""
    elif not direct_answer:
        direct_answer = (
            "Please provide more detail about the legal issue you want "
            "the assistant to address."
            if intent == "clarification_needed"
            else "I can help with questions about legal and regulatory documents."
        )

    return {
        "intent": intent,
        "query_key_terms": _normalize_query_key_terms(
            raw_result.get("query_key_terms")
        ),
        "rewritten_query": rewritten_query,
        "direct_answer": direct_answer,
    }


async def extract_query_key_terms_and_rewrite_query(
    original_query: str,
    history_chats: Sequence[Mapping[str, Any]],
) -> IntentAnalysis:
    """
    Ask the LLM to classify intent and prepare a standalone retrieval query.

    The LLM call uses ``ainvoke`` because the query graph and MongoDB client
    are asynchronous. This avoids blocking FastAPI's event loop while waiting
    for the OpenAI-compatible endpoint.
    """
    history_text = _format_history_for_prompt(history_chats)
    prompt = load_prompt(
        "rewritten_query_and_key_terms",
        history_text=history_text,
        query=original_query,
    )
    llm_client = get_llm_client(json_mode=True)
    response = await llm_client.ainvoke([HumanMessage(content=prompt)])
    parsed = _extract_json_object(response.content)
    result = _normalize_intent_analysis(parsed, original_query)
    logger.info(
        "Intent analysis completed: "
        f"intent={result['intent']}, "
        f"key_terms={result['query_key_terms']}, "
        f"rewritten_query={result['rewritten_query']}"
    )
    return result


async def node_intent_confirm(state: QueryGraphState) -> dict[str, Any]:
    """
    Prepare the user's query and choose direct-answer or retrieval routing.

    Inputs:
        ``session_id`` identifies MongoDB history, ``original_query`` is the
        current user text, and ``is_stream`` controls progress notifications.

    Outputs:
        ``history``, ``query_key_terms``, and ``rewritten_query`` support later
        retrieval and answer generation. ``answer`` stays empty for legal
        questions, causing the graph to continue to ``node_hybrid_query``. A
        clarification or conversational answer causes direct routing to
        ``node_answer_output``.
    """
    # Step 1: Read and validate the required graph inputs, then register this
    # node as running. Invalid input is rejected before external services run.
    function_name = sys._getframe().f_code.co_name
    session_id = str(state.get("session_id") or "").strip()
    original_query = str(state.get("original_query") or "").strip()
    is_stream = bool(state.get("is_stream", False))
    if not session_id:
        raise ValueError("node_intent_confirm requires a non-empty session_id.")
    if not original_query:
        raise ValueError("node_intent_confirm requires a non-empty original_query.")

    logger.info(
        f">>> [{function_name}] started: "
        f"session_id={session_id}, is_stream={is_stream}"
    )
    add_running_task(session_id, function_name, is_stream)

    try:
        # Step 2: Load recent MongoDB messages. Earlier turns allow the LLM to
        # resolve references such as "it", "that law", or "the previous one".
        # The current user message has not been stored yet, so the returned
        # history contains only earlier turns and cannot duplicate the query.
        history_chats = await get_recent_messages(
            session_id=session_id,
            limit=MAX_HISTORY_MESSAGES,
        )

        # Step 3: Classify intent, extract useful legal key terms, and rewrite
        # the current message as a standalone query for later retrieval nodes.
        analysis = await extract_query_key_terms_and_rewrite_query(
            original_query,
            history_chats,
        )
        logger.info(f"result from extract_query_key_terms_and_rewrite_query {analysis}")

        # Step 4: Save the exact user message and the generated query metadata
        # in MongoDB so future follow-up questions can reuse this information.
        # Save the exact user text as authoritative history. The rewritten
        # query and key terms are metadata used to resolve future follow-ups.
        await save_chat_message(
            session_id=session_id,
            role="user",
            text=original_query,
            rewritten_query=analysis["rewritten_query"],
            query_key_terms=analysis["query_key_terms"],
            image_urls=[],
        )

        # Step 5: Build this node's partial LangGraph update. An empty answer
        # continues to retrieval; a direct answer bypasses retrieval.
        update: dict[str, Any] = {
            "history": history_chats,
            "query_key_terms": analysis["query_key_terms"],
            "rewritten_query": analysis["rewritten_query"],
            "answer": analysis["direct_answer"],
        }

        # Step 6: Mark the task complete, record the selected route, and return
        # only the graph-state fields produced by this intent node.
        add_done_task(session_id, function_name, is_stream)
        route = (
            "retrieval"
            if analysis["intent"] == "legal_retrieval"
            else "direct_answer"
        )
        logger.info(
            f">>> [{function_name}] finished: "
            f"session_id={session_id}, route={route}"
        )
        return update
    except Exception:
        logger.exception(
            f">>> [{function_name}] failed: session_id={session_id}"
        )
        raise


async def _manual_test() -> None:
    """Run one integration request against configured MongoDB and LLM APIs."""
    mock_state: QueryGraphState = {  # type: ignore[typeddict-item]
        "session_id": "test_session_001",
        "original_query": (
            "May a news article be quoted in part when the publisher and "
            "author are acknowledged?"
        ),
        "is_stream": False,
    }
    result_state = await node_intent_confirm(mock_state)
    print(json.dumps(result_state, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(_manual_test())
