"""
Generate or deliver the final legal-assistant answer.

Workflow
========

```text
state
  |-- direct intent answer --------------------.
  |-- insufficient retrieval -> safe message -+--> publish answer
  `-- sufficient evidence -> prompt -> LLM ----'        |
                                                        |-- extract images
                                                        |-- save Mongo history
                                                        `-- return final state
```

The node never asks the LLM to answer a legal question after both local and
network retrieval have failed the evidence-quality check. This prevents an
unsupported answer from being generated merely because the graph reached its
terminal answer node.
"""

from __future__ import annotations

import re
import sys
from typing import Any
from urllib.parse import urlparse

from app.clients.mongo_utils import save_chat_message
from app.core.logger import logger
from app.llm.llm_utils import get_llm_client
from app.prompt.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.sse_utils import SSEEvent, push_to_session
from app.utils.task_utils import (
    add_done_task,
    add_running_task,
    set_task_result,
)


MAX_CONTEXT_CHARACTERS = 12_000
MAX_SOURCE_DISCLOSURE_CHARACTERS = 12_000
IMAGE_PATTERN = re.compile(r"!\[.*?\]\((.*?)\)")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")


def _build_evidence_context(documents: list[dict[str, Any]]) -> str:
    """Format reranked evidence without exceeding the answer prompt budget."""
    sections: list[str] = []
    used_characters = 0
    for rank, document in enumerate(documents, start=1):
        text = str(document.get("text") or "").strip()
        if not text:
            continue
        metadata = {
            "source_type": (
                "internal_legal_database"
                if document.get("source") == "local"
                else "open_web"
            ),
            "law_title": document.get("law_title", ""),
            "article_number": document.get("article_number", ""),
            "paragraph_start": document.get("paragraph_start", ""),
            "paragraph_end": document.get("paragraph_end", ""),
            "content_type": document.get("content_type", ""),
            "title": document.get("title", ""),
            "url": document.get("url", ""),
            "score": document.get("score", ""),
        }
        metadata_text = "; ".join(
            f"{key}={value}"
            for key, value in metadata.items()
            if value not in {None, ""}
        )
        section = f"[{rank}] {metadata_text}\n{text}"
        remaining = MAX_CONTEXT_CHARACTERS - used_characters
        if remaining <= 0:
            break
        sections.append(section[:remaining])
        used_characters += min(len(section), remaining)
    return "\n\n".join(sections)


def _markdown_text(value: Any) -> str:
    """Escape minimal Markdown punctuation in a user-visible source label."""
    return str(value or "").strip().replace("[", "\\[").replace(
        "]", "\\]"
    )


def _safe_web_url(value: Any) -> str:
    """Return only an HTTP(S) URL suitable for a Markdown source link."""
    url = str(value or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url.replace("(", "%28").replace(")", "%29")


def _local_source_label(document: dict[str, Any]) -> str:
    """Build one readable law, article/recital, and paragraph citation."""
    law_title = _markdown_text(
        document.get("law_title") or document.get("law_id") or "Legal source"
    )
    parts = [law_title]

    article_number = _markdown_text(document.get("article_number"))
    content_type = str(document.get("content_type") or "").lower()
    if article_number:
        parts.append(f"Article {article_number}")
    elif content_type == "recital":
        match = re.search(
            r"(?:^|:)recital_([^:]+)",
            str(document.get("chunk_id") or ""),
        )
        if match:
            parts.append(f"Recital {_markdown_text(match.group(1))}")

    paragraph_start = _markdown_text(document.get("paragraph_start"))
    paragraph_end = _markdown_text(document.get("paragraph_end"))
    if paragraph_start and paragraph_end and paragraph_end != paragraph_start:
        parts.append(f"Paragraphs/points {paragraph_start}–{paragraph_end}")
    elif paragraph_start:
        parts.append(f"Paragraph/point {paragraph_start}")

    title = _markdown_text(
        document.get("article_title") or document.get("title")
    )
    if title and title.casefold() != law_title.casefold():
        parts.append(title)
    return " — ".join(parts)


def _build_source_disclosure(documents: list[dict[str, Any]]) -> str:
    """Create a deterministic source appendix for the generated answer."""
    local_sources: list[str] = []
    web_sources: list[str] = []
    seen_local: set[str] = set()
    seen_web: set[str] = set()
    disclosed_local_characters = 0

    for document in documents:
        if document.get("source") == "local":
            label = _local_source_label(document)
            if label and label not in seen_local:
                seen_local.add(label)
                entry = f"- **{label}**"
                text = str(document.get("text") or "").strip()
                remaining = (
                    MAX_SOURCE_DISCLOSURE_CHARACTERS
                    - disclosed_local_characters
                )
                if text and remaining > 0:
                    displayed_text = text[:remaining]
                    disclosed_local_characters += len(displayed_text)
                    if len(displayed_text) < len(text):
                        displayed_text = (
                            f"{displayed_text.rstrip()}\n[Retrieved legal "
                            "text truncated for display.]"
                        )
                    quoted_text = "\n".join(
                        f"  > {line}" if line else "  >"
                        for line in displayed_text.splitlines()
                    )
                    entry = (
                        f"{entry}\n\n"
                        "  **Retrieved legal text:**\n\n"
                        f"{quoted_text}"
                    )
                local_sources.append(entry)
            continue
        if document.get("source") != "web":
            continue
        title = _markdown_text(document.get("title") or "Open-web source")
        url = _safe_web_url(document.get("url"))
        label = f"[{title}]({url})" if url else title
        if label not in seen_web:
            seen_web.add(label)
            web_sources.append(label)

    sections: list[str] = []
    if local_sources:
        sections.append(
            "### Internal legal database sources\n"
            + "\n\n".join(local_sources)
        )
    if web_sources:
        sections.append(
            "### Open-web source warning\n"
            "The following material came from an open-web network search. "
            "It may be incomplete, outdated, or inaccurate and must not be "
            "treated as authoritative legal support. Verify it against an "
            "official or consolidated legal text.\n"
            + "\n".join(f"- {source}" for source in web_sources)
        )
    return "\n\n".join(sections)


def _format_history(history: list[dict[str, Any]]) -> str:
    """Convert stored conversation records into compact chronological text."""
    lines: list[str] = []
    for message in history:
        role = str(message.get("role") or "unknown").strip()
        text = str(message.get("text") or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines) or "No previous conversation is available."


def _build_answer_prompt(state: QueryGraphState) -> str:
    """Render the legal answer prompt from evidence and query metadata."""
    question = str(
        state.get("rewritten_query")
        or state.get("original_query")
        or ""
    ).strip()
    if not question:
        raise ValueError("Answer generation requires a non-empty query.")
    return load_prompt(
        "answer_out",
        context=_build_evidence_context(state.get("reranked_docs", [])),
        history=_format_history(state.get("history", [])),
        key_terms=", ".join(state.get("query_key_terms", [])),
        question=question,
    )


def _insufficient_evidence_answer(state: QueryGraphState) -> str:
    """Return a transparent limitation message after the fallback is exhausted."""
    if (
        not state.get("network_fallback_used", False)
        or state.get("retrieval_is_sufficient", False)
    ):
        return ""
    reason = str(
        state.get("retrieval_failure_reason")
        or "No sufficiently relevant evidence was retrieved."
    ).strip()
    return (
        "I could not find sufficient authoritative evidence to answer this "
        f"question reliably. {reason} Please provide a more specific law, "
        "jurisdiction, citation, or factual context."
    )


async def _generate_answer(
    prompt: str,
    *,
    session_id: str,
    is_stream: bool,
) -> str:
    """Generate an answer asynchronously and optionally publish token deltas."""
    model = get_llm_client()
    if not is_stream:
        response = await model.ainvoke(prompt)
        return str(response.content)

    parts: list[str] = []
    async for chunk in model.astream(prompt):
        delta = str(chunk.content or "")
        if not delta:
            continue
        parts.append(delta)
        push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})
    return "".join(parts)


def _extract_image_urls(documents: list[dict[str, Any]]) -> list[str]:
    """Collect unique image URLs from ranked local and network evidence."""
    image_urls: list[str] = []
    seen: set[str] = set()
    for document in documents:
        url = str(document.get("url") or "").strip()
        candidates = []
        if url.lower().endswith(IMAGE_SUFFIXES):
            candidates.append(url)
        candidates.extend(
            IMAGE_PATTERN.findall(str(document.get("text") or ""))
        )
        for candidate in candidates:
            normalized = candidate.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                image_urls.append(normalized)
    return image_urls


async def node_answer_output(state: QueryGraphState) -> dict[str, Any]:
    """Publish a direct, evidence-grounded, or insufficient-evidence answer."""
    # Step 1: Validate identifiers and register final-answer task progress.
    function_name = sys._getframe().f_code.co_name
    session_id = str(state.get("session_id") or "").strip()
    is_stream = bool(state.get("is_stream", False))
    if not session_id:
        raise ValueError("node_answer_output requires a non-empty session_id.")
    add_running_task(session_id, function_name, is_stream)

    try:
        # Step 2: Prefer an intent node's direct response, then an explicit
        # insufficient-evidence message, and call the LLM only when needed.
        answer = str(state.get("answer") or "").strip()
        answer_was_streamed = False
        generated_from_evidence = False
        if not answer:
            answer = _insufficient_evidence_answer(state)
        if not answer:
            prompt = _build_answer_prompt(state)
            answer = await _generate_answer(
                prompt,
                session_id=session_id,
                is_stream=is_stream,
            )
            answer_was_streamed = is_stream
            generated_from_evidence = True

        # Step 3: Append a deterministic source section. This is implemented
        # in code rather than relying only on the LLM: local evidence always
        # identifies its law and provision, while web evidence always carries
        # a warning and includes its available links.
        if generated_from_evidence:
            disclosure = _build_source_disclosure(
                state.get("reranked_docs", [])
            )
            if disclosure:
                disclosure_delta = f"\n\n{disclosure}"
                answer = f"{answer.rstrip()}{disclosure_delta}"
                if is_stream:
                    push_to_session(
                        session_id,
                        SSEEvent.DELTA,
                        {"delta": disclosure_delta},
                    )

        # Step 4: Publish the answer through the configured non-streaming or
        # SSE channel. Generated streaming deltas are not sent a second time.
        odrl_state = state.get("odrl_state") or {}
        raw_policy_rules = odrl_state.get("validated_policy_rules")
        if not raw_policy_rules and state.get("validated_policy_rules"):
            raw_policy_rules = state.get("validated_policy_rules", [])
        validated_policy_rules = [
            dict(rule)
            for rule in raw_policy_rules
            if isinstance(rule, dict)
        ]
        odrl_document = odrl_state.get("odrl_document")
        if is_stream and not answer_was_streamed:
            push_to_session(session_id, SSEEvent.DELTA, {"delta": answer})
        if not is_stream:
            set_task_result(session_id, "answer", answer)
            set_task_result(
                session_id,
                "validated_policy_rules",
                validated_policy_rules,
            )
            set_task_result(session_id, "odrl_document", odrl_document)

        # Step 5: Extract images only from retrieved evidence and expose them
        # through graph state and the terminal streaming event.
        image_urls = _extract_image_urls(state.get("reranked_docs", []))
        if is_stream:
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                {
                    "answer": answer,
                    "status": "completed",
                    "image_urls": image_urls,
                    "validated_policy_rules": validated_policy_rules,
                    "odrl_document": odrl_document,
                },
            )

        # Step 6: Persist the assistant response with legal query metadata so
        # future intent rewriting has complete chronological context.
        await save_chat_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            rewritten_query=str(
                state.get("rewritten_query")
                or state.get("original_query")
                or ""
            ),
            query_key_terms=state.get("query_key_terms", []),
            image_urls=image_urls,
            validated_policy_rules=validated_policy_rules,
            odrl_document=(
                odrl_document if isinstance(odrl_document, dict) else None
            ),
        )

        # Step 7: Complete task tracking and return this node's owned fields.
        add_done_task(session_id, function_name, is_stream)
        logger.info(
            f">>> [{function_name}] finished: session_id={session_id}, "
            f"answer_characters={len(answer)}, images={len(image_urls)}"
        )
        return {
            "answer": answer,
            "image_urls": image_urls,
            "validated_policy_rules": validated_policy_rules,
            "odrl_state": odrl_state,
        }
    except Exception:
        logger.exception(
            f">>> [{function_name}] failed: session_id={session_id}"
        )
        raise
