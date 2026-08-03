from __future__ import annotations

from datasets import info

"""
LangGraph node for enriching law-document child chunks with legal key terms.

Position in the ingestion workflow:

    node_document_split
        -> node_key_terms_extraction
        -> node_import_es
        -> node_bge_embedding
        -> node_import_milvus

``node_document_split`` creates complete parent records and smaller child
records. This node sends only the child records to an LLM because children are
the units searched by Elasticsearch and Milvus. The extracted terms are a
retrieval aid; they never replace the original legal text.

The node updates ``state["chunks"]`` in place and also writes a human-readable
``chunks_with_keywords.json`` backup for later inspection.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logger import logger
from app.import_process.agent.state import ChunkRecord, ImportGraphState
from app.llm.llm_utils import get_llm_client
from app.utils.task_utils import add_done_task, add_running_task


# Send several child chunks in one request instead of making one API request per
# chunk. This reduces latency and API overhead while keeping each prompt small
# enough for normal LLM context windows.
KEYWORD_BATCH_SIZE = 8

# If the first response omits chunk IDs or is malformed, retry only unresolved
# chunks in smaller requests. Smaller prompts and responses reduce the chance
# that the model reaches its output-token limit before returning every ID.
KEYWORD_RETRY_BATCH_SIZE = 2

# Extraction uses at most three rounds:
# 1. the normal batch;
# 2. unresolved chunks in pairs; and
# 3. any still-unresolved chunks individually.
MAX_KEYWORD_EXTRACTION_ATTEMPTS = 3

# The prompt asks for 3-8 terms, and this local limit protects downstream fields
# if the LLM ignores that instruction and returns too many values.
MAX_KEY_TERMS = 8

# Reject unexpectedly long phrases. A key term should be a concise retrieval
# phrase, not a sentence or a copied paragraph.
MAX_KEY_TERM_LENGTH = 100

# Legal-location references describe where text appears rather than what the
# text means. Even a reference written without digits, such as "Article twelve",
# is therefore removed from retrieval terms.
LEGAL_LOCATION_REFERENCE_RE = re.compile(
    r"\b(?:"
    r"articles?|recitals?|paragraphs?|points?|sections?|chapters?|"
    r"annex(?:es)?"
    r")\b",
    re.IGNORECASE,
)


# The system prompt establishes the LLM's role, the exact response schema, and
# quality rules. ``json_mode=True`` helps produce JSON, but explicit instructions
# are still necessary because JSON mode does not guarantee our required keys.
SYSTEM_PROMPT = """
You extract retrieval key terms from European Union law and regulation text.

Return only a JSON object with this exact structure:
{
  "chunks": [
    {
      "chunk_id": "the input chunk_id",
      "key_terms": ["term one", "term two"]
    }
  ]
}

Rules:
1. Return one result for every input chunk_id.
2. Extract 3 to 8 concise, legally meaningful terms when the text supports them.
3. Prefer exact legal phrases, regulated entities, rights, duties, procedures,
   legal tests, exceptions, and defined terms found in the text.
4. Do not return generic words such as "law", "article", "paragraph",
   "regulation", "data", or "information" unless they are part of a specific
   legal phrase.
5. Do not invent legal concepts that are absent from the supplied text.
6. Do not return legal citations, instrument identifiers, provision references,
   dates, years, deadlines, or numbered ranges. Forbidden examples include
   "Directive (EU) 2016/680", "Directive 2000/31/EC", "Articles 12 to 15",
   "Article 6", "Recital 42", "26 April 2016", and "30 days".
7. A key term must not contain digits. Extract the underlying legal concept
   instead, but only when that concept is explicitly supported by the text.
   For example, prefer "data subject access rights" over "Articles 12 to 15".
8. Do not use a cited law's formal title as a key term. For example, do not
   return "Directive 2011/24/EU of the European Parliament and of the Council".
9. Do not include explanations, citations, numbering, or Markdown.
10. If a chunk contains no useful legal key term, return an empty list.
""".strip()


def step_1_get_chunks_for_extraction(state: ImportGraphState) -> tuple[list[ChunkRecord], list[ChunkRecord]]:
    """
    Validate the splitter output and select child chunks that still need terms.

    Parent records retain the complete Article or other high-level legal unit,
    but retrieval is performed against child records. Consequently, the LLM is
    called only for children and existing non-empty key terms are preserved when
    the node is run again.
    """
    # The previous splitter node stores every parent and child in this list.
    # ``get`` provides a clearer validation error than an immediate KeyError.
    chunks = state.get("chunks", [])

    # Reject an invalid container early. Later code assumes list iteration and
    # would otherwise fail with a less useful error.
    if not isinstance(chunks, list):
        raise TypeError("state['chunks'] must be a list.")

    # An empty list usually means the workflow skipped or failed the document
    # splitting node, so there is nothing meaningful to enrich.
    if not chunks:
        raise ValueError("state['chunks'] is empty; run node_document_split first.")

    # This list contains only children that require an LLM call. ``chunks`` still
    # contains every record and will be returned unchanged except for key terms.
    pending_children: list[ChunkRecord] = []
    for chunk in chunks:
        # ChunkRecord is a TypedDict for templates typing, but runtime input can
        # still be malformed, so validate each item before using dictionary APIs.
        if not isinstance(chunk, dict):
            raise TypeError("Every item in state['chunks'] must be a dictionary.")

        # Normalize terms on every run, including terms loaded from an older
        # chunks_with_keywords.json backup. This removes citation-heavy values
        # produced before the current filtering rules were added.
        chunk["key_terms"] = _normalise_key_terms(chunk.get("key_terms", []))

        # A chunk is pending only when:
        # 1. it is a searchable child rather than a complete parent;
        # 2. it contains legal text for the LLM to inspect; and
        # 3. it has not already been enriched.
        #
        # The third condition makes this node idempotent: rerunning it does not
        # spend API calls replacing existing terms.
        if (
            chunk.get("record_type") == "child"
            and chunk.get("text", "").strip()
            and not chunk["key_terms"]
        ):
            # ``chunk_id`` is how results are matched back to chunks. Without it,
            # assigning an LLM result safely would be impossible.
            if not chunk.get("chunk_id", "").strip():
                raise ValueError("A child chunk is missing chunk_id.")
            pending_children.append(chunk)

    # Return both collections: the complete list is eventually stored in state,
    # while the smaller list controls which chunks are sent to the LLM.
    return chunks, pending_children


def _build_user_prompt(batch: list[ChunkRecord]) -> str:
    """Build one compact JSON payload for a batch of child chunks."""
    # Send only fields that help term extraction. Other metadata, such as vector
    # values or source paths, would increase prompt size without improving terms.
    payload = {
        "chunks": [
            {
                # The model must echo this stable identifier so we can map each
                # result back even if it changes the output order.
                "chunk_id": chunk["chunk_id"],
                # These fields give useful legal context, especially when the
                # chunk text begins in the middle of a larger Article.
                "content_type": chunk.get("content_type", ""),
                "article_number": chunk.get("article_number", ""),
                "article_title": chunk.get("article_title", ""),
                # This is the authoritative source from which terms are derived.
                "text": chunk["text"],
            }
            for chunk in batch
        ]
    }

    # ``ensure_ascii=False`` keeps legal names and non-ASCII punctuation readable
    # instead of converting them to sequences such as ``\u2018``.
    return (
        "Extract key terms from every legal chunk in this JSON input. "
        "Keep each chunk_id unchanged.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _response_content_to_text(content: Any) -> str:
    """Normalize common LangChain text and content-block response formats."""
    # Most OpenAI-compatible chat models return a plain string.
    if isinstance(content, str):
        return content.strip()

    # Some providers return content as blocks, for example:
    # [{"type": "text", "text": "{\"chunks\": [...]}"}].
    # Join all recognized text blocks before parsing the JSON.
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        return "\n".join(text_parts).strip()

    # This fallback converts unusual response types into text so the JSON parser
    # can produce one consistent, understandable validation error.
    return str(content or "").strip()


def _parse_json_object(content: Any) -> dict[str, Any]:
    """Parse JSON-mode output while tolerating an accidental Markdown fence."""
    # First normalize provider-specific response formats to one text string.
    text = _response_content_to_text(content)

    # JSON mode should return bare JSON, but some models still wrap it in
    # `````json`` fences. Remove only an outer fence, not content inside JSON.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    # Convert the model output to normal Python objects. Re-raise parsing errors
    # as ValueError so callers do not depend on json.JSONDecodeError internals.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc.msg}") from exc

    # The required top-level schema is an object. A valid JSON list is still the
    # wrong contract and must not be accepted silently.
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object.")
    return parsed


def _normalise_key_terms(value: Any) -> list[str]:
    """
    Validate, filter, deduplicate, and limit one chunk's extracted terms.

    Terms containing digits or legal-location words are removed because they
    are usually citations, dates, deadlines, or provision references rather
    than semantic concepts likely to appear in a user's policy question.
    """
    # A malformed value such as a string or object is treated as no terms. This
    # prevents accidentally iterating over individual characters in a string.
    if not isinstance(value, list):
        return []

    terms: list[str] = []

    # Store case-folded values separately so variants such as "Data Subject" and
    # "data subject" are considered duplicates while preserving the first term's
    # original capitalization in ``terms``.
    seen: set[str] = set()
    for item in value:
        # Ignore numbers, nulls, nested objects, and other invalid list members.
        if not isinstance(item, str):
            continue

        # Collapse repeated whitespace and remove common punctuation that models
        # sometimes place around list items.
        term = re.sub(r"\s+", " ", item).strip(" \t\r\n,;.")
        normalised = term.casefold()

        # Reject numeric content broadly. This covers years, dates, durations,
        # instrument numbers, Article numbers, and ranges such as "12 to 15".
        contains_digit = bool(re.search(r"\d", term))

        # Reject Article/recital/paragraph-style references even if their number
        # is written as a word instead of a digit.
        contains_legal_location = bool(LEGAL_LOCATION_REFERENCE_RE.search(term))

        # Reject empty strings, sentence-like oversized output, citations, and
        # duplicates before they can enter Elasticsearch or Milvus metadata.
        if (
            not term
            or len(term) > MAX_KEY_TERM_LENGTH
            or contains_digit
            or contains_legal_location
            or normalised in seen
        ):
            continue

        terms.append(term)
        seen.add(normalised)

        # Stop at the configured schema limit even if the LLM returned more.
        if len(terms) >= MAX_KEY_TERMS:
            break
    return terms


def step_2_extract_batch_key_terms(
    llm_client: Any,
    batch: list[ChunkRecord],
) -> tuple[dict[str, list[str]], set[str]]:
    """
    Call the LLM once and return both valid and omitted chunk results.

    Missing IDs are not raised as an exception because otherwise one incomplete
    response would discard valid terms returned for the rest of the batch.
    Malformed JSON or an invalid top-level schema still raises an exception so
    the retry coordinator can resend those chunks in smaller requests.
    """
    # Capture the exact allowed IDs before calling the model. This blocks unknown
    # or hallucinated IDs from being written into our records.
    expected_ids = {chunk["chunk_id"] for chunk in batch}

    # SystemMessage contains stable role/schema rules. HumanMessage contains only
    # the current batch, so the same system instructions can serve every call.
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_user_prompt(batch)),
    ]

    # ``invoke`` is synchronous. The returned object is normally an AIMessage
    # whose ``content`` field contains the JSON response.
    response = llm_client.invoke(messages)

    # Provider metadata often exposes ``finish_reason="length"`` when output was
    # truncated. Retain it in debug logs to distinguish truncation from ordinary
    # instruction-following inconsistency without depending on one provider.
    response_metadata = getattr(response, "response_metadata", None)
    if response_metadata:
        finish_reason = (
            response_metadata.get("finish_reason")
            or response_metadata.get("stop_reason")
        )
        logger.debug(
            "Key-term LLM response metadata: "
            f"finish_reason={finish_reason}, metadata={response_metadata}"
        )

    parsed = _parse_json_object(response.content)

    # The top-level object must contain the list described by SYSTEM_PROMPT.
    result_items = parsed.get("chunks")
    if not isinstance(result_items, list):
        raise ValueError("LLM response is missing the 'chunks' list.")

    # Build a mapping by ID instead of relying on response order. LLMs sometimes
    # reorder items even when asked to preserve input order.
    extracted: dict[str, list[str]] = {}
    for item in result_items:
        # Skip malformed list members instead of allowing one bad member to hide
        # otherwise valid results in this batch.
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id")

        # Ignore invented IDs and duplicate responses. The first valid result for
        # each expected ID wins deterministically.
        if chunk_id not in expected_ids or chunk_id in extracted:
            continue

        # Validate and clean terms before they can enter graph state or storage.
        extracted[chunk_id] = _normalise_key_terms(item.get("key_terms"))

    # An explicit empty key_terms list still counts as a valid result. Only IDs
    # entirely absent from the response are retried.
    missing_ids = expected_ids - extracted.keys()
    return extracted, missing_ids


def _split_retry_batches(
    chunks: list[ChunkRecord],
    batch_size: int,
) -> list[list[ChunkRecord]]:
    """Split unresolved chunks into ordered retry groups without overlap."""
    return [
        chunks[start:start + batch_size]
        for start in range(0, len(chunks), batch_size)
    ]


def _extract_batch_with_retries(
    llm_client: Any,
    batch: list[ChunkRecord],
    batch_number: int,
) -> tuple[dict[str, list[str]], list[str]]:
    """
    Preserve partial results and retry only chunks that remain unresolved.

    The first round sends the original batch. The second sends unresolved
    chunks in pairs, and the final round sends them individually. This bounded
    strategy improves completeness without creating an unbounded retry loop or
    repeatedly spending tokens on chunks that already succeeded.
    """
    chunks_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in batch
    }
    ordered_chunk_ids = [
        chunk["chunk_id"]
        for chunk in batch
    ]
    extracted: dict[str, list[str]] = {}
    last_errors: list[str] = []

    for attempt in range(1, MAX_KEYWORD_EXTRACTION_ATTEMPTS + 1):
        unresolved_chunks = [
            chunks_by_id[chunk_id]
            for chunk_id in ordered_chunk_ids
            if chunk_id not in extracted
        ]
        if not unresolved_chunks:
            break

        if attempt == 1:
            retry_batches = [unresolved_chunks]
        elif attempt == MAX_KEYWORD_EXTRACTION_ATTEMPTS:
            retry_batches = _split_retry_batches(unresolved_chunks, 1)
        else:
            retry_batches = _split_retry_batches(
                unresolved_chunks,
                KEYWORD_RETRY_BATCH_SIZE,
            )

        for retry_batch in retry_batches:
            retry_ids = [
                chunk["chunk_id"]
                for chunk in retry_batch
            ]
            try:
                retry_results, missing_ids = step_2_extract_batch_key_terms(
                    llm_client,
                    retry_batch,
                )
                extracted.update(retry_results)

                if missing_ids:
                    logger.warning(
                        f"Key-term extraction batch {batch_number}, "
                        f"attempt {attempt} omitted chunk IDs: "
                        + ", ".join(sorted(missing_ids))
                    )
            except Exception as exc:
                error = (
                    f"attempt {attempt} for "
                    f"{', '.join(retry_ids)}: {type(exc).__name__}: {exc}"
                )
                last_errors.append(error)
                logger.warning(
                    f"Key-term extraction batch {batch_number} {error}"
                )

    failed_ids = [
        chunk_id
        for chunk_id in ordered_chunk_ids
        if chunk_id not in extracted
    ]
    if failed_ids and last_errors:
        logger.debug(
            f"Key-term extraction batch {batch_number} retry errors: "
            + " | ".join(last_errors)
        )
    return extracted, failed_ids


def step_3_update_chunks_with_key_terms(
    pending_children: list[ChunkRecord],
    llm_client: Any,
) -> tuple[int, list[str]]:
    """
    Enrich child chunks batch by batch.

    Valid partial results are preserved immediately. Omitted or failed chunks
    are retried in smaller requests, and only chunks unresolved after all
    bounded attempts retain empty terms. Because key terms are optional
    retrieval enrichment, final failures do not discard authoritative legal
    text or stop the complete import pipeline.
    """
    # Number of child records that received at least one usable term. This is a
    # chunk count, not the total number of individual terms.
    extracted_count = 0

    # Keep batch-level error descriptions for the final warning summary.
    failed_batches: list[str] = []

    # ``range`` advances by batch size: 0, 8, 16, ... . Slicing safely handles a
    # final batch containing fewer than eight chunks.
    for start in range(0, len(pending_children), KEYWORD_BATCH_SIZE):
        batch = pending_children[start:start + KEYWORD_BATCH_SIZE]

        # Convert the zero-based list offset into a one-based number for logs.
        batch_number = start // KEYWORD_BATCH_SIZE + 1
        terms_by_chunk_id, failed_ids = _extract_batch_with_retries(
            llm_client,
            batch,
            batch_number,
        )

        # Chunk dictionaries are shared with ``state["chunks"]``. Assigning here
        # therefore enriches the original state records without rebuilding them.
        for chunk in batch:
            chunk_id = chunk["chunk_id"]
            if chunk_id not in terms_by_chunk_id:
                continue

            key_terms = terms_by_chunk_id[chunk_id]
            # logger.info(f"\noriginal text: {chunk['text']}")
            # logger.info(f"extracting key terms for the batch chunk: {key_terms}")
            chunk["key_terms"] = key_terms

            # An explicit empty list is valid but does not count as enriched.
            if key_terms:
                extracted_count += 1

        if failed_ids:
            error = (
                "LLM response still omitted chunk IDs after retries: "
                + ", ".join(failed_ids)
            )
            failed_batches.append(f"batch {batch_number}: {error}")
            logger.error(
                f"Key-term extraction partially failed for batch "
                f"{batch_number}: {error}"
            )

        logger.info(
            f"Key-term extraction batch {batch_number} completed: "
            f"chunks={len(batch)}, "
            f"resolved={len(terms_by_chunk_id)}, "
            f"failed={len(failed_ids)}"
        )

    # Return statistics separately from chunks because chunks were mutated in
    # place and are already available to the caller.
    return extracted_count, failed_batches


def step_4_backup_chunks_with_keywords(
    state: ImportGraphState,
    chunks: list[ChunkRecord],
) -> str:
    """Write enriched chunks to ``chunks_with_keywords.json`` for inspection."""
    # Prefer the workflow's output directory. If it is absent, save beside the
    # Markdown source so the backup remains associated with its document.
    local_dir = state.get("local_dir") or str(Path(state.get("md_path", ".")).parent)

    # ``exist_ok=True`` supports both new output directories and repeated runs.
    os.makedirs(local_dir, exist_ok=True)
    backup_file_path = os.path.join(local_dir, "chunks_with_keywords.json")

    # UTF-8 plus ``ensure_ascii=False`` preserves legal names and punctuation.
    # Indentation makes this debugging artifact easy to review manually.
    with open(backup_file_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=4)

    logger.info(f"Backed up chunks with key terms to: {backup_file_path}")
    return backup_file_path


def node_key_terms_extraction(state: ImportGraphState) -> ImportGraphState:
    """
    Extract legal key terms for child chunks and update the graph state.

    The orchestration sequence is:

    1. Mark this LangGraph task as running.
    2. Validate chunks and select unenriched children.
    3. Reuse one JSON-mode LLM client for all batches.
    4. Add validated ``key_terms`` directly to child dictionaries.
    5. Save ``chunks_with_keywords.json``.
    6. Mark this task as done and return the updated state.
    """
    # Discover the current function name instead of duplicating a string in logs
    # and task tracking. A future rename will therefore remain consistent.
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] Started. Chunk count: {len(state.get('chunks', []))}")

    # Task utilities expose node progress to the rest of the application.
    add_running_task(state["task_id"], function_name)

    # ``chunks`` is the full state list; ``pending_children`` is the subset that
    # needs LLM enrichment.
    chunks, pending_children = step_1_get_chunks_for_extraction(state)

    # Defaults also cover the idempotent case where every child already has
    # terms and no LLM client needs to be initialized.
    extracted_count = 0
    failed_batches: list[str] = []

    if pending_children:
        # JSON mode requests a structured JSON object from the OpenAI-compatible
        # provider. The utility caches this client for subsequent calls.
        llm_client = get_llm_client(json_mode=True)
        extracted_count, failed_batches = step_3_update_chunks_with_key_terms(
            pending_children,
            llm_client,
        )
    else:
        logger.info("No child chunks require key-term extraction.")

    # Reassign the list explicitly for clarity in LangGraph state transitions,
    # even though child dictionaries were updated in place.
    state["chunks"] = chunks

    # Always create a backup, including idempotent runs and partially failed
    # runs, so the user can inspect the exact state sent to downstream nodes.
    backup_file_path = step_4_backup_chunks_with_keywords(state, chunks)

    # Batch failures do not stop ingestion, but they remain visible in logs.
    if failed_batches:
        logger.warning(
            "Key-term extraction completed with failed batches: "
            + " | ".join(failed_batches)
        )

    logger.info(
        f">>> [{function_name}] Finished. "
        f"pending_children={len(pending_children)}, "
        f"enriched_children={extracted_count}, "
        f"failed_batches={len(failed_batches)}, "
        f"backup={backup_file_path}"
    )

    # Moving the node from running to done happens after the backup succeeds.
    add_done_task(state["task_id"], function_name)
    return state


def test_node_key_terms_extraction_with_split_example() -> None:
    """
    Manual integration test using the GDPR example from node_document_split.

    This test calls the configured external LLM and may incur API usage. Run it
    directly only when the API configuration is available.
    """
    from app.import_process.agent.nodes.node_document_split import (
        node_document_split,
    )
    from app.utils.path_util import PROJECT_ROOT

    # Reuse the same real GDPR Markdown example used by the document-splitter
    # manual integration test.
    test_md_path = os.path.join(
        PROJECT_ROOT,
        "output",
        "General_Data_Protection_Regulation_GDPR",
        "General_Data_Protection_Regulation_GDPR.md",
    )
    if not os.path.exists(test_md_path):
        raise FileNotFoundError(f"GDPR test Markdown does not exist: {test_md_path}")

    # The splitter reads ``md_path`` because ``md_content`` starts empty, writes
    # ``chunks.json``, and places generated records in state["chunks"].
    test_state: ImportGraphState = {
        "task_id": "test_key_terms_extraction",
        "md_path": test_md_path,
        "md_content": "",
        "file_title": "General_Data_Protection_Regulation_GDPR",
        "local_dir": os.path.dirname(test_md_path),
    }

    # Run the same two-node order used by the real LangGraph workflow.
    split_state = node_document_split(test_state)
    final_state = node_key_terms_extraction(split_state)

    # Count only child chunks because parent chunks intentionally keep empty
    # key_terms in this design.
    children = [
        chunk
        for chunk in final_state["chunks"]
        if chunk.get("record_type") == "child"
    ]
    logger.info(
        f"Manual key-term test completed: children={len(children)}, "
        f"children_with_terms={sum(bool(chunk.get('key_terms')) for chunk in children)}"
    )


if __name__ == "__main__":
    # Running this module directly performs a real LLM integration test and
    # writes chunks_with_keywords.json beside the GDPR example Markdown.
    test_node_key_terms_extraction_with_split_example()
