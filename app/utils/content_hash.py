from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any


HASH_ALGORITHM = "sha256"


def normalize_text_for_hash(text: str) -> str:
    """
    Canonicalize text before hashing without changing its legal wording.

    Line endings and trailing horizontal whitespace are formatting details that
    should not create a new document version. Internal words, punctuation, and
    paragraph boundaries remain unchanged because they can carry legal meaning.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(
        line.rstrip()
        for line in normalized.split("\n")
    )
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def sha256_text(text: str) -> str:
    """Return a lowercase SHA-256 hexadecimal digest for canonical text."""
    normalized = normalize_text_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def calculate_document_hash(md_content: str) -> str:
    """Fingerprint the normalized authoritative Markdown document."""
    if not isinstance(md_content, str) or not md_content.strip():
        raise ValueError("Cannot hash an empty legal document.")
    return sha256_text(md_content)


def calculate_chunk_hash(chunk: Mapping[str, Any]) -> str:
    """
    Fingerprint one legal record using identity, structure, and normalized text.

    Including legal identity metadata prevents identical wording in two
    different provisions from being treated as the same record.
    """
    required_fields = (
        "chunk_id",
        "parent_id",
        "law_id",
        "record_type",
        "content_type",
        "text",
    )
    structural_fields = (
        "law_title",
        "chapter_number",
        "chapter_title",
        "article_number",
        "article_title",
        "paragraph_start",
        "paragraph_end",
        "language",
        "status",
    )
    missing_fields = [
        field_name
        for field_name in required_fields
        if not chunk.get(field_name)
    ]
    if missing_fields:
        raise ValueError(
            "Cannot calculate chunk hash; missing fields: "
            + ", ".join(missing_fields)
        )

    payload = {
        field_name: (
            normalize_text_for_hash(str(chunk[field_name]))
            if field_name == "text"
            else str(chunk[field_name]).strip()
        )
        for field_name in required_fields
    }
    payload.update(
        {
            field_name: str(chunk[field_name]).strip()
            for field_name in structural_fields
            if chunk.get(field_name) is not None
        }
    )
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
