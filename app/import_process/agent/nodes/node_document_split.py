"""
Structure-aware splitter for European Union law and regulation Markdown.

Workflow
========

The node runs four explicit steps:

    ImportGraphState
        |
        |-- step_1_get_content
        |     Read Markdown from state or disk, normalize line endings, and
        |     determine the document title used for metadata.
        |
        |-- step_2_split_law_document
        |     Scan the document in source order and recognize recitals,
        |     chapters/titles/parts, Articles, numbered paragraphs, lettered
        |     points, definitions, and Annexes.
        |
        |     Each legal unit produces:
        |       parent = the complete Article, recital group, or Annex
        |       child  = the smallest useful retrieval unit
        |
        |-- step_3_refine_chunks
        |     Normalize text and validate IDs, record types, uniqueness, and
        |     parent-child relationships. This step does not split text again.
        |
        |-- step_4_add_content_hashes
        |     Add a shared document hash and one structural hash per chunk.
        |
        |-- step_5_backup_chunks
        |     Save the final records as a timestamped chunks_*.json backup.
        |
        v
    state["chunks"] -> key-term extraction -> Elasticsearch / Milvus

Design principle
================

Legal structure is the primary splitting rule. Token counts are safety limits,
not arbitrary cut points. The splitter prefers to preserve a complete numbered
paragraph or group of lettered points instead of cutting through a legal
sentence. Parent records provide interpretation context; child records provide
precise retrieval targets.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.import_process.agent.state import ImportGraphState, ChunkRecord
from app.core.logger import logger
from app.utils.content_hash import (
    calculate_chunk_hash,
    calculate_document_hash,
)
from app.utils.task_utils import add_running_task, add_done_task


# Desired size for unnumbered fallback content such as Annex paragraphs. Article
# children do not attempt to reach this size: their legal structure comes first.
CHILD_TARGET_TOKENS = 512

# Generic unnumbered/Annex grouping stops before adding another complete block
# above this size. One indivisible block may still exceed the limit.
CHILD_HARD_MAX_TOKENS = 700

# Prefer smaller point groups because the governing paragraph introduction is
# repeated in every resulting child. Paragraphs at or below the maximum remain
# intact even when they contain many lettered points.
POINT_GROUP_TARGET_TOKENS = 400
POINT_GROUP_MAX_TOKENS = 550

# Ordinary paragraphs are children by default. Only consecutive, plain, very
# short paragraphs are combined, and then only into a compact retrieval unit.
TINY_PARAGRAPH_MAX_TOKENS = 60
TINY_PARAGRAPH_GROUP_MAX_TOKENS = 250
TINY_PARAGRAPH_GROUP_MAX_BLOCKS = 3

# Recital parents may contain up to five individual recital children.
RECITAL_GROUP_SIZE = 5


@dataclass
class NumberedBlock:
    """
    One parsed legal unit before it becomes a final ChunkRecord.

    Examples include paragraph ``1``, definition ``4``, or a grouped point
    range such as ``1(a)-1(c)``. ``text`` retains the legal wording.
    """

    label: str
    text: str
    unit_type: str = "paragraph"


@dataclass
class LegalContext:
    """Document-level metadata copied into every generated parent and child."""

    law_id: str
    law_title: str
    source_file: str
    language: str = "en"
    status: str = "in_force"


# Top-level legal structure markers. ``_strip_heading`` is called before these
# expressions, so they work with both ``Article 6`` and ``## Article 6``.
ARTICLE_MARKER_RE = re.compile(r"^Article\s+(?P<number>[0-9]+[A-Za-z]?)$", re.IGNORECASE)
CHAPTER_MARKER_RE = re.compile(r"^(?P<kind>CHAPTER|TITLE|PART)\s+(?P<number>[IVXLCDM]+|[0-9]+|[A-Z])\b.*$", re.IGNORECASE)
SECTION_MARKER_RE = re.compile(r"^Section\s+[0-9]+[A-Za-z]?$", re.IGNORECASE)

# Internal legal-unit markers used inside an Article or recital section.
NUMBERED_PARAGRAPH_RE = re.compile(r"^(?P<label>[0-9]+)\.\s+")
PAREN_NUMBER_RE = re.compile(r"^\((?P<label>[0-9]+)\)\s+")
LETTER_POINT_RE = re.compile(r"^\((?P<label>[a-z])\)\s+", re.IGNORECASE)


def step_1_get_content(state: ImportGraphState) -> tuple[str, str]:
    """
    Read and normalize the Markdown input used by the legal parser.

    The graph can provide Markdown in two ways:

    1. ``state["md_content"]`` already contains text from an earlier node.
    2. ``state["md_path"]`` points to a Markdown file that must be read.

    In-memory content has priority because an image-processing node may have
    rewritten links without replacing the original source file.

    Returns:
        A tuple of normalized Markdown content and a document/file title.
    """
    # Prefer content already produced by an upstream graph node.
    md_content = state.get("md_content", "")

    # Fall back to the Markdown path only when state contains no text.
    if not md_content and state.get("md_path"):
        md_path = Path(state["md_path"])
        if md_path.exists():
            md_content = md_path.read_text(encoding="utf-8")

            # Store disk content back in state so downstream nodes use the same
            # exact source that this splitter parsed.
            state["md_content"] = md_content

    # Splitting an empty document would create invalid empty records, so fail
    # before any parsing or backup file is attempted.
    if not md_content:
        logger.error("[step_1_get_content] empty Markdown content; cannot split document.")
        raise ValueError("md_content is empty and md_path could not be read.")

    # Normalize every platform's line ending to ``\n``:
    # - Windows ``\r\n`` becomes ``\n``;
    # - old Mac ``\r`` becomes ``\n``;
    # - Linux/current macOS ``\n`` remains unchanged.
    #
    # Parsing by lines is now deterministic on every operating system.
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

    # Use explicitly supplied metadata when available; otherwise infer the title
    # from a source path.
    file_title = state.get("file_title") or _file_title_from_state(state)
    return md_content, file_title


def _file_title_from_state(state: ImportGraphState) -> str:
    """Infer a filename without its extension from available source paths."""
    # ``md_path`` is preferred because it may point to rewritten Markdown created
    # after PDF conversion or image processing.
    for key in ("md_path", "local_file_path"):
        path = state.get(key, "")
        if path:
            return Path(path).stem

    # This stable fallback prevents empty identifiers if neither path is present.
    return "default_file"


def _strip_heading(line: str) -> str:
    return re.sub(r"^\s*#{1,6}\s+", "", line.strip()).strip()


def _is_heading(line: str) -> bool:
    return bool(re.match(r"^\s*#{1,6}\s+\S", line))


def _clean_text(text: str) -> str:
    """Normalize horizontal whitespace and excessive blank lines."""
    # Preserve paragraph boundaries while removing noisy repeated spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Keep at most one blank line between legal blocks.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _token_count(text: str) -> int:
    """
    Estimate token size without depending on one embedding/model tokenizer.

    Words and punctuation marks are counted separately. This is only a stable
    grouping heuristic; it is not an exact OpenAI or BGE token count.
    """
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _slugify(value: str) -> str:
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def _infer_law_id(file_title: str, source_file: str = "") -> str:
    """
    Map known law filenames to stable cross-storage identifiers.

    The same ``law_id`` is later stored in Elasticsearch and Milvus, allowing
    retrieval to filter one law or search all laws together.
    """
    # Search both title and path because either may contain the recognizable law
    # name while the other contains a shortened or generated filename.
    value = f"{file_title} {source_file}".lower()
    if "gdpr" in value or "general_data_protection_regulation" in value:
        return "gdpr"
    if "eu_data_act" in value or "data_act" in value or "data act" in value:
        return "eu_data_act"
    if "data_governance_act" in value or "data governance" in value:
        return "data_governance_act"
    if "eu_ai_act" in value or "ai_act" in value or "artificial intelligence act" in value:
        return "eu_ai_act"
    if "free_flow" in value or "non_personal_data" in value or "non-personal data" in value:
        return "free_flow_non_personal_data"

    # Unknown laws still receive a deterministic ID derived from their filename.
    return _slugify(file_title)


def _infer_law_title(md_content: str, file_title: str) -> str:
    """Find a likely formal regulation title near the start of the document."""
    # Covers and formal titles normally occur near the beginning. Limiting the
    # scan avoids accidentally using text from a later Article.

    # for line in md_content.split("\n")[:80]:
    #     title = _strip_heading(line)
    #     if not title:
    #         continue
    #
    #     # EU regulation titles commonly contain "REGULATION" and are long enough
    #     # to be distinguishable from a short navigation heading.
    #     if "REGULATION" in title.upper() and len(title) > 20:
    #         return title

    # If no formal title is recognized,
    # turn the filename into readable text.
    # using the file name, because the head-line title is hard to get
    return file_title.replace("_", " ").replace("-", " ").strip()


def _build_context(state: ImportGraphState, md_content: str, file_title: str) -> LegalContext:
    """Create metadata shared by all records generated from this document."""
    # Prefer the processed Markdown path, then the original input path, and
    # finally the title so source_file is never empty.
    source_file = state.get("md_path") or state.get("local_file_path") or file_title
    return LegalContext(
        law_id=_infer_law_id(file_title, source_file),
        law_title=_infer_law_title(md_content, file_title),
        source_file=source_file,
    )


def _article_number(line: str) -> str | None:
    """Return an Article number such as ``6`` or ``6a``, otherwise ``None``."""
    match = ARTICLE_MARKER_RE.match(_strip_heading(line))
    if match:
        return match.group("number")
    return None


def _chapter_number(line: str) -> str | None:
    """Return the number/label from a Chapter, Title, or Part marker."""
    match = CHAPTER_MARKER_RE.match(_strip_heading(line))
    if match:
        return match.group("number")
    return None


def _is_chapter_marker(line: str) -> bool:
    return _chapter_number(line) is not None


def _is_annex_marker(line: str) -> bool:
    """Recognize Annex headings after removing optional Markdown hashes."""
    text = _strip_heading(line)
    return bool(re.match(r"^ANNEX\b", text, re.IGNORECASE))


def _is_whereas_marker(line: str) -> bool:
    """Recognize ``Whereas`` or ``Whereas:`` as the recital-section marker."""
    return _strip_heading(line).rstrip(":").lower() == "whereas"


def _is_structural_marker(line: str) -> bool:
    """Return whether a line begins one of the supported top-level legal units."""
    return (
        _article_number(line) is not None
        or _is_chapter_marker(line)
        or _is_annex_marker(line)
        or _is_whereas_marker(line)
    )


def _looks_like_article_title(line: str) -> bool:
    """
    Decide whether the line after ``Article N`` is an Article title.

    This is intentionally conservative. Mistaking paragraph text for a title
    would remove that text from the Article body and damage legal meaning.
    """
    text = _strip_heading(line)
    if not text:
        return False

    # A new top-level marker cannot be the title of the previous Article.
    if _is_structural_marker(line):
        return False

    # Section labels and numbered/lettered legal units belong to structure/body.
    if SECTION_MARKER_RE.match(text):
        return False
    if NUMBERED_PARAGRAPH_RE.match(text) or PAREN_NUMBER_RE.match(text) or LETTER_POINT_RE.match(text):
        return False

    # Article titles are normally short. This limit avoids treating a long
    # unnumbered opening sentence as metadata.
    if len(text) > 180:
        return False
    return True


def _base_record(context: LegalContext) -> ChunkRecord:
    """
    Create metadata shared by every parent and child record.

    Record-specific fields such as ``chunk_id``, Article number, record type,
    and text are added later by the relevant record builder.
    """
    return {
        "law_id": context.law_id,
        "law_title": context.law_title,
        "language": context.language,
        "status": context.status,
        "source_file": context.source_file,
        "key_terms": [],
    }


def _article_parent_id(law_id: str, article_number: str) -> str:
    """Build a stable parent ID such as ``gdpr:article_6``."""
    return f"{law_id}:article_{_slugify(article_number)}"


def _label_suffix(label: str) -> str:
    """
    Convert a legal label into an ID-safe suffix.

    For example, ``1(a)`` becomes ``1_point_a``. Other labels use the general
    slugifier.
    """
    match = re.match(r"^(?P<paragraph>[0-9]+)\((?P<point>[a-z])\)$", label, re.IGNORECASE)
    if match:
        return f"{match.group('paragraph')}_point_{match.group('point').lower()}"
    return _slugify(label)


def _range_suffix(prefix: str, start: str, end: str) -> str:
    """
    Build a readable child-ID suffix for one label or a legal range.

    Examples:
        paragraph + 2..3       -> ``paragraphs_2_3``
        paragraph + 1(a)..1(c) -> ``paragraph_1_points_a_c``
        definition + 1..5      -> ``definitions_1_5``
    """
    # A one-block child uses a singular prefix.
    if start == end:
        return f"{prefix}_{_label_suffix(start)}"

    # Lettered points within the same paragraph receive a specialized readable
    # range instead of a generic slug.
    start_point = re.match(r"^(?P<paragraph>[0-9]+)\((?P<point>[a-z])\)$", start, re.IGNORECASE)
    end_point = re.match(r"^(?P<paragraph>[0-9]+)\((?P<point>[a-z])\)$", end, re.IGNORECASE)
    if start_point and end_point and start_point.group("paragraph") == end_point.group("paragraph"):
        return (
            f"{prefix}_{start_point.group('paragraph')}_points_"
            f"{start_point.group('point').lower()}_{end_point.group('point').lower()}"
        )

    # Numbered paragraph and definition ranges use a plural prefix.
    return f"{prefix}s_{_label_suffix(start)}_{_label_suffix(end)}"


def _record_text(heading: str, title: str, body: str) -> str:
    """
    Assemble self-contained retrieval text.

    Repeating the Article/recital/Annex heading and title in each child means a
    retrieved child remains understandable when viewed without its parent.
    """
    parts = [heading]
    if title:
        parts.append(title)
    if body:
        parts.append(body)
    return _clean_text("\n\n".join(parts))


def _parse_numbered_blocks(lines: list[str], pattern: re.Pattern[str]) -> list[NumberedBlock]:
    """
    Parse consecutive numbered legal units while retaining continuation lines.

    ``pattern`` determines the unit type:

    - ``NUMBERED_PARAGRAPH_RE`` parses ``1. ...`` Article paragraphs;
    - ``PAREN_NUMBER_RE`` parses ``(1) ...`` definitions or recitals.

    Every line after a marker remains attached to that marker until another
    matching marker begins. This preserves multi-line sentences and subpoints.
    """
    blocks: list[NumberedBlock] = []

    # These variables hold the legal unit currently being accumulated.
    current_label = ""
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        # PDF-to-Markdown conversion sometimes promotes a legal paragraph to a
        # heading (for example, ``## 2. Paragraph 1 shall ...``). Match against
        # the legal text after removing Markdown syntax so that such paragraphs
        # are not lost or absorbed into the preceding unit.
        legal_line = _strip_heading(stripped)
        match = pattern.match(legal_line)
        if match:
            # A new marker closes the previous numbered block.
            if current_label and current_lines:
                blocks.append(NumberedBlock(current_label, _clean_text("\n".join(current_lines))))

            # Begin the new block with its marker line included in the text.
            current_label = match.group("label")
            current_lines = [legal_line]
        elif current_label:
            # Blank lines, continuation sentences, and lettered points remain
            # attached to the current numbered paragraph/definition.
            current_lines.append(line)

    # The final block has no following marker to trigger its append.
    if current_label and current_lines:
        blocks.append(NumberedBlock(current_label, _clean_text("\n".join(current_lines))))

    # Defensive filtering avoids carrying empty units into grouping logic.
    return [block for block in blocks if block.text]


def _split_paragraph_by_points(block: NumberedBlock) -> list[NumberedBlock]:
    """
    Divide one long numbered paragraph into groups of lettered points.

    Legal paragraphs often have an introductory sentence followed by points:

        1. Processing is lawful only if:
        (a) the data subject has given consent;
        (b) processing is necessary for a contract;

    The introductory sentence is called the preamble here. It is repeated in
    every point group because all of the points depend on it for their meaning.
    """
    lines = block.text.split("\n")

    # Lines before the first ``(a)`` form the paragraph's governing sentence.
    preamble_lines: list[str] = []

    # Each detected lettered point is first captured as its own temporary block.
    point_blocks: list[NumberedBlock] = []
    current_point = ""
    current_lines: list[str] = []

    for line in lines:
        legal_line = _strip_heading(line.strip())
        match = LETTER_POINT_RE.match(legal_line)
        if match:
            # Starting a new point closes the preceding point.
            if current_point and current_lines:
                point_blocks.append(NumberedBlock(f"{block.label}({current_point})", _clean_text("\n".join(current_lines))))
            current_point = match.group("label").lower()
            current_lines = [legal_line]
        elif current_point:
            # Continuation lines remain part of the active lettered point.
            current_lines.append(line)
        else:
            # No point has started yet, so this line belongs to the preamble.
            preamble_lines.append(line)

    # Flush the last point after iteration.
    if current_point and current_lines:
        point_blocks.append(NumberedBlock(f"{block.label}({current_point})", _clean_text("\n".join(current_lines))))

    # If no lettered markers were recognized, preserve the original paragraph
    # unchanged instead of applying generic text splitting.
    if not point_blocks:
        return [block]

    preamble = _clean_text("\n".join(preamble_lines))
    split_blocks: list[NumberedBlock] = []
    group: list[NumberedBlock] = []

    for point in point_blocks:
        # The governing sentence is present in every emitted group, so include
        # it in every size estimate as well.
        candidate_text = _join_block_text(group + [point], preamble)
        if group and _token_count(candidate_text) > POINT_GROUP_MAX_TOKENS:
            # Emit the existing group before the new point makes it too large.
            split_blocks.append(_point_group_to_block(block.label, group, preamble))
            group = [point]
        else:
            group.append(point)

        # Once a useful retrieval size is reached, close this semantic point
        # range rather than packing it up to the maximum and leaving a tiny tail
        # whenever the source offers more points.
        if _token_count(_join_block_text(group, preamble)) >= POINT_GROUP_TARGET_TOKENS:
            split_blocks.append(_point_group_to_block(block.label, group, preamble))
            group = []

    # Emit the final partial point group.
    if group:
        split_blocks.append(_point_group_to_block(block.label, group, preamble))

    return split_blocks


def _point_group_to_block(paragraph_label: str, points: list[NumberedBlock], preamble: str) -> NumberedBlock:
    """Convert adjacent point blocks into one labeled retrieval block."""
    # Point labels arrive as ``1(a)``. Extract only ``a`` and ``c`` to build a
    # compact range label such as ``1(a)-1(c)``.
    start_point = points[0].label.split("(", 1)[1].rstrip(")")
    end_point = points[-1].label.split("(", 1)[1].rstrip(")")
    label = f"{paragraph_label}({start_point})" if start_point == end_point else f"{paragraph_label}({start_point})-{paragraph_label}({end_point})"

    # The preamble is optional and is deliberately repeated in every group.
    text = _join_block_text(points, preamble)
    return NumberedBlock(label, text, unit_type="point_range")


def _join_block_text(blocks: list[NumberedBlock], prefix: str = "") -> str:
    """Join complete legal blocks with paragraph spacing and optional context."""
    parts = [prefix] if prefix else []
    parts.extend(block.text for block in blocks)
    return _clean_text("\n\n".join(parts))


def _normalise_group_label(label: str, *, first: bool = True) -> str:
    """Select the left or right endpoint from an existing range label."""
    if "-" not in label:
        return label
    left, right = label.split("-", 1)
    return left if first else right


def _group_blocks(
    blocks: list[NumberedBlock],
    *,
    target_tokens: int = CHILD_TARGET_TOKENS,
    hard_max_tokens: int = CHILD_HARD_MAX_TOKENS,
    max_blocks: int | None = None,
) -> list[list[NumberedBlock]]:
    """
    Group adjacent legal blocks without splitting inside them.

    A group is emitted when:

    - adding a block would exceed ``hard_max_tokens``;
    - adding a block would exceed ``max_blocks``; or
    - the current group reaches ``target_tokens``.

    The function treats an existing point-range block as indivisible. It is
    emitted alone because it was already deliberately formed by
    ``_split_paragraph_by_points``.
    """
    groups: list[list[NumberedBlock]] = []
    current: list[NumberedBlock] = []

    for block in blocks:
        # A label containing ``-`` represents an already-created legal range,
        # such as 1(a)-1(c). Keep this semantic group intact and standalone.
        if "-" in block.label:
            if current:
                groups.append(current)
                current = []
            groups.append([block])
            continue

        # Test the prospective group before committing the new block.
        candidate = current + [block]
        too_many_blocks = max_blocks is not None and len(candidate) > max_blocks

        # ``current`` ensures a single large legal block is not replaced by an
        # empty group; indivisible blocks are allowed to stand alone.
        too_many_tokens = current and _token_count(_join_block_text(candidate)) > hard_max_tokens

        if too_many_blocks or too_many_tokens:
            # Close the current group and start a new one with this complete block.
            groups.append(current)
            current = [block]
        else:
            current = candidate

        # Once the preferred size is reached, emit now rather than adding more
        # material merely because the hard maximum has not been reached.
        if _token_count(_join_block_text(current)) >= target_tokens:
            groups.append(current)
            current = []

    # Flush a final group smaller than the target.
    if current:
        groups.append(current)

    return [group for group in groups if group]


def _is_tiny_plain_paragraph(block: NumberedBlock) -> bool:
    """Return whether a paragraph is safe for conservative adjacent grouping."""
    if block.unit_type != "paragraph" or not block.label.isdigit():
        return False
    if _token_count(block.text) > TINY_PARAGRAPH_MAX_TOKENS:
        return False

    # A short governing sentence with lettered points is still a structured
    # legal unit and must not be merged with a later numbered paragraph.
    return not any(
        LETTER_POINT_RE.match(_strip_heading(line.strip()))
        for line in block.text.split("\n")
    )


def _group_article_paragraphs(
    blocks: list[NumberedBlock],
) -> list[list[NumberedBlock]]:
    """
    Keep numbered paragraphs separate except for compact runs of tiny text.

    This is deliberately stricter than ``_group_blocks``. Article paragraphs
    commonly state independent rights, duties, exceptions, and remedies, so a
    512-token target must not combine them merely because they are adjacent.
    Point ranges and normal-sized paragraphs are always standalone children.
    """
    groups: list[list[NumberedBlock]] = []
    tiny_group: list[NumberedBlock] = []

    def flush_tiny_group() -> None:
        if tiny_group:
            groups.append(list(tiny_group))
            tiny_group.clear()

    for block in blocks:
        if not _is_tiny_plain_paragraph(block):
            flush_tiny_group()
            groups.append([block])
            continue

        candidate = tiny_group + [block]
        if tiny_group and (
            len(candidate) > TINY_PARAGRAPH_GROUP_MAX_BLOCKS
            or _token_count(_join_block_text(candidate))
            > TINY_PARAGRAPH_GROUP_MAX_TOKENS
        ):
            flush_tiny_group()

        tiny_group.append(block)

    flush_tiny_group()
    return groups


def _fallback_text_blocks(text: str) -> list[NumberedBlock]:
    """
    Create synthetic blocks when no legal numbering can be recognized.

    Blank-line-separated paragraphs become blocks labeled ``1``, ``2``, etc.
    This fallback is mainly useful for unnumbered Articles and Annex content.
    """
    paragraphs = [_clean_text(item) for item in re.split(r"\n\s*\n", text) if _clean_text(item)]
    if not paragraphs:
        return []

    blocks: list[NumberedBlock] = []
    for index, paragraph in enumerate(paragraphs, start=1):
        # Synthetic labels provide deterministic child IDs and range metadata.
        blocks.append(NumberedBlock(str(index), paragraph))
    return blocks


def _make_article_records(
    context: LegalContext,
    *,
    chapter_number: str,
    chapter_title: str,
    article_number: str,
    article_title: str,
    body: str,
) -> list[ChunkRecord]:
    """
    Create one complete Article parent and its retrieval-oriented children.

    Child splitting priority:

    1. Definition Articles use one child per parenthesized definition.
    2. Normal Articles use one child per numbered paragraph by default.
    3. Some source formats use ``(1)`` for normal Article paragraphs, so that
       pattern is attempted as a compatibility fallback.
    4. Unnumbered text falls back to blank-line-separated blocks.
    5. Only oversized paragraphs are divided into lettered-point groups.
    6. Consecutive plain paragraphs may be grouped only when each is tiny.
    """
    records: list[ChunkRecord] = []

    # Every child from this Article points back to this deterministic ID.
    parent_id = _article_parent_id(context.law_id, article_number)
    heading = f"Article {article_number}"

    # Parent text always contains the complete Article, including its heading and
    # title, so it can be fetched later for controlled context expansion.
    parent_text = _record_text(heading, article_title, body)

    # Definition Articles require different child grouping and metadata.
    content_type = "definition" if "definition" in article_title.lower() else "article"

    # Start with document-level metadata, then add Article-specific fields.
    parent = _base_record(context)
    parent.update(
        {
            "chunk_id": parent_id,
            "parent_id": parent_id,
            "chapter_number": chapter_number,
            "chapter_title": chapter_title,
            "article_number": article_number,
            "article_title": article_title,
            "record_type": "parent",
            "content_type": "article",
            "text": parent_text,
        }
    )
    records.append(parent)

    body_lines = body.split("\n")

    # Parse the numbering convention most appropriate to this Article type.
    blocks = _parse_numbered_blocks(body_lines, PAREN_NUMBER_RE if content_type == "definition" else NUMBERED_PARAGRAPH_RE)
    used_fallback_blocks = False

    # Compatibility fallback for documents that format ordinary paragraphs as
    # ``(1)`` rather than ``1.``.
    if not blocks and content_type != "definition":
        blocks = _parse_numbered_blocks(body_lines, PAREN_NUMBER_RE)

    # Last-resort structural fallback for short or unnumbered Articles.
    if not blocks:
        blocks = _fallback_text_blocks(body)
        used_fallback_blocks = True

    # ``expanded_blocks`` may contain original numbered paragraphs or smaller
    # point ranges produced from unusually long/multi-point paragraphs.
    expanded_blocks: list[NumberedBlock] = []
    for block in blocks:
        has_lettered_points = any(
            LETTER_POINT_RE.match(_strip_heading(line.strip()))
            for line in block.text.split("\n")
        )

        # Definitions remain individual terms. A normal paragraph is split only
        # when it exceeds the point-group limit and offers explicit boundaries.
        if (
            content_type != "definition"
            and has_lettered_points
            and _token_count(block.text) > POINT_GROUP_MAX_TOKENS
        ):
            expanded_blocks.extend(_split_paragraph_by_points(block))
        else:
            expanded_blocks.append(block)

    if content_type == "definition":
        # A definition is normally queried by its term, so unrelated neighboring
        # definitions should not dilute its embedding.
        groups = [[block] for block in expanded_blocks]
    elif used_fallback_blocks:
        # Unnumbered prose lacks authoritative paragraph labels; retain the
        # general complete-block grouping behavior for this fallback only.
        groups = _group_blocks(expanded_blocks)
    else:
        groups = _group_article_paragraphs(expanded_blocks)

    for group in groups:
        # Range metadata identifies the first and last legal units represented by
        # this child, including endpoints from a pre-grouped point range.
        start = _normalise_group_label(group[0].label, first=True)
        end = _normalise_group_label(group[-1].label, first=False)

        # Repeat Article heading/title in every child for standalone retrieval.
        text = _record_text(heading, article_title, _join_block_text(group))
        prefix = "definition" if content_type == "definition" else "paragraph"

        # Produce stable, human-readable IDs from the legal range.
        suffix = _range_suffix(prefix, start, end)

        child = _base_record(context)
        child.update(
            {
                "chunk_id": f"{parent_id}:{suffix}",
                "parent_id": parent_id,
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
                "article_number": article_number,
                "article_title": article_title,
                "paragraph_start": start,
                "paragraph_end": end,
                "record_type": "child",
                "content_type": content_type,
                "text": text,
            }
        )
        records.append(child)

    # Parent appears first, followed by children in original legal order.
    return records


def _make_recital_records(context: LegalContext, recital_lines: list[str]) -> list[ChunkRecord]:
    """
    Create recital-group parents and individual-recital children.

    Recitals are interpretive material, not operative Articles. Keeping their
    record type separate allows retrieval to include or filter them deliberately.
    """
    # EU recitals normally use parenthesized numbering: ``(1)``, ``(2)``, etc.
    blocks = _parse_numbered_blocks(recital_lines, PAREN_NUMBER_RE)
    records: list[ChunkRecord] = []

    # Recital parents are deliberately larger than Article children because they
    # provide topic/context groups. Each individual recital remains a child.
    for group in _group_blocks(blocks, target_tokens=700, hard_max_tokens=1000, max_blocks=RECITAL_GROUP_SIZE):
        start = group[0].label
        end = group[-1].label
        parent_id = f"{context.law_id}:recitals_{start}_{end}"

        # Parent text contains every recital in this group.
        parent = _base_record(context)
        parent.update(
            {
                "chunk_id": parent_id,
                "parent_id": parent_id,
                "paragraph_start": start,
                "paragraph_end": end,
                "record_type": "parent",
                "content_type": "recital",
                "text": _record_text("Whereas", f"Recitals {start}-{end}", _join_block_text(group)),
            }
        )
        records.append(parent)

        # Each recital becomes the smallest retrieval unit and links to its
        # surrounding recital-group parent.
        for block in group:
            child = _base_record(context)
            child.update(
                {
                    "chunk_id": f"{context.law_id}:recital_{block.label}",
                    "parent_id": parent_id,
                    "paragraph_start": block.label,
                    "paragraph_end": block.label,
                    "record_type": "child",
                    "content_type": "recital",
                    "text": _record_text("Whereas", f"Recital {block.label}", block.text),
                }
            )
            records.append(child)

    return records


def _make_annex_records(context: LegalContext, annex_title: str, annex_body: str, index: int) -> list[ChunkRecord]:
    """
    Create one complete Annex parent and paragraph-group children.

    Annex layouts vary widely and may contain schedules, technical requirements,
    or tables. The current fallback treats blank-line-separated content as
    blocks, preserving the whole Annex as the authoritative parent.
    """
    # The sequential index gives stable IDs even when an Annex title contains no
    # clear Roman numeral or machine-friendly label.
    parent_id = f"{context.law_id}:annex_{index}"

    # Store the complete Annex as the parent record.
    parent = _base_record(context)
    parent.update(
        {
            "chunk_id": parent_id,
            "parent_id": parent_id,
            "record_type": "parent",
            "content_type": "annex",
            "article_title": annex_title,
            "text": _record_text(annex_title, "", annex_body),
        }
    )

    records: list[ChunkRecord] = [parent]

    # Annex content often lacks Article-style numbering, so split it first by
    # visible paragraph boundaries and then group complete paragraphs by size.
    blocks = _fallback_text_blocks(annex_body)
    for group_index, group in enumerate(_group_blocks(blocks), start=1):
        child = _base_record(context)
        child.update(
            {
                "chunk_id": f"{parent_id}:part_{group_index}",
                "parent_id": parent_id,
                "record_type": "child",
                "content_type": "annex",
                "article_title": annex_title,
                "paragraph_start": group[0].label,
                "paragraph_end": group[-1].label,
                "text": _record_text(annex_title, "", _join_block_text(group)),
            }
        )
        records.append(child)

    return records

def get_current_time():
    current_time = time.time()
    formatted_time = datetime.fromtimestamp(current_time).strftime(
        "%Y-%m-%d-%H-%M-%S"
    )

    return formatted_time
def step_2_split_law_document(md_content: str, context: LegalContext) -> list[ChunkRecord]:
    """
    Parse one Markdown law document and create structure-aware RAG records.

    This function is the main structural parsing step. It scans the Markdown
    from top to bottom and recognizes the major legal units supported by this
    module:

    - a ``Whereas:`` section, which is converted into recital records;
    - chapter/title/part headings, whose metadata is attached to later Articles;
    - Article headings, which become Article parent and child records;
    - Annex headings, which become Annex parent and child records.

    The helper functions called below perform the finer splitting. For example,
    ``_make_article_records`` creates the complete Article parent and divides its
    body into legally meaningful child chunks such as numbered paragraphs,
    groups of lettered points, or definition groups. Therefore, this function
    finds the high-level section boundaries; it does not split legal sentences.

    Args:
        md_content: Complete Markdown text of one law or regulation.
        context: Document-level metadata shared by every generated record,
            including ``law_id``, title, source file, language, and status.

    Returns:
        A flat list containing both parent and child records. Parent-child
        relationships are represented by the records' ``parent_id`` values.
        The original legal order is preserved in the list.

    If no recognizable legal structure is found, the whole document is kept as
    one parent plus one child. This fallback prevents an unusual document from
    being silently lost during ingestion.
    """

    # Work with lines because legal headings and numbered units normally begin
    # on their own lines. Keeping the original order also makes boundary
    # detection deterministic.
    lines = md_content.split("\n")

    # All generated parent and child records are appended here in source order.
    records: list[ChunkRecord] = []

    # Chapter metadata is parser state: after a chapter heading is found, these
    # values are copied to every following Article until another chapter begins.
    current_chapter_number = ""
    current_chapter_title = ""

    # Annexes do not always have a usable numeric label, so a sequential index
    # guarantees stable IDs such as ``law_id:annex_1``.
    annex_index = 0

    # ``index`` is advanced manually because each recognized section consumes a
    # variable number of lines. After processing a section, it jumps directly to
    # the first line that belongs to the next section.
    index = 0

    while index < len(lines):
        line = lines[index]

        # A "Whereas:" marker introduces the regulation's recitals. Recitals are
        # explanatory material and must remain separate from operative Articles.
        if _is_whereas_marker(line):
            start = index + 1
            end = start

            # Collect recital lines until the next top-level legal structure.
            # ``end`` itself is not included, so that marker is processed again
            # by the appropriate branch on the next loop iteration.
            while end < len(lines) and not (_is_chapter_marker(lines[end])
                                            or _article_number(lines[end])
                                            or _is_annex_marker(lines[end])):
                end += 1

            # The helper recognizes individual numbered recitals and creates
            # recital parents/children without mixing them with Article text.
            records.extend(_make_recital_records(context, lines[start:end]))
            index = end
            continue

        # Chapter, Title, and Part headings provide context for Articles but do
        # not create standalone records in this implementation.
        chapter_number = _chapter_number(line)
        if chapter_number:
            current_chapter_number = chapter_number

            # Reset the old title immediately: a new chapter may have no title,
            # and it must not accidentally inherit the previous chapter's title.
            current_chapter_title = ""
            title_index = index + 1

            # Legal Markdown often has blank lines between a chapter number and
            # its title, so move to the next non-empty line.
            while title_index < len(lines) and not lines[title_index].strip():
                title_index += 1

            # Accept the next line as the chapter title only when it is ordinary
            # text, not another legal marker such as an Article or section label.
            if title_index < len(lines) and not _is_structural_marker(lines[title_index]):
                title_candidate = _strip_heading(lines[title_index])
                if title_candidate and not SECTION_MARKER_RE.match(title_candidate):
                    current_chapter_title = title_candidate

                    # The title line has been consumed together with the chapter
                    # marker, so continue scanning after it.
                    index = title_index + 1
                    continue

            # No separate title was found; consume only the chapter marker.
            index += 1
            continue

        # An Article is the normal parent unit for legal retrieval.
        article_number = _article_number(line)
        if article_number:
            title = ""
            title_index = index + 1

            # Skip spacing between ``Article N`` and a possible Article title.
            while title_index < len(lines) and not lines[title_index].strip():
                title_index += 1

            # By default, the first non-empty line belongs to the Article body.
            body_start = title_index

            # Some source files put a short title after the Article heading.
            # When the heuristic recognizes one, exclude it from the body and
            # store it separately as Article metadata.
            if title_index < len(lines) and _looks_like_article_title(lines[title_index]):
                title = _strip_heading(lines[title_index])
                body_start = title_index + 1

            # The Article body extends to, but does not include, the next Article,
            # chapter/title/part marker, or Annex. Internal numbered paragraphs
            # are intentionally not treated as top-level boundaries here.
            end = body_start
            while end < len(lines) and not (_article_number(lines[end]) or _is_chapter_marker(lines[end]) or _is_annex_marker(lines[end])):
                end += 1

            # Normalize whitespace before handing the complete Article to the
            # detailed Article splitter.
            body = _clean_text("\n".join(lines[body_start:end]))
            records.extend(
                _make_article_records(
                    context,
                    chapter_number=current_chapter_number,
                    chapter_title=current_chapter_title,
                    article_number=article_number,
                    article_title=title,
                    body=body,
                )
            )

            # Resume at the top-level marker that ended this Article.
            index = end
            continue

        # Annexes are stored separately because they may contain tables,
        # schedules, forms, or technical requirements rather than Article text.
        if _is_annex_marker(line):
            annex_index += 1
            annex_title = _strip_heading(line)
            start = index + 1
            end = start

            # An Annex continues until the next Annex marker or end of document.
            # Article-like text inside an Annex therefore remains inside it.
            while end < len(lines) and not _is_annex_marker(lines[end]):
                end += 1

            # The helper creates one complete Annex parent and smaller child
            # blocks while retaining the same parent ID.
            records.extend(_make_annex_records(context, annex_title, _clean_text("\n".join(lines[start:end])), annex_index))
            index = end
            continue

        # Ignore text outside recognized structures, such as a cover page,
        # publication metadata, or table of contents. Move forward one line.
        index += 1

    # Safety fallback for Markdown that uses an unsupported layout or contains
    # no recognizable recital, Article, or Annex headings.
    if not records:
        logger.warning("No legal structure found; falling back to one parent and one child record.")
        parent_id = f"{context.law_id}:document"
        text = _clean_text(md_content)

        # Keep the full source as a retrievable parent record.
        parent = _base_record(context)
        parent.update(
            {
                "chunk_id": parent_id,
                "parent_id": parent_id,
                "record_type": "parent",
                "content_type": "article",
                "text": text,
            }
        )

        # Also create a child because downstream embedding/vector-storage stages
        # operate on child records. ``dict`` makes a separate shallow copy before
        # changing its ID and record type.
        child = dict(parent)
        child.update(
            {
                "chunk_id": f"{parent_id}:part_1",
                "record_type": "child",
            }
        )
        records.extend([parent, child])

    # Records remain in document order and are validated by
    # ``step_3_refine_chunks`` before they are passed to storage nodes.
    return records


def step_3_refine_chunks(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    """
    Final safety pass after legal-structure splitting.

    The real legal splitting already happens in step_2_split_law_document. This
    step deliberately does not re-split by fixed size because that could break a
    legal sentence or point. It only removes empty records and validates stable
    parent-child metadata before downstream Elasticsearch/Milvus nodes run.
    """
    # Build a new ordered list containing only validated, non-empty records.
    refined_chunks: list[ChunkRecord] = []

    # Track IDs during the same pass so duplicate Elasticsearch/Milvus keys fail
    # before data reaches either storage backend.
    seen_chunk_ids: set[str] = set()

    for chunk in chunks:
        # Normalize identifiers and text before validating them. This avoids IDs
        # that differ only by accidental surrounding whitespace.
        chunk_id = chunk.get("chunk_id", "").strip()
        parent_id = chunk.get("parent_id", "").strip()
        text = _clean_text(chunk.get("text", ""))
        record_type = chunk.get("record_type", "")

        # IDs are mandatory because they connect parent retrieval, child vectors,
        # Elasticsearch records, and Milvus records.
        if not chunk_id:
            raise ValueError("Chunk record is missing chunk_id.")
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate chunk_id generated: {chunk_id}")
        if not parent_id:
            raise ValueError(f"Chunk {chunk_id} is missing parent_id.")

        # Empty records provide no retrieval value. Skip them while preserving
        # all valid records and log the exact ID for diagnosis.
        if not text:
            logger.warning(f"Skipping empty legal chunk: {chunk_id}")
            continue

        # Restrict values to the record types understood by downstream schemas.
        if record_type not in {"parent", "child", "parent_child"}:
            raise ValueError(
                f"Chunk {chunk_id} has invalid record_type: {record_type!r}"
            )

        # Store normalized values back in the record.
        chunk["chunk_id"] = chunk_id
        chunk["parent_id"] = parent_id
        chunk["text"] = text

        # Supply schema-compatible defaults for records created by fallback paths
        # or loaded from older intermediate files.
        chunk.setdefault("key_terms", [])
        chunk.setdefault("language", "en")
        chunk.setdefault("status", "in_force")

        refined_chunks.append(chunk)
        seen_chunk_ids.add(chunk_id)

    # Collect IDs that are valid parent targets. ``parent_child`` is supported
    # for short units that can serve both retrieval roles.
    parent_ids = {
        chunk["chunk_id"]
        for chunk in refined_chunks
        if chunk.get("record_type") in {"parent", "parent_child"}
    }

    # Verify referential integrity after all empty records have been removed.
    # A child must never reference a parent absent from the final output list.
    missing_parent_ids = sorted({
        chunk["parent_id"]
        for chunk in refined_chunks
        if (
            chunk.get("record_type") == "child"
            and chunk.get("parent_id") not in parent_ids
        )
    })

    if missing_parent_ids:
        # Limit the error preview to avoid an enormous exception for a badly
        # malformed source while still exposing useful parent IDs.
        raise ValueError(
            "Child chunks reference missing parent records: "
            + ", ".join(missing_parent_ids[:10])
        )

    return refined_chunks


def step_4_add_content_hashes(
    chunks: list[ChunkRecord],
    md_content: str,
) -> str:
    """
    Add deterministic document and chunk fingerprints to every legal record.

    The same document hash is copied to all parents and children. Each chunk
    receives its own hash based on stable identity metadata and authoritative
    text. Hashes are metadata; stable legal ``chunk_id`` values remain the
    Elasticsearch document IDs and Milvus primary keys.
    """
    document_hash = calculate_document_hash(md_content)
    for chunk in chunks:
        chunk["document_hash"] = document_hash
        chunk["chunk_hash"] = calculate_chunk_hash(chunk)
    return document_hash


def step_5_backup_chunks(state: ImportGraphState, sections: list[dict[str, Any]]) -> None:
    """
    Persist validated chunks as a timestamped ``chunks_*.json`` backup.

    This is an intermediate artifact. Downstream nodes continue using the
    in-memory ``state["chunks"]`` list.
    """
    # Prefer the graph's output directory; otherwise save beside the Markdown.
    local_dir = state.get("local_dir") or str(Path(state.get("md_path", ".")).parent)

    # Repeated imports and pre-existing output directories are both supported.
    os.makedirs(local_dir, exist_ok=True)

    backup_file_path = os.path.join(local_dir, "chunks_" + str(get_current_time())+ ".json")

    # Preserve non-ASCII legal names and use indentation for manual review.
    with open(backup_file_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)
    logger.info(f"Backed up law chunks to: {backup_file_path}")


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    Orchestrate all law-splitting steps for one LangGraph state.

    The node mutates and returns the same state object. Its main output is
    ``state["chunks"]``, an ordered list of validated parents and children.
    """
    # 1. Discover the runtime function name, log entry, and expose running status
    # to the task-progress utilities.
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] Started. Current state: {state}")
    add_running_task(state["task_id"], function_name)

    # 2. Obtain normalized Markdown and title metadata.
    md_content, file_title = step_1_get_content(state)

    # Log only a preview rather than the complete potentially large document.
    logger.info(f"md_content: {md_content[:1000]} and file_title: {file_title}")

    # 3. Infer metadata copied into every parent and child record.
    context = _build_context(state, md_content, file_title)

    # 4. Perform the legal-structure-aware parent-child split.
    chunks = step_2_split_law_document(md_content, context)

    # 5. Normalize and validate IDs, text, record types, and parent links.
    chunks = step_3_refine_chunks(chunks)

    # 6. Add deterministic fingerprints for database idempotency and auditing.
    document_hash = step_4_add_content_hashes(chunks, md_content)

    # 7. Publish chunks to graph state and save the reviewable intermediate file.
    state["document_hash"] = document_hash
    state["chunks"] = chunks
    step_5_backup_chunks(state, chunks)

    # 8. Report output composition for ingestion monitoring.
    parent_count = sum(1 for chunk in chunks if chunk.get("record_type") == "parent")
    child_count = sum(1 for chunk in chunks if chunk.get("record_type") == "child")
    logger.info(
        f"Law document split completed. law_id={context.law_id}, "
        f"document_hash={document_hash}, parents={parent_count}, "
        f"children={child_count}, total={len(chunks)}"
    )

    # 9. Log successful completion and move this task from running to done.
    logger.info(f">>> [{function_name}] Finished. Current state: {state}")
    add_done_task(state["task_id"], function_name)
    return state


def test_node_document_split() -> None:
    """Small local smoke test for legal parent-child splitting."""
    sample_md = """
# REGULATION TEST

## Whereas:

(1) First recital text.

(2) Second recital text.

## CHAPTER I

## GENERAL PROVISIONS

## Article 1

## Subject matter

1. This Regulation lays down rules.

2. It applies to data.

## Article 2

## Definitions

(1) ‘data’ means information;

(2) ‘metadata’ means descriptive information;
"""
    state: ImportGraphState = {
        "task_id": "test_law_split",
        "md_content": sample_md,
        "file_title": "Test_Regulation",
        "local_dir": "/tmp/law_split_test",
        "md_path": "",
    }
    result = node_document_split(state)
    chunks = result["chunks"]
    assert any(chunk["record_type"] == "parent" for chunk in chunks)
    assert any(chunk["record_type"] == "child" for chunk in chunks)
    assert any(chunk["content_type"] == "recital" for chunk in chunks)
    assert any(chunk["content_type"] == "definition" for chunk in chunks)
    logger.info(f"Smoke test generated {len(chunks)} chunks")


def test_node_document_split_with_md_img_example() -> None:
    """
    Local integration example retained from the previous splitter style.

    This mirrors the old manual test flow:
    node_md_img -> node_document_split -> inspect final chunks.
    It is intentionally not used by automated unittests because node_md_img can
    call external services depending on local configuration.
    """
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img_process import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    test_md_name = os.path.join(
        "output",
        "General_Data_Protection_Regulation_GDPR",
        "General_Data_Protection_Regulation_GDPR.md",
    )
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
        return

    test_state = {
        "md_path": test_md_path,
        "task_id": "test_task_123456",
        "md_content": "",
        "file_title": "General_Data_Protection_Regulation_GDPR",
        "local_dir": os.path.join(
            PROJECT_ROOT,
            "output",
            "General_Data_Protection_Regulation_GDPR",
        ),
    }

    logger.info("开始本地测试 - MD图片处理全流程")
    result_state = node_md_img(test_state)
    logger.info("本地MD图片处理全流程完成")
    logger.info("\n=== 开始执行文档切分节点集成测试 ===")

    logger.info(">> 开始运行当前节点：node_document_split（法律文档切分）")
    final_state = node_document_split(result_state)
    final_chunks = final_state.get("chunks", [])
    logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")


if __name__ == "__main__":
    # test_node_document_split()
    test_node_document_split_with_md_img_example()
