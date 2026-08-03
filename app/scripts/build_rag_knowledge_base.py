"""
Build the legal RAG knowledge base from a source document or directory.

This developer CLI accepts one Markdown/PDF file or discovers all supported
files in a directory, then sends each document through the same LangGraph
workflow used by the upload API:

    source file
        -> PDF-to-Markdown / Markdown image processing
        -> structure-aware legal splitting
        -> key-term extraction
        -> Elasticsearch parent/child records
        -> child embeddings
        -> Milvus vectors

Documents are processed sequentially by default. This is intentional: the
configured TEI embedding service has limited concurrency, and parallel imports
can easily produce HTTP 429 "Model is overloaded" responses.

Example:

    uv run python -m app.scripts.build_rag_knowledge_base ./laws

    uv run python -m app.scripts.build_rag_knowledge_base \
        ./laws/EU_AI_Act.md

Use ``--dry-run`` first to inspect discovery and derived law IDs without
loading the graph or contacting Elasticsearch, the LLM, TEI, or Milvus.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, Sequence

from app.core.logger import logger
from app.import_process.agent.state import get_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    set_task_result,
    update_task_status,
)


SUPPORTED_EXTENSIONS = {".md", ".pdf"}
DEFAULT_OUTPUT_PARENT = PROJECT_ROOT / "output" / "knowledge_base_builds"
REPORT_FILENAME = "batch_import_report.json"


class ImportGraph(Protocol):
    """Small protocol needed by the batch runner and its isolated tests."""

    def astream(
        self,
        state: dict[str, Any],
        *,
        stream_mode: str,
    ) -> Any:
        """Yield LangGraph node updates asynchronously."""


@dataclass(frozen=True)
class DiscoveredDocument:
    """One source document and the stable law ID its chunks will receive."""

    source_path: str
    law_id: str


@dataclass(frozen=True)
class FileImportResult:
    """Outcome of processing one source document."""

    source_path: str
    law_id: str
    task_id: str
    status: str
    work_dir: str
    duration_seconds: float
    error: str = ""


@dataclass(frozen=True)
class BatchImportSummary:
    """Serializable summary written after every batch run."""

    source_dir: str
    output_dir: str
    started_at: str
    finished_at: str
    discovered_count: int
    processed_count: int
    succeeded_count: int
    failed_count: int
    stopped_early: bool
    results: list[FileImportResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert nested dataclasses into JSON-compatible dictionaries."""
        return asdict(self)


def discover_documents(
    source_dir: Path,
    *,
    recursive: bool = True,
) -> list[Path]:
    """
    Return one source file or supported directory files in deterministic order.

    Only lowercase ``.md`` and ``.pdf`` extensions are accepted because the
    existing graph entry node routes using those exact suffixes. A directly
    supplied unsupported file is rejected with a clear error. Unsupported
    files inside a directory are ignored, and an empty supported-file set is
    treated as an input error rather than a successful no-op.

    The parameter keeps its historical ``source_dir`` name for compatibility,
    although it may now point to either a directory or one source file.
    """
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source path does not exist: {source_dir}"
        )

    if source_dir.is_file():
        if source_dir.suffix not in SUPPORTED_EXTENSIONS:
            extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported source file: {source_dir}. "
                f"Expected: {extensions}."
            )
        return [source_dir]

    if not source_dir.is_dir():
        raise ValueError(
            f"Source path is neither a regular file nor a directory: "
            f"{source_dir}"
        )

    candidates = source_dir.rglob("*") if recursive else source_dir.glob("*")
    documents = sorted(
        (
            path.resolve()
            for path in candidates
            if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )
    if not documents:
        extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"No supported documents found in {source_dir}. "
            f"Expected: {extensions}."
        )
    return documents


def infer_law_id(source_path: Path) -> str:
    """
    Derive the exact law ID used by the legal document splitter.

    Calling the splitter's helper prevents the batch validator from drifting
    away from chunk ID generation when known-law mappings are extended later.
    This import is safe during ``--dry-run`` and does not initialize database
    clients or load the full LangGraph.
    """
    from app.import_process.agent.nodes.node_document_split import (
        _infer_law_id,
    )

    return _infer_law_id(source_path.stem, str(source_path))


def describe_documents(paths: Sequence[Path]) -> list[DiscoveredDocument]:
    """Attach the splitter-compatible law ID to every discovered path."""
    return [
        DiscoveredDocument(
            source_path=str(path),
            law_id=infer_law_id(path),
        )
        for path in paths
    ]


def validate_unique_law_ids(
    documents: Sequence[DiscoveredDocument],
) -> None:
    """
    Reject documents that would write records under the same ``law_id``.

    Chunk primary keys begin with ``law_id``. Importing two source documents
    for the same law in one batch could therefore overwrite earlier chunks
    while leaving stale records behind. Failing before graph execution makes
    this destructive ambiguity visible to the developer.
    """
    paths_by_law_id: dict[str, list[str]] = {}
    for document in documents:
        paths_by_law_id.setdefault(document.law_id, []).append(
            document.source_path
        )

    duplicates = {
        law_id: paths
        for law_id, paths in paths_by_law_id.items()
        if len(paths) > 1
    }
    if not duplicates:
        return

    details = "; ".join(
        f"{law_id}: {', '.join(paths)}"
        for law_id, paths in sorted(duplicates.items())
    )
    raise ValueError(
        "Multiple source documents resolve to the same law_id. "
        "Import one authoritative document per law to avoid overwriting "
        f"Elasticsearch and Milvus records. Conflicts: {details}"
    )


def _safe_directory_stem(value: str) -> str:
    """Create a readable filesystem component from a source filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "document"


@lru_cache(maxsize=1)
def _get_import_graph() -> ImportGraph:
    """
    Load the compiled ingestion graph only when real processing begins.

    The graph imports external-service integrations, so keeping this import
    lazy allows discovery, ``--dry-run``, and unit tests to run offline.
    """
    from app.import_process.agent.import_main_graph import kb_import_app

    return kb_import_app


def _completed_persistence(final_updates: dict[str, Any]) -> None:
    """Verify that both authoritative text and vectors reached their stores."""
    if final_updates.get("es_import_success") is not True:
        raise RuntimeError(
            "The graph finished without confirming Elasticsearch import."
        )
    if final_updates.get("milvus_import_success") is not True:
        raise RuntimeError(
            "The graph finished without confirming Milvus import."
        )


async def import_document(
    document: DiscoveredDocument,
    output_dir: Path,
    graph: ImportGraph,
) -> FileImportResult:
    """
    Process one document and convert all exceptions into a report result.

    Each document receives a unique task directory for generated Markdown,
    extracted images, ``chunks.json``, and ``chunks_with_keywords.json``.
    The original source file remains untouched.
    """
    task_id = str(uuid.uuid4())
    source_path = Path(document.source_path)
    directory_name = (
        f"{_safe_directory_stem(source_path.stem)}-{task_id[:8]}"
    )
    work_dir = output_dir / directory_name
    work_dir.mkdir(parents=True, exist_ok=False)

    state = get_default_state()
    state.update(
        {
            "task_id": task_id,
            "local_file_path": str(source_path),
            "local_dir": str(work_dir),
        }
    )

    started = time.monotonic()
    update_task_status(task_id, TASK_STATUS_PROCESSING)
    set_task_result(task_id, "error", "")
    logger.info(
        f"Starting legal document import: task_id={task_id}, "
        f"law_id={document.law_id}, source={source_path}"
    )

    try:
        final_updates: dict[str, Any] = {}
        async for event in graph.astream(state, stream_mode="updates"):
            for node_name, update in event.items():
                update_keys = (
                    sorted(update)
                    if isinstance(update, dict)
                    else []
                )
                if isinstance(update, dict):
                    final_updates.update(update)
                logger.info(
                    f"Import node completed: task_id={task_id}, "
                    f"node={node_name}, updated_fields={update_keys}"
                )

        _completed_persistence(final_updates)
        update_task_status(task_id, TASK_STATUS_COMPLETED)
        duration = time.monotonic() - started
        logger.info(
            f"Legal document import completed: task_id={task_id}, "
            f"law_id={document.law_id}, seconds={duration:.2f}"
        )
        return FileImportResult(
            source_path=str(source_path),
            law_id=document.law_id,
            task_id=task_id,
            status=TASK_STATUS_COMPLETED,
            work_dir=str(work_dir),
            duration_seconds=round(duration, 3),
        )
    except Exception as exc:
        duration = time.monotonic() - started
        error = f"{type(exc).__name__}: {exc}"
        set_task_result(task_id, "error", error)
        update_task_status(task_id, TASK_STATUS_FAILED)
        logger.exception(
            f"Legal document import failed: task_id={task_id}, "
            f"law_id={document.law_id}, source={source_path}"
        )
        return FileImportResult(
            source_path=str(source_path),
            law_id=document.law_id,
            task_id=task_id,
            status=TASK_STATUS_FAILED,
            work_dir=str(work_dir),
            duration_seconds=round(duration, 3),
            error=error,
        )


def write_report(summary: BatchImportSummary, report_path: Path) -> None:
    """Atomically write the batch report so partial JSON is never retained."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_suffix(f"{report_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(report_path)


def get_report_path(summary: BatchImportSummary) -> Path:
    """
    Return a per-document report path or the shared batch report path.

    A single-document run already creates a unique work-directory name such as
    ``Open_Data_Directive-5fa9ead6``. Reusing that name as the report prefix
    keeps reports from separate single-file runs from overwriting one another.
    True multi-document runs retain the concise ``batch_import_report.json``
    name because the report describes several work directories.
    """
    output_dir = Path(summary.output_dir)
    if summary.discovered_count == 1 and len(summary.results) == 1:
        work_directory_name = Path(summary.results[0].work_dir).name
        return output_dir / f"{work_directory_name}_{REPORT_FILENAME}"
    return output_dir / REPORT_FILENAME


async def build_knowledge_base(
    documents: Sequence[DiscoveredDocument],
    output_dir: Path,
    graph: ImportGraph,
    *,
    source_dir: Path,
    fail_fast: bool = False,
) -> BatchImportSummary:
    """
    Import documents sequentially and persist a machine-readable batch report.

    A failed document does not prevent later documents from being attempted
    unless ``fail_fast`` is enabled. The function itself does not close shared
    service clients because callers may inject a fake graph or reuse clients;
    the CLI entry point owns that application-level lifecycle.
    """
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    results: list[FileImportResult] = []

    for document in documents:
        result = await import_document(document, output_dir, graph)
        results.append(result)
        if fail_fast and result.status == TASK_STATUS_FAILED:
            break

    finished_at = datetime.now(timezone.utc)
    succeeded_count = sum(
        result.status == TASK_STATUS_COMPLETED for result in results
    )
    failed_count = sum(
        result.status == TASK_STATUS_FAILED for result in results
    )
    summary = BatchImportSummary(
        source_dir=str(source_dir.expanduser().resolve()),
        output_dir=str(output_dir),
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        discovered_count=len(documents),
        processed_count=len(results),
        succeeded_count=succeeded_count,
        failed_count=failed_count,
        stopped_early=len(results) < len(documents),
        results=results,
    )
    write_report(summary, get_report_path(summary))
    return summary


def _default_output_dir() -> Path:
    """Create a timestamped default path without touching the filesystem."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_PARENT / timestamp


def build_argument_parser() -> argparse.ArgumentParser:
    """Define the command-line interface in one testable function."""
    parser = argparse.ArgumentParser(
        description=(
            "Import one legal Markdown/PDF document or all supported "
            "documents in a folder into Elasticsearch and Milvus."
        )
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help=(
            "Authoritative .md/.pdf law document, or a directory containing "
            "law documents."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-file artifacts and the JSON report. "
            "Defaults to output/knowledge_base_builds/<timestamp>."
        ),
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help=(
            "For directory input, inspect only its direct children. "
            "This option has no effect on a single file."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered documents and law IDs without importing them.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed document instead of continuing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N documents after deterministic sorting.",
    )
    return parser


def _print_discovery(documents: Sequence[DiscoveredDocument]) -> None:
    """Print a compact plan suitable for shell use and dry-run review."""
    print(f"Discovered {len(documents)} legal document(s):")
    for index, document in enumerate(documents, start=1):
        print(
            f"  {index}. law_id={document.law_id} "
            f"path={document.source_path}"
        )


async def _close_shared_clients() -> None:
    """Close clients initialized by any graph node during this process."""
    from app.clients.es_utils import es_client_manager
    from app.clients.milvus_utils import milvus_client_manager

    await es_client_manager.close()
    await milvus_client_manager.close()


async def async_main(arguments: argparse.Namespace) -> int:
    """Run validated CLI arguments and return a conventional process code."""
    documents = describe_documents(
        discover_documents(
            arguments.source_dir,
            recursive=not arguments.non_recursive,
        )
    )

    if arguments.limit is not None:
        if arguments.limit < 1:
            raise ValueError("--limit must be greater than zero.")
        documents = documents[: arguments.limit]

    validate_unique_law_ids(documents)
    _print_discovery(documents)
    if arguments.dry_run:
        print("Dry run complete; no external services were contacted.")
        return 0

    output_dir = arguments.output_dir or _default_output_dir()
    try:
        summary = await build_knowledge_base(
            documents,
            output_dir,
            _get_import_graph(),
            source_dir=arguments.source_dir,
            fail_fast=arguments.fail_fast,
        )
    finally:
        await _close_shared_clients()

    report_path = get_report_path(summary)
    print(
        f"Import finished: succeeded={summary.succeeded_count}, "
        f"failed={summary.failed_count}, report={report_path}"
    )
    return 1 if summary.failed_count else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the asynchronous importer, and handle input errors."""
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        return asyncio.run(async_main(arguments))
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())


"""
  # Check files and derived law IDs first
  uv run python -m app.scripts.build_rag_knowledge_base ./laws --dry-run

  # Build the knowledge base
  uv run python -m app.scripts.build_rag_knowledge_base ./laws

  # Build the knowledge base from one law document
  uv run python -m app.scripts.build_rag_knowledge_base \
    ./laws/EU_AI_Act.md

  # Custom artifact directory
  uv run python -m app.scripts.build_rag_knowledge_base ./laws \
    --output-dir ./output/my_build

  Sequential processing intentionally helps avoid TEI HTTP 429 overload errors.

  Validation

  - All 38 relevant tests pass.
  - CLI help and Python compilation pass.
  - No whitespace errors in the new files.
  
  
  python -m app.scripts.build_rag_knowledge_base ./doc/laws  --output-dir ./output/laws_build
  
"""
