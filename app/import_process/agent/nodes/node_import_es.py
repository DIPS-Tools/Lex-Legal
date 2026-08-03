from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from app.clients.es_utils import es_client_manager
from app.config.es_config import es_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.repositories.es.es_repository import ESRepository
from app.utils.task_utils import add_done_task, add_running_task


async def node_import_es(
    state: ImportGraphState,
) -> dict[str, Any]:
    """
    Store all validated parent and child chunk records in Elasticsearch.

    Elasticsearch is the authoritative text store and provides BM25 retrieval.
    The repository owns index creation, YAML-based validation, bulk requests,
    and partial-failure detection; this node only coordinates graph state.
    """
    function_name = sys._getframe().f_code.co_name
    task_id = state.get("task_id", "unknown")
    chunks = state.get("chunks", [])

    logger.info(
        f">>> [{function_name}] Started. "
        f"task_id={task_id}, chunks={len(chunks)}"
    )
    add_running_task(task_id, function_name)

    if not chunks:
        raise ValueError(
            f"No chunks found for Elasticsearch import. "
            f"task_id={task_id}"
        )

    es_client_manager.init()
    repository = ESRepository(
        client=es_client_manager.get_client(),
        index_name=es_config.index_name,
    )

    try:
        await repository.ensure_index()
        success_count = await repository.bulk_upsert(chunks)

        chunk_ids_by_law: dict[str, list[str]] = {}
        for chunk in chunks:
            chunk_ids_by_law.setdefault(
                chunk["law_id"],
                [],
            ).append(chunk["chunk_id"])

        deleted_stale_count = 0
        for law_id, chunk_ids in chunk_ids_by_law.items():
            deleted_stale_count += await repository.delete_stale_records(
                law_id,
                chunk_ids,
            )
    except Exception:
        logger.exception(
            f"Elasticsearch import failed. task_id={task_id}"
        )
        raise

    logger.info(
        f"Elasticsearch import completed. "
        f"task_id={task_id}, index={repository.index_name}, "
        f"count={success_count}, stale_deleted={deleted_stale_count}"
    )
    add_done_task(task_id, function_name)

    return {
        "es_import_success": True,
        "es_imported_count": success_count,
        "es_deleted_stale_count": deleted_stale_count,
        "es_index_name": repository.index_name,
        "es_import_error": "",
    }



async def test_import_es_example() -> None:
    """
    Manual integration test using the GDPR example from node_document_split.

    This test calls the configured external LLM and may incur API usage. Run it
    directly only when the API configuration is available.
    """
    from app.import_process.agent.nodes.node_document_split import (
        node_document_split,
    )
    from app.import_process.agent.nodes.node_key_terms_extraction import (
        node_key_terms_extraction
    )
    from app.utils.path_util import PROJECT_ROOT

    # Reuse the same real GDPR Markdown example used by the document-splitter
    # manual integration test.
    test_md_path = os.path.join(
        PROJECT_ROOT,
        "output",
        "General_Data_Protection_Regulation_GDPR",
        "General_Data_Protection_Regulation_TEST.md",
    )
    if not os.path.exists(test_md_path):
        raise FileNotFoundError(f"GDPR test Markdown does not exist: {test_md_path}")

    # The splitter reads ``md_path`` because ``md_content`` starts empty, writes
    # ``chunks.json``, and places generated records in state["chunks"].
    test_state: ImportGraphState = {
        "task_id": "test_key_terms_extraction",
        "md_path": test_md_path,
        "md_content": "",
        "file_title": "General_Data_Protection_Regulation_TEST",
        "local_dir": os.path.dirname(test_md_path),
    }

    # Run the same two-node order used by the real LangGraph workflow.
    try:
        split_state = node_document_split(test_state)
        final_state = node_key_terms_extraction(split_state)

        # node_import_es is asynchronous because it uses AsyncElasticsearch.
        # Awaiting it executes the coroutine and prevents the
        # "coroutine was never awaited" RuntimeWarning.
        es_result = await node_import_es(final_state)
        final_state.update(es_result)

        # Count only child chunks because parent chunks intentionally keep empty
        # key_terms in this design.
        children = [
            chunk
            for chunk in final_state["chunks"]
            if chunk.get("record_type") == "child"
        ]
        logger.info(
            f"Manual Elasticsearch import test completed: "
            f"children={len(children)}, "
            f"children_with_terms="
            f"{sum(bool(chunk.get('key_terms')) for chunk in children)}, "
            f"es_imported_count={final_state['es_imported_count']}"
        )
    finally:
        # This standalone test owns the client lifecycle. In the real
        # application, close it once during application shutdown instead.
        await es_client_manager.close()


if __name__ == "__main__":
    # asyncio.run creates an event loop and executes the asynchronous test.
    asyncio.run(test_import_es_example())
