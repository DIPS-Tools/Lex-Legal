from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from app.clients.milvus_utils import milvus_client_manager
from app.config.milvus_config import milvus_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.repositories.milvus.milvus_repository import (
    MilvusRepository,
)
from app.utils.task_utils import add_done_task, add_running_task


async def node_import_milvus(
    state: ImportGraphState,
) -> dict[str, Any]:

    """
    Store child embedding records in the configured Milvus collection.

    ``node_bge_embedding`` already converts child chunks into the strict
    YAML-derived Milvus shape and places them in ``embeddings_content``.
    embeddings_content: list[MilvusRecord], where  MilvusRecord is:

        class MilvusRecord(TypedDict, total=False):
            chunk_id: Required[str]
            parent_id: Required[str]
            law_id: Required[str]
            chapter_number: str
            article_number: str
            record_type: str
            content_type: str
            key_terms: list[str]
            language: str
            status: str
            dense_vector: Required[list[float]]

    This node creates the collection when needed and idempotently upserts every
    record using its stable string ``chunk_id`` primary key.
    """
    function_name = sys._getframe().f_code.co_name
    task_id = state.get("task_id", "unknown")

    # get embedding
    embedding_records = state.get("embeddings_content", [])

    logger.info(
        f">>> [{function_name}] Started. "
        f"task_id={task_id}, embeddings={len(embedding_records)}"
    )
    add_running_task(task_id, function_name)

    if not isinstance(embedding_records, list) or not embedding_records:
        raise ValueError(
            f"No embedding records found for Milvus import. "
            f"task_id={task_id}"
        )
    if state.get("embedding_success") is False:
        raise ValueError(
            f"Embedding generation did not complete successfully. "
            f"task_id={task_id}"
        )

    # get milvus client
    milvus_client_manager.init()
    #
    repository = MilvusRepository(
        client=milvus_client_manager.get_client(),
        collection_name=milvus_config.chunks_collection,
    )

    try:
        await repository.ensure_collection()
        success_count = await repository.bulk_upsert(embedding_records)

        chunk_ids_by_law: dict[str, list[str]] = {}
        for record in embedding_records:
            chunk_ids_by_law.setdefault(
                record["law_id"],
                [],
            ).append(record["chunk_id"])

        deleted_stale_count = 0
        for law_id, chunk_ids in chunk_ids_by_law.items():
            deleted_stale_count += await repository.delete_stale_records(
                law_id,
                chunk_ids,
            )
    except Exception as exc:
        state["milvus_import_success"] = False
        state["milvus_imported_count"] = 0
        state["milvus_collection_name"] = repository.collection_name
        state["milvus_import_error"] = str(exc)
        logger.exception(
            f"Milvus import failed. task_id={task_id}"
        )
        raise

    logger.info(
        f"Milvus import completed. "
        f"task_id={task_id}, collection={repository.collection_name}, "
        f"count={success_count}, stale_deleted={deleted_stale_count}"
    )
    add_done_task(task_id, function_name)

    return {
        "milvus_import_success": True,
        "milvus_imported_count": success_count,
        "milvus_deleted_stale_count": deleted_stale_count,
        "milvus_collection_name": repository.collection_name,
        "milvus_import_error": "",
    }


async def test_import_milvus_example() -> None:
    """
    Split the local GDPR example, embed its children, and import every vector.

    This test calls the configured TEI embedding service and Milvus database.
    It intentionally skips key-term extraction because key terms are not
    required to generate or store vectors.
    """
    from app.config.embedding_config import embedding_config
    from app.import_process.agent.nodes.node_bge_embedding import (
        node_bge_embedding,
    )
    from app.import_process.agent.nodes.node_document_split import (
        node_document_split,
    )
    from app.utils.path_util import PROJECT_ROOT

    # Reuse the same real GDPR Markdown example used by the document-splitter
    # and embedding-node manual integration tests.
    test_md_path = os.path.join(
        PROJECT_ROOT,
        "output",
        "General_Data_Protection_Regulation_GDPR",
        "General_Data_Protection_Regulation_TEST.md",
    )
    if not os.path.exists(test_md_path):
        raise FileNotFoundError(
            f"GDPR test Markdown does not exist: {test_md_path}"
        )

    # The splitter reads ``md_path`` because ``md_content`` starts empty,
    # writes ``chunks.json``, and places generated records in state["chunks"].
    test_state: ImportGraphState = {
        "task_id": "test_milvus_import",
        "md_path": test_md_path,
        "md_content": "",
        "file_title": "General_Data_Protection_Regulation_TEST",
        "local_dir": os.path.dirname(test_md_path),
    }

    try:
        split_state = node_document_split(test_state)
        final_state = await node_bge_embedding(split_state)

        # Keep the source text separately because MilvusRecordInfo deliberately
        # excludes text; authoritative text remains in Elasticsearch.
        child_text_by_id = {
            chunk["chunk_id"]: chunk["text"]
            for chunk in final_state["chunks"]
            if chunk.get("record_type") in {"child", "parent_child"}
        }
        embedding_records = final_state.get("embeddings_content", [])

        print(
            f"\nGenerated {len(embedding_records)} child embeddings "
            f"with model {embedding_config.model}.\n"
        )
        for record_number, record in enumerate(
            embedding_records,
            start=1,
        ):
            chunk_id = record["chunk_id"]
            vector = record["dense_vector"]
            print("=" * 100)
            print(f"Child number: {record_number}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Parent ID: {record['parent_id']}")
            print(f"Text: {child_text_by_id.get(chunk_id, '')}")
            print(f"Vector dimension: {len(vector)}")
            print(f"Dense vector: {vector}")

        result = await node_import_milvus(final_state)
        final_state.update(result)

        print(
            f"Imported {result['milvus_imported_count']} record into "
            f"{result['milvus_collection_name']}."
        )
        logger.info(
            f"Manual Milvus import test completed: "
            f"children={len(embedding_records)}, "
            f"dimension={embedding_config.dimension}, "
            f"imported={result['milvus_imported_count']}"
        )
    finally:
        await milvus_client_manager.close()


if __name__ == "__main__":
    asyncio.run(test_import_milvus_example())
