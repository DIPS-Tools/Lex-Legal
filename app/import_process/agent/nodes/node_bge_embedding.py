from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from app.clients.embedding_utils import embedding_client_manager
from app.config.embedding_config import embedding_config
from app.config.meta_config import meta_config
from app.core.logger import logger
from app.entities.milvus_record import MilvusRecordInfo
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task


def _children_for_embedding(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select searchable child records with non-empty authoritative text.
    get all child records with non-empty authoritative text.
    """

    children: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise TypeError("Every item in state['chunks'] must be a dictionary.")
        if chunk.get("record_type") not in {"child", "parent_child"}:
            continue
        if not chunk.get("text", "").strip():
            raise ValueError(
                f"Child chunk {chunk.get('chunk_id', '<unknown>')} "
                "has empty text."
            )
        children.append(chunk)
    return children


def _validate_embedding_dimension() -> int:
    """Ensure TEI output and Milvus collection dimensions use one contract."""
    vector_config = meta_config.get_milvus_field("dense_vector")
    milvus_dimension = vector_config.dimension
    if milvus_dimension is None:
        raise ValueError("Milvus dense_vector dimension is not configured.")
    if embedding_config.dimension != milvus_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: embedding model is configured "
            f"for {embedding_config.dimension}, but Milvus expects "
            f"{milvus_dimension}. Set EMBEDDING_DIMENSION to match the "
            f"model and configs/meta_config.yml."
        )
    return milvus_dimension


async def node_bge_embedding(
    state: ImportGraphState,
) -> ImportGraphState:
    """
    Generate dense embeddings for legal child chunks using the TEI endpoint.

    Parent records remain in Elasticsearch and are not embedded. Exact
    Milvus-ready records are written to ``state["embeddings_content"]``.
    """
    function_name = sys._getframe().f_code.co_name
    task_id = state.get("task_id", "unknown")

    # chunks are need to embedd
    chunks = state.get("chunks", [])

    logger.info(
        f">>> [{function_name}] Started. "
        f"task_id={task_id}, chunks={len(chunks)}"
    )
    add_running_task(task_id, function_name)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("state['chunks'] must be a non-empty list.")
    # check embedding dimension, let milvus and embedding mode same
    _validate_embedding_dimension()

    # where children chunks are needed to embedding
    children = _children_for_embedding(chunks)

    if not children:
        raise ValueError("No child chunks are available for embedding.")

    # get embedding client
    embedding_client_manager.init()
    client = embedding_client_manager.get_client()
    # record embeddings
    embedding_records: list[dict[str, Any]] = []

    try:
        for start in range(0, len(children), embedding_config.batch_size):
            batch = children[start:start + embedding_config.batch_size]
            texts = [chunk["text"] for chunk in batch]
            vectors = await client.aembed_documents(texts)

            if len(vectors) != len(batch):
                raise ValueError(
                    "Embedding result count does not match the child batch."
                )
            # This code combines each legal child chunk with its corresponding embedding vector.
            for chunk, vector in zip(batch, vectors, strict=True):
                """
                  - batch contains child chunk dictionaries.
                  - vectors contains embedding vectors returned by the embedding model.
                  - zip() pairs them by position:
                      - first chunk → first vector
                      - second chunk → second vector
                """
                record = MilvusRecordInfo.from_embedding_record(
                    {
                        **chunk,
                        "dense_vector": vector,
                    }
                )
                """
                  This creates a new dictionary:
                
                  - **chunk copies all chunk fields.
                  - "dense_vector": vector adds the generated vector.
                  - The original chunk dictionary is not modified.
                
                  For example:
                
                  {
                      "chunk_id": "gdpr:article_6:paragraph_1",
                      "parent_id": "gdpr:article_6",
                      "text": "Processing shall be lawful...",
                      "dense_vector": [0.12, -0.03, ...],
                  }
                """
                embedding_records.append(record.to_entity())

            logger.info(
                f"Embedded child batch: "
                f"start={start}, count={len(batch)}"
            )
    except Exception as exc:
        state["embedding_success"] = False
        state["embedding_error"] = str(exc)
        logger.exception(
            f"Embedding node failed. task_id={task_id}"
        )
        raise

    state["embeddings_content"] = embedding_records
    state["embedding_success"] = True
    state["embedded_count"] = len(embedding_records)
    state["embedding_error"] = ""

    logger.info(
        f">>> [{function_name}] Finished. "
        f"model={embedding_config.model}, "
        f"dimension={embedding_config.dimension}, "
        f"embedded_children={len(embedding_records)}"
    )
    add_done_task(task_id, function_name)
    return state




async def test_embedding_example() -> None:
    """
    Split the local GDPR example, embed its children, and print every vector.

    This test calls the configured TEI embedding service. It intentionally skips
    key-term extraction because key terms are not required to generate vectors.
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
    for record_number, record in enumerate(embedding_records, start=1):
        chunk_id = record["chunk_id"]
        vector = record["dense_vector"]
        print("=" * 100)
        print(f"Child number: {record_number}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Parent ID: {record['parent_id']}")
        print(f"Text: {child_text_by_id.get(chunk_id, '')}")
        print(f"Vector dimension: {len(vector)}")
        print(f"Dense vector: {vector}")

    logger.info(
        f"Manual embedding test completed: "
        f"children={len(embedding_records)}, "
        f"dimension={embedding_config.dimension}"
    )


if __name__ == "__main__":
    # asyncio.run creates an event loop and executes the async TEI test.
    asyncio.run(test_embedding_example())
