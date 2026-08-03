from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class MilvusConfig:
    """Connection and ingestion settings for the Milvus vector database."""

    milvus_url: str
    chunks_collection: str
    keywords_name_collection: str
    document_name_collection: str
    timeout: float = 30
    batch_size: int = 200

    def __post_init__(self) -> None:
        if not self.milvus_url:
            raise ValueError("MILVUS_URL must not be empty.")
        if not self.chunks_collection:
            raise ValueError("CHUNKS_COLLECTION must not be empty.")
        if self.timeout <= 0:
            raise ValueError("MILVUS_TIMEOUT must be greater than zero.")
        if self.batch_size <= 0:
            raise ValueError("MILVUS_BATCH_SIZE must be greater than zero.")


milvus_config = MilvusConfig(
    milvus_url=os.getenv("MILVUS_URL", "http://127.0.0.1:19530"),
    chunks_collection=os.getenv(
        "CHUNKS_COLLECTION",
        "rag_query_law_chunks",
    ),
    keywords_name_collection=os.getenv(
        "KEYWORDS_NAME_COLLECTION",
        "rag_query_law_keywords",
    ),
    document_name_collection=os.getenv(
        "DOCU_NAME_COLLECTION",
        "rag_query_law_ducument_names",
    ),
    timeout=float(os.getenv("MILVUS_TIMEOUT", "30")),
    batch_size=int(os.getenv("MILVUS_BATCH_SIZE", "200")),
)
