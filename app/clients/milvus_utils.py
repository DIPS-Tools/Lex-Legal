from __future__ import annotations

from typing import Optional

from pymilvus import AsyncMilvusClient

from app.config.milvus_config import MilvusConfig, milvus_config
from app.core.logger import logger


class MilvusClientManager:
    """Create, reuse, and close the asynchronous PyMilvus client."""

    def __init__(self, config: MilvusConfig):
        self.config = config
        self.client: Optional[AsyncMilvusClient] = None

    def init(self) -> None:
        """
        Initialize the client once without performing a database operation.

        ``AsyncMilvusClient`` uses ``uri`` rather than the
        Elasticsearch-style ``hosts`` argument and keeps database I/O from
        blocking the LangGraph event loop.
        """
        if self.client is not None:
            return

        logger.info(
            f"Initializing Milvus client: uri={self.config.milvus_url}"
        )
        self.client = AsyncMilvusClient(
            uri=self.config.milvus_url,
            timeout=self.config.timeout,
        )

    def get_client(self) -> AsyncMilvusClient:
        """Return the initialized client or explain the required setup."""
        if self.client is None:
            raise RuntimeError(
                "Milvus client has not been initialized. "
                "Call milvus_client_manager.init() first."
            )
        return self.client

    async def close(self) -> None:
        """Close the asynchronous client during application shutdown."""
        if self.client is None:
            return

        client = self.client
        self.client = None
        await client.close()


milvus_client_manager = MilvusClientManager(milvus_config)
