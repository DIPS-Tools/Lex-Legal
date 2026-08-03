"""
Asynchronous MongoDB storage for query conversation history.

Workflow
========

```text
MongoDBConfig
    -> HistoryMongoTool.init()
    -> AsyncIOMotorClient
    -> database
    -> chat_message collection
    -> compound session/time index

Query workflow
    -> save_chat_message()
    -> HistoryMongoTool.save_chat_message()
    -> MongoDB
```

All database operations in this module are asynchronous. Every caller must use
``await``. This is important because Motor returns awaitable results; treating
them as synchronous PyMongo results causes un-awaited-coroutine warnings and
silently skips database work.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Sequence

from bson import ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)
from pymongo import ASCENDING, DESCENDING

from app.config.mongodb_config import MongoDBConfig, mongodb_config


load_dotenv()


class HistoryMongoTool:
    """
    Own the asynchronous MongoDB resources used for conversation history.

    Creating this Python object does not open a network connection. ``init()``
    performs the connection check and index creation. This lazy lifecycle keeps
    imports, graph construction, and unit tests independent from MongoDB.
    """

    COLLECTION_NAME = "chat_message"
    HISTORY_INDEX_NAME = "session_id_1_ts_-1"
    VALID_ROLES = {"user", "assistant", "system", "tool"}

    def __init__(
        self,
        config_mongodb: MongoDBConfig = mongodb_config,
    ) -> None:
        self.config_mongodb = config_mongodb
        self.db_name = str(config_mongodb.database_name or "").strip()
        self.client: AsyncIOMotorClient | None = None
        self.db: AsyncIOMotorDatabase | None = None
        self.chat_message: AsyncIOMotorCollection | None = None

    def _get_url(self) -> str:
        """
        Return a normalized endpoint without inserting credentials into it.

        ``MONGO_BASE_URL`` may be either ``host:port`` or a complete
        ``mongodb://``/``mongodb+srv://`` URI. Username and password are passed
        separately to Motor so reserved characters do not corrupt the URI and
        secrets are less likely to appear in logs.
        """
        endpoint = str(self.config_mongodb.endpoint or "").strip()
        if not endpoint:
            raise ValueError("MONGO_BASE_URL is required.")
        if not endpoint.startswith(("mongodb://", "mongodb+srv://")):
            endpoint = f"mongodb://{endpoint}"
        return endpoint

    def _validate_config(self) -> None:
        """Reject incomplete configuration before creating a client."""
        self._get_url()
        if not self.db_name:
            raise ValueError("MONGO_DB_NAME is required.")

    def _require_collection(self) -> AsyncIOMotorCollection:
        """Return the initialized collection or raise an actionable error."""
        if self.chat_message is None:
            raise RuntimeError(
                "MongoDB history has not been initialized. "
                "Call await HistoryMongoTool.init() first."
            )
        return self.chat_message

    async def init(self) -> "HistoryMongoTool":
        """
        Connect to MongoDB and create the history query index.

        Motor creates connections lazily, so ``ping`` is awaited to verify the
        server during initialization. The compound index supports filtering by
        ``session_id`` and then reading newest messages by ``ts`` efficiently.
        Calling ``init`` more than once is safe.
        """
        if self.client is not None:
            return self

        self._validate_config()
        client = AsyncIOMotorClient(
            self._get_url(),
            username=self.config_mongodb.username or None,
            password=self.config_mongodb.password or None,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )

        try:
            await client.admin.command("ping")
            database = client[self.db_name]
            collection = database[self.COLLECTION_NAME]

            # This compound index matches get_recent_messages(): first filter
            # one session, then read that session's newest timestamps. Naming
            # the index also makes MongoDB administration easier.
            await collection.create_index(
                [
                    ("session_id", ASCENDING),
                    ("ts", DESCENDING),
                ],
                name=self.HISTORY_INDEX_NAME,
            )
        except Exception:
            client.close()
            logging.exception(
                "Failed to initialize MongoDB conversation history."
            )
            raise

        self.client = client
        self.db = database
        self.chat_message = collection
        logging.info(
            "MongoDB conversation history initialized: "
            f"database={self.db_name}, collection={self.COLLECTION_NAME}"
        )
        return self

    def get_client(self) -> AsyncIOMotorClient:
        """Return the initialized Motor client."""
        if self.client is None:
            raise RuntimeError(
                "MongoDB client has not been initialized. "
                "Call await HistoryMongoTool.init() first."
            )
        return self.client

    def close(self) -> None:
        """Close MongoDB resources safely; repeated calls are allowed."""
        if self.client is not None:
            self.client.close()
        self.client = None
        self.db = None
        self.chat_message = None

    async def clear_history(self, session_id: str) -> int:
        """Delete every message belonging to one session."""
        normalized_session_id = self._required_text(
            session_id,
            "session_id",
        )
        result = await self._require_collection().delete_many(
            {"session_id": normalized_session_id}
        )
        logging.info(
            f"Deleted {result.deleted_count} messages for "
            f"session={normalized_session_id}"
        )
        return int(result.deleted_count)

    async def save_chat_message(
        self,
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        query_key_terms: Sequence[str] | None = None,
        image_urls: list[str] | None = None,
        message_id: str | None = None,
    ) -> str:
        """
        Insert one message, or replace its mutable fields when an ID is given.

        Timestamps are stored as timezone-aware UTC datetimes. BSON supports
        datetimes natively and MongoDB can sort and filter them efficiently.
        """
        normalized_session_id = self._required_text(
            session_id,
            "session_id",
        )
        normalized_role = self._required_text(role, "role").lower()
        if normalized_role not in self.VALID_ROLES:
            raise ValueError(
                f"Unsupported role {role!r}. "
                f"Expected one of {sorted(self.VALID_ROLES)}."
            )

        document = {
            "session_id": normalized_session_id,
            "role": normalized_role,
            "text": self._required_text(text, "text"),
            "rewritten_query": str(rewritten_query or "").strip(),
            "query_key_terms": self._normalize_string_list(query_key_terms),
            "image_urls": self._normalize_string_list(image_urls),
            "ts": datetime.now(timezone.utc),
        }
        collection = self._require_collection()

        if message_id:
            object_id = self._object_id(message_id)
            result = await collection.update_one(
                {"_id": object_id},
                {"$set": document},
            )
            if result.matched_count == 0:
                raise LookupError(
                    f"MongoDB history message does not exist: {message_id}"
                )
            return str(object_id)

        result = await collection.insert_one(document)
        return str(result.inserted_id)

    async def update_message_query_key_terms(
        self,
        ids: Sequence[str],
        query_key_terms: Sequence[str],
    ) -> int:
        """Set legal item/law names on selected history messages."""
        object_ids = [self._object_id(value) for value in ids]
        if not object_ids:
            return 0

        result = await self._require_collection().update_many(
            {"_id": {"$in": object_ids}},
            {"$set": {"query_key_terms": self._normalize_string_list(query_key_terms)}},
        )
        return int(result.modified_count)

    async def get_recent_messages(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Return the newest N messages in chronological order.

        MongoDB first reads newest-to-oldest so ``limit`` selects the most
        recent records. Python then reverses them into old-to-new order for an
        LLM conversation prompt.
        """
        normalized_session_id = self._required_text(
            session_id,
            "session_id",
        )
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer.")

        cursor = (
            self._require_collection()
            .find({"session_id": normalized_session_id})
            .sort("ts", DESCENDING)
            .limit(limit)
        )
        messages = await cursor.to_list(length=limit)
        messages.reverse()
        return [self._serialize_document(message) for message in messages]

    async def save_langchain_messages(
        self,
        session_id: str,
        messages: Sequence[Any],
    ) -> list[str]:
        """
        Save LangChain ``BaseMessage`` objects through the common schema.

        LangChain performs object-to-dictionary conversion; this class stores
        only BSON-compatible data. Importing the helpers locally keeps MongoDB
        usable even when LangChain message conversion is not needed.
        """
        from langchain_core.messages import messages_to_dict

        role_by_message_type = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }
        message_ids: list[str] = []
        for message in messages_to_dict(list(messages)):
            data = message.get("data", {})
            message_type = str(message.get("type", ""))
            role = role_by_message_type.get(message_type)
            if role is None:
                raise ValueError(
                    f"Unsupported LangChain message type: {message_type!r}"
                )
            message_ids.append(
                await self.save_chat_message(
                    session_id=session_id,
                    role=role,
                    text=str(data.get("content", "")),
                )
            )
        return message_ids

    async def get_recent_langchain_messages(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[Any]:
        """Read MongoDB records and restore LangChain message objects."""
        from langchain_core.messages import messages_from_dict

        documents = await self.get_recent_messages(session_id, limit)
        message_type_by_role = {
            "user": "human",
            "assistant": "ai",
            "system": "system",
            "tool": "tool",
        }
        message_dicts = [
            {
                "type": message_type_by_role[document["role"]],
                "data": {"content": document["text"]},
            }
            for document in documents
        ]
        return messages_from_dict(message_dicts)

    @staticmethod
    def _required_text(value: Any, field_name: str) -> str:
        """Normalize a required string field."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")
        return value.strip()

    @staticmethod
    def _normalize_string_list(
        values: Sequence[str] | None,
    ) -> list[str]:
        """Normalize optional string lists and remove duplicates."""
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values or []:
            if not isinstance(value, str) or not value.strip():
                continue
            cleaned = value.strip()
            if cleaned not in seen:
                seen.add(cleaned)
                normalized.append(cleaned)
        return normalized

    @staticmethod
    def _object_id(value: str) -> ObjectId:
        """Validate and convert a public string ID into a BSON ObjectId."""
        if not ObjectId.is_valid(value):
            raise ValueError(f"Invalid MongoDB message_id: {value!r}")
        return ObjectId(value)

    @staticmethod
    def _serialize_document(document: dict[str, Any]) -> dict[str, Any]:
        """Convert BSON-only values into FastAPI JSON-compatible values."""
        serialized = dict(document)
        if "_id" in serialized:
            serialized["_id"] = str(serialized["_id"])
        timestamp = serialized.get("ts")
        if isinstance(timestamp, datetime):
            serialized["ts"] = timestamp.isoformat()
        return serialized


_history_mongo_tool: HistoryMongoTool | None = None
_history_mongo_lock = asyncio.Lock()


async def get_history_mongo_tool() -> HistoryMongoTool:
    """Return one lazily initialized history client per application process."""
    global _history_mongo_tool
    if _history_mongo_tool is None:
        async with _history_mongo_lock:
            if _history_mongo_tool is None:
                tool = HistoryMongoTool(mongodb_config)
                await tool.init()
                _history_mongo_tool = tool
    return _history_mongo_tool


async def clear_history(session_id: str) -> int:
    """Compatibility helper for deleting one session's history."""
    tool = await get_history_mongo_tool()
    return await tool.clear_history(session_id)


async def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    query_key_terms: Sequence[str] | None = None,
    image_urls: list | None = None,
    message_id: str | None = None,
) -> str:
    """Compatibility helper for inserting or updating one message."""
    tool = await get_history_mongo_tool()
    return await tool.save_chat_message(
        session_id=session_id,
        role=role,
        text=text,
        rewritten_query=rewritten_query,
        query_key_terms=query_key_terms,
        image_urls=image_urls,
        message_id=message_id,
    )


async def update_message_query_key_terms(
    ids: Sequence[str],
    query_key_terms: Sequence[str],
) -> int:
    """Compatibility helper for updating message metadata."""
    tool = await get_history_mongo_tool()
    return await tool.update_message_query_key_terms(ids, query_key_terms)


async def get_recent_messages(
    session_id: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Compatibility helper for reading recent JSON-safe messages."""
    tool = await get_history_mongo_tool()
    return await tool.get_recent_messages(session_id, limit)


async def _manual_test() -> None:
    """Run a small integration test against the configured MongoDB server."""
    session_id = "history_mongo_manual_test"
    await save_chat_message(session_id, "user", "What is GDPR?")
    await save_chat_message(
        session_id,
        "assistant",
        "GDPR is the EU General Data Protection Regulation.",
    )
    print(await get_recent_messages(session_id, limit=5))
    # print(f"Deleted: {await clear_history(session_id)}")


if __name__ == "__main__":
    asyncio.run(_manual_test())
