from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from elasticsearch import AsyncElasticsearch, BadRequestError
from pydantic import BaseModel

from app.config.es_config import es_config
from app.config.meta_config import meta_config
from app.entities.es_record import (
    ESRecordInfo,
    ESSearchResult,
)


class ESBulkIndexError(RuntimeError):
    """Raised when one or more items in an Elasticsearch bulk request fail."""

    def __init__(self, failures: list[dict[str, Any]]):
        self.failures = failures
        preview = ", ".join(
            str(item.get("chunk_id", "unknown"))
            for item in failures[:10]
        )
        super().__init__(
            f"Elasticsearch bulk indexing failed for "
            f"{len(failures)} records: {preview}"
        )


class ESRepository:
    """Persistence boundary for legal chunk records in Elasticsearch."""

    def __init__(
        self,
        client: AsyncElasticsearch,
        index_name: str | None = None,
    ):
        resolved_index_name = index_name or es_config.index_name
        if not resolved_index_name:
            raise ValueError(
                "Elasticsearch index name is empty. "
                "Set ES_INDEX_NAME."
            )

        self.client = client
        self.index_name = resolved_index_name

    async def ensure_index(self) -> None:
        """Create the index or add newly configured fields to its mapping."""
        mappings = {
            "dynamic": False,
            **meta_config.build_es_mapping(),
        }

        if await self.client.indices.exists(index=self.index_name):
            # Elasticsearch supports additive mapping updates. Sending the
            # two optional hash properties makes them available without
            # touching established legal-text mappings.
            hash_properties = {
                field_name: mappings["properties"][field_name]
                for field_name in ("document_hash", "chunk_hash")
            }
            await self.client.indices.put_mapping(
                index=self.index_name,
                properties=hash_properties,
            )
            return

        try:
            await self.client.indices.create(
                index=self.index_name,
                mappings=mappings,
            )
        except BadRequestError as exc:
            error = (
                exc.body.get("error", {})
                if isinstance(exc.body, dict)
                else {}
            )
            if error.get("type") != "resource_already_exists_exception":
                raise

    @staticmethod
    def _to_record_info(value: Any) -> BaseModel:
        """Convert a chunk dictionary or existing model into ESRecordInfo."""
        if isinstance(value, ESRecordInfo):
            return value
        if isinstance(value, BaseModel):
            value = value.model_dump(exclude_none=True)
        if not isinstance(value, Mapping):
            raise TypeError(
                "Elasticsearch records must be mappings or Pydantic models."
            )
        return ESRecordInfo.from_chunk(dict(value))

    async def bulk_upsert(
        self,
        values: Sequence[Any],
        *,
        batch_size: int = 200,
        refresh: bool = True,
    ) -> int:
        """
        Validate and idempotently index records in bounded bulk requests.

        ``chunk_id`` is used as Elasticsearch ``_id``. Re-importing the same
        law therefore replaces the existing document instead of creating a
        duplicate.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if not values:
            return 0

        success_count = 0
        failures: list[dict[str, Any]] = []

        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            operations: list[dict[str, Any]] = []

            for value in batch:
                record_info = self._to_record_info(value)
                document = record_info.to_document()
                operations.append(
                    {
                        "index": {
                            "_index": self.index_name,
                            "_id": document["chunk_id"],
                        }
                    }
                )
                operations.append(document)

            response = await self.client.bulk(
                operations=operations,
                refresh=False,
            )
            response_body = getattr(response, "body", response)
            items = response_body.get("items", [])

            if len(items) != len(batch):
                raise RuntimeError(
                    "Elasticsearch bulk response item count does not "
                    "match the request batch."
                )

            for item in items:
                result = item.get("index", {})
                status = result.get("status", 500)
                if 200 <= status < 300 and "error" not in result:
                    success_count += 1
                    continue

                failures.append(
                    {
                        "chunk_id": result.get("_id"),
                        "status": status,
                        "error": result.get("error"),
                    }
                )

        if refresh and success_count:
            await self.client.indices.refresh(index=self.index_name)

        if failures:
            raise ESBulkIndexError(failures)

        return success_count

    async def delete_stale_records(
        self,
        law_id: str,
        current_chunk_ids: Sequence[str],
        *,
        refresh: bool = True,
    ) -> int:
        """
        Delete records for one law that are absent from its latest import.

        Upsert replaces matching IDs but cannot remove provisions that
        disappeared after an amendment or splitter change. This cleanup runs
        only after the current records were successfully indexed.
        """
        normalized_law_id = law_id.strip()
        if not normalized_law_id:
            raise ValueError("law_id must not be empty.")

        normalized_ids = list(dict.fromkeys(
            chunk_id.strip()
            for chunk_id in current_chunk_ids
            if isinstance(chunk_id, str) and chunk_id.strip()
        ))
        if not normalized_ids:
            raise ValueError(
                "current_chunk_ids must contain at least one ID."
            )

        response = await self.client.delete_by_query(
            index=self.index_name,
            query={
                "bool": {
                    "filter": [
                        {"term": {"law_id": normalized_law_id}},
                    ],
                    "must_not": [
                        {"ids": {"values": normalized_ids}},
                    ],
                }
            },
            conflicts="proceed",
            refresh=refresh,
        )
        response_body = getattr(response, "body", response)
        if not isinstance(response_body, Mapping):
            raise RuntimeError(
                "Elasticsearch returned an invalid delete-by-query response."
            )
        deleted = response_body.get("deleted")
        if not isinstance(deleted, int):
            raise RuntimeError(
                "Elasticsearch delete-by-query response omitted "
                "the deleted count."
            )
        return deleted

    async def search(
        self,
        keyword: str,
        *,
        score_threshold: float | None = None,
        limit: int = 20,
    ) -> list[ESSearchResult]:
        """Run BM25 search over YAML-configured searchable fields."""
        if not keyword.strip():
            return []
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        request: dict[str, Any] = {
            "index": self.index_name,
            "query": {
                "multi_match": {
                    "query": keyword,
                    "fields": meta_config.es_search_fields(),
                }
            },
            "size": limit,
        }
        if score_threshold is not None:
            request["min_score"] = score_threshold

        response = await self.client.search(**request)
        response_body = getattr(response, "body", response)

        return [
            ESSearchResult(
                record=ESRecordInfo.model_validate(hit["_source"]),
                score=hit.get("_score"),
            )
            for hit in response_body.get("hits", {}).get("hits", [])
        ]

    async def get_by_chunk_ids(
        self,
        chunk_ids: Sequence[str],
    ) -> dict[str, BaseModel]:
        """Fetch authoritative Elasticsearch records keyed by chunk ID."""
        normalized_ids = list(dict.fromkeys(
            chunk_id.strip()
            for chunk_id in chunk_ids
            if isinstance(chunk_id, str) and chunk_id.strip()
        ))
        if not normalized_ids:
            return {}

        response = await self.client.mget(
            index=self.index_name,
            ids=normalized_ids,
        )
        response_body = getattr(response, "body", response)
        if not isinstance(response_body, Mapping):
            raise RuntimeError("Elasticsearch returned an invalid mget response.")

        records: dict[str, BaseModel] = {}
        for document in response_body.get("docs", []):
            if not document.get("found", False):
                continue
            record = ESRecordInfo.model_validate(document.get("_source", {}))
            records[str(record.chunk_id)] = record
        return records
