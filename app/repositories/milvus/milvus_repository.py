from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel
from pymilvus import AsyncMilvusClient, DataType, MilvusClient

from app.config.meta_config import MilvusFieldConfig, meta_config
from app.config.milvus_config import milvus_config
from app.entities.milvus_record import MilvusRecordInfo


class MilvusUpsertError(RuntimeError):
    """Raised when Milvus does not confirm every requested upsert."""


class MilvusSchemaMismatchError(RuntimeError):
    """Raised when an existing collection cannot store current records."""


class MilvusRepository:
    """Persistence boundary for YAML-defined legal embedding records."""

    def __init__(
        self,
        client: AsyncMilvusClient,
        collection_name: str | None = None,
    ):
        resolved_name = collection_name or milvus_config.chunks_collection
        if not resolved_name:
            raise ValueError(
                "Milvus collection name is empty. Set CHUNKS_COLLECTION."
            )

        self.client = client
        self.collection_name = resolved_name

    @staticmethod
    def _data_type(field_config: MilvusFieldConfig) -> DataType:
        """Map the project YAML types to PyMilvus schema types."""
        data_types = {
            "VARCHAR": DataType.VARCHAR,
            "ARRAY_VARCHAR": DataType.ARRAY,
            "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
        }
        try:
            return data_types[field_config.data_type]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported Milvus field type: "
                f"{field_config.data_type!r}"
            ) from exc

    @classmethod
    def build_schema(cls):
        """Build a strict Milvus collection schema from meta_config.yml."""
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )

        for field_config in meta_config.milvus_meta:
            field_options: dict[str, Any] = {
                "is_primary": field_config.primary_key,
                "description": field_config.description,
            }
            if (
                not field_config.required
                and field_config.default is None
            ):
                # Legal metadata such as chapter_number and article_number is
                # legitimately absent for recitals and some annex chunks.
                field_options["nullable"] = True

            if field_config.data_type == "VARCHAR":
                field_options["max_length"] = field_config.max_length
            elif field_config.data_type == "ARRAY_VARCHAR":
                field_options.update(
                    {
                        "element_type": DataType.VARCHAR,
                        "max_capacity": field_config.max_capacity,
                        "max_length": field_config.max_length,
                    }
                )
            elif field_config.data_type == "FLOAT_VECTOR":
                field_options["dim"] = field_config.dimension

            schema.add_field(
                field_name=field_config.name,
                datatype=cls._data_type(field_config),
                **field_options,
            )

        return schema

    @staticmethod
    def build_index_params():
        """Build configured vector indexes from meta_config.yml."""
        index_params = MilvusClient.prepare_index_params()

        for field_config in meta_config.milvus_vector_fields():
            if not field_config.index_type:
                continue
            # build index for the column of dense_vector
            index_params.add_index(
                # which one colum is need to add index
                field_name=field_config.name,
                # index name
                index_name=f"{field_config.name}_index",
                # the algriam is used to do search
                index_type=field_config.index_type,
                # similary compution mothed
                metric_type=field_config.metric_type,
                params=dict(field_config.index_params),
            )

        return index_params

    @classmethod
    def validate_existing_schema(
        cls,
        description: Mapping[str, Any],
    ) -> None:
        """Reject legacy collections that conflict with the YAML contract."""
        actual_fields = {
            field.get("name", field.get("field_name")): field
            for field in description.get("fields", [])
        }
        expected_names = set(meta_config.milvus_field_map())
        actual_names = {
            name
            for name in actual_fields
            if isinstance(name, str)
        }
        if actual_names != expected_names:
            raise MilvusSchemaMismatchError(
                "Existing Milvus collection fields do not match "
                "configs/meta_config.yml. "
                f"Expected {sorted(expected_names)}, "
                f"found {sorted(actual_names)}."
            )

        primary_config = next(
            field_config
            for field_config in meta_config.milvus_meta
            if field_config.primary_key
        )
        primary_field = actual_fields[primary_config.name]
        actual_primary_type = primary_field.get(
            "type",
            primary_field.get("datatype"),
        )
        if (
            not primary_field.get("is_primary")
            or actual_primary_type != cls._data_type(primary_config)
        ):
            raise MilvusSchemaMismatchError(
                "Existing Milvus collection must use string field "
                f"{primary_config.name!r} as its primary key."
            )

        vector_config = meta_config.get_milvus_field("dense_vector")
        vector_field = actual_fields[vector_config.name]
        vector_params = vector_field.get("params", {})
        actual_dimension = vector_params.get(
            "dim",
            vector_field.get("dim"),
        )
        try:
            actual_dimension = int(actual_dimension)
        except (TypeError, ValueError):
            actual_dimension = None
        if actual_dimension != vector_config.dimension:
            raise MilvusSchemaMismatchError(
                "Existing Milvus dense_vector dimension does not match "
                f"the configured dimension {vector_config.dimension}; "
                f"found {actual_dimension!r}."
            )

    async def ensure_collection(self) -> None:
        """Create, validate, or safely extend the configured collection."""
        if await self.client.has_collection(
            collection_name=self.collection_name
        ):
            description = await self.client.describe_collection(
                collection_name=self.collection_name
            )
            description = await self._add_missing_hash_fields(description)
            self.validate_existing_schema(description)
            return

        try:
            await self.client.create_collection(
                collection_name=self.collection_name,
                schema=self.build_schema(),
                index_params=self.build_index_params(),
            )
        except Exception:
            if await self.client.has_collection(
                collection_name=self.collection_name
            ):
                description = await self.client.describe_collection(
                    collection_name=self.collection_name
                )
                description = await self._add_missing_hash_fields(description)
                self.validate_existing_schema(description)
                return
            raise

    async def _add_missing_hash_fields(
        self,
        description: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """
        Add optional hash fields to collections created by older releases.

        Milvus allows nullable scalar fields to be added without recreating the
        collection. Only the two known SHA-256 metadata fields are migrated;
        every other schema difference remains a hard compatibility error.
        """
        actual_names = {
            field.get("name", field.get("field_name"))
            for field in description.get("fields", [])
        }
        hash_field_names = ("document_hash", "chunk_hash")
        expected_names = set(meta_config.milvus_field_map())
        missing_non_hash_fields = (
            expected_names - actual_names - set(hash_field_names)
        )
        if missing_non_hash_fields:
            # Do not mutate an unrelated legacy schema. The normal validator
            # will raise a complete expected-versus-actual error.
            return description

        for field_name in hash_field_names:
            if field_name in actual_names:
                continue

            field_config = meta_config.get_milvus_field(field_name)
            if (
                field_config.required
                or field_config.primary_key
                or field_config.data_type != "VARCHAR"
            ):
                raise MilvusSchemaMismatchError(
                    f"Hash field {field_name!r} must be an optional VARCHAR."
                )

            await self.client.add_collection_field(
                collection_name=self.collection_name,
                field_name=field_name,
                data_type=self._data_type(field_config),
                desc=field_config.description,
                nullable=True,
                max_length=field_config.max_length,
            )

        if all(field_name in actual_names for field_name in hash_field_names):
            return description
        return await self.client.describe_collection(
            collection_name=self.collection_name
        )

    @staticmethod
    def _to_record_info(value: Any) -> BaseModel:
        """Convert a dictionary or model into a validated MilvusRecordInfo."""
        if isinstance(value, MilvusRecordInfo):
            return value
        if isinstance(value, BaseModel):
            value = value.model_dump(exclude_none=True)
        if not isinstance(value, Mapping):
            raise TypeError(
                "Milvus records must be mappings or Pydantic models."
            )
        return MilvusRecordInfo.from_embedding_record(dict(value))

    @classmethod
    def _to_entity(cls, value: Any) -> dict[str, Any]:
        """
        Return a complete entity compatible with old and new collections.

        Collections created before optional fields were marked nullable still
        reject omitted values. Empty strings preserve "not applicable" for
        optional VARCHAR metadata while allowing those existing collections to
        accept recital and annex records safely.
        """
        entity = cls._to_record_info(value).to_entity()
        for field_config in meta_config.milvus_meta:
            if field_config.name in entity or field_config.required:
                continue
            if field_config.data_type == "VARCHAR":
                entity[field_config.name] = ""
            elif field_config.data_type == "ARRAY_VARCHAR":
                entity[field_config.name] = []
        return entity

    async def bulk_upsert(
        self,
        values: Sequence[Any],
        *,
        batch_size: int | None = None,
    ) -> int:
        """
        Validate and idempotently upsert embedding records in bounded batches.

        The YAML schema defines ``chunk_id`` as the string primary key.
        Re-importing the same legal chunk therefore replaces its vector and
        metadata instead of creating a duplicate.
        """
        resolved_batch_size = batch_size or milvus_config.batch_size
        if resolved_batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if not values:
            return 0

        success_count = 0
        for start in range(0, len(values), resolved_batch_size):
            batch = values[start:start + resolved_batch_size]
            entities = [
                self._to_entity(value)
                for value in batch
            ]
            response = await self.client.upsert(
                collection_name=self.collection_name,
                data=entities,
            )
            response_body = getattr(response, "body", response)
            if not isinstance(response_body, Mapping):
                raise MilvusUpsertError(
                    "Milvus returned an invalid upsert response."
                )

            confirmed_count = response_body.get(
                "upsert_count",
                response_body.get("insert_count"),
            )
            if confirmed_count != len(entities):
                raise MilvusUpsertError(
                    f"Milvus confirmed {confirmed_count!r} of "
                    f"{len(entities)} requested upserts."
                )
            success_count += confirmed_count

        return success_count

    async def get_by_chunk_ids(
        self,
        chunk_ids: Sequence[str],
        *,
        output_fields: list[str] | None = None,
        batch_size: int = 200,
    ) -> list[dict[str, Any]]:
        """Read records by their stable string primary keys."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        normalized_ids = list(dict.fromkeys(
            chunk_id
            for chunk_id in chunk_ids
            if isinstance(chunk_id, str) and chunk_id
        ))
        if not normalized_ids:
            return []

        if output_fields is None:
            output_fields = [
                field_config.name
                for field_config in meta_config.milvus_scalar_fields()
            ]

        results: list[dict[str, Any]] = []
        for start in range(0, len(normalized_ids), batch_size):
            batch = normalized_ids[start:start + batch_size]
            records = await self.client.get(
                collection_name=self.collection_name,
                ids=batch,
                output_fields=output_fields,
            )
            results.extend(records or [])
        return results

    async def delete_stale_records(
        self,
        law_id: str,
        current_chunk_ids: Sequence[str],
    ) -> int:
        """
        Delete vectors for one law that are absent from its latest import.

        Querying existing primary keys and deleting the set difference avoids a
        very large ``not in`` filter and keeps cleanup independent of vector
        index settings.
        """
        normalized_law_id = law_id.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", normalized_law_id):
            raise ValueError(
                "law_id may contain only letters, numbers, '_' and '-'."
            )

        normalized_ids = {
            chunk_id.strip()
            for chunk_id in current_chunk_ids
            if isinstance(chunk_id, str) and chunk_id.strip()
        }
        if not normalized_ids:
            raise ValueError(
                "current_chunk_ids must contain at least one ID."
            )

        records = await self.client.query(
            collection_name=self.collection_name,
            filter=f'law_id == "{normalized_law_id}"',
            output_fields=["chunk_id"],
        )
        stale_ids = sorted({
            str(record["chunk_id"])
            for record in records or []
            if record.get("chunk_id") not in normalized_ids
        })
        if not stale_ids:
            return 0

        response = await self.client.delete(
            collection_name=self.collection_name,
            ids=stale_ids,
        )
        response_body = getattr(response, "body", response)
        if not isinstance(response_body, Mapping):
            raise RuntimeError("Milvus returned an invalid delete response.")
        deleted = response_body.get("delete_count")
        if deleted != len(stale_ids):
            raise RuntimeError(
                f"Milvus confirmed deletion of {deleted!r} of "
                f"{len(stale_ids)} stale records."
            )
        return deleted

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 20,
        filter_expression: str = "",
        output_fields: list[str] | None = None,
        score_threshold: float | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search legal child records by dense-vector similarity.

        The returned dictionaries flatten Milvus's nested ``entity`` payload
        and add a normalized ``score`` field. They therefore contain the stable
        ``chunk_id`` needed to fuse these results with Elasticsearch BM25
        results using reciprocal-rank fusion.
        """
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")
        if limit > 16_384:
            raise ValueError("limit must not exceed 16,384.")
        if not isinstance(filter_expression, str):
            raise TypeError("filter_expression must be a string.")

        vector_config = meta_config.get_milvus_field("dense_vector")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            for value in query_vector
        ):
            raise TypeError("query_vector must contain only numbers.")
        vector = [float(value) for value in query_vector]
        if vector_config.dimension is None:
            raise ValueError("Milvus dense_vector dimension is not configured.")
        if len(vector) != vector_config.dimension:
            raise ValueError(
                f"query_vector requires dimension="
                f"{vector_config.dimension}, got {len(vector)}."
            )
        if any(not math.isfinite(value) for value in vector):
            raise ValueError("query_vector contains a non-finite value.")
        if (
            score_threshold is not None
            and not math.isfinite(score_threshold)
        ):
            raise ValueError("score_threshold must be finite.")

        if output_fields is None:
            output_fields = [
                field_config.name
                for field_config in meta_config.milvus_scalar_fields()
            ]
        unknown_output_fields = sorted(
            set(output_fields) - set(meta_config.milvus_field_map())
        )
        if unknown_output_fields:
            raise ValueError(
                "Unknown Milvus output fields: "
                + ", ".join(unknown_output_fields)
            )

        resolved_search_params = search_params or {
            "metric_type": vector_config.metric_type or "COSINE",
            "params": {
                "ef": max(64, limit),
            },
        }
        configured_metric = (vector_config.metric_type or "COSINE").upper()
        requested_metric = str(
            resolved_search_params.get("metric_type", configured_metric)
        ).upper()
        if requested_metric != configured_metric:
            raise ValueError(
                f"Search metric {requested_metric!r} does not match "
                f"the indexed metric {configured_metric!r}."
            )

        await self.client.load_collection(
            collection_name=self.collection_name
        )
        response = await self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field=vector_config.name,
            filter=filter_expression,
            limit=limit,
            output_fields=output_fields,
            search_params=resolved_search_params,
        )
        if not response:
            return []

        results: list[dict[str, Any]] = []
        for hit in response[0]:
            entity = dict(hit.get("entity") or {})
            chunk_id = entity.get("chunk_id", hit.get("id"))
            score = hit.get("distance", hit.get("score"))
            if chunk_id is None or score is None:
                raise RuntimeError(
                    "Milvus search result is missing chunk_id or score."
                )

            normalized_score = float(score)
            if (
                score_threshold is not None
                and normalized_score < score_threshold
            ):
                continue

            results.append(
                {
                    **entity,
                    "chunk_id": str(chunk_id),
                    "score": normalized_score,
                }
            )

        return results
