from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config.config_loader import load_config


class MetaConfigError(ValueError):
    """Raised when meta_config.yml is missing or invalid."""


@dataclass
class ESFieldConfig:
    name: str
    description: str = ""
    data_type: str = "keyword"

    example: Any = None

    required: bool = False
    default: Any = None
    allowed_values: list[Any] = field(default_factory=list)

    searchable: bool = False
    filterable: bool = False
    sortable: bool = False
    aggregatable: bool = False

    keyword_subfield: bool = False
    boost: float = 1.0




@dataclass
class MilvusFieldConfig:
    name: str
    description: str = ""
    data_type: str = "VARCHAR"
    example: Any = None


    required: bool = False
    default: Any = None
    allowed_values: list[Any] = field(default_factory=list)

    primary_key: bool = False
    filterable: bool = False

    max_length: int | None = None
    max_capacity: int | None = None

    dimension: int | None = None
    metric_type: str | None = None
    index_type: str | None = None
    index_params: dict[str, Any] = field(default_factory=dict)





@dataclass
class MetaConfig:
    version: str = "1.0"
    es_meta: list[ESFieldConfig] = field(default_factory=list)
    milvus_meta: list[MilvusFieldConfig] = field(default_factory=list)

    def es_field_map(self) -> dict[str, ESFieldConfig]:
        """Return Elasticsearch fields indexed by field name."""
        return {
            item.name: item
            for item in self.es_meta
        }

    def milvus_field_map(self) -> dict[str, MilvusFieldConfig]:
        """Return Milvus fields indexed by field name."""
        return {
            item.name: item
            for item in self.milvus_meta
        }

    def get_es_field(self, name: str) -> ESFieldConfig:
        """Get one Elasticsearch field configuration."""
        try:
            return self.es_field_map()[name]
        except KeyError as exc:
            raise MetaConfigError(
                f"Unknown Elasticsearch metadata field: {name!r}"
            ) from exc

    def get_milvus_field(self, name: str) -> MilvusFieldConfig:
        """Get one Milvus field configuration."""
        try:
            return self.milvus_field_map()[name]
        except KeyError as exc:
            raise MetaConfigError(
                f"Unknown Milvus metadata field: {name!r}"
            ) from exc

    def es_search_fields(self) -> list[str]:
        """
        Return searchable Elasticsearch fields with boosts.

        Example:
            [
                "law_title^2",
                "article_title^4",
                "key_terms^2.5",
                "text"
            ]
        """
        search_fields: list[str] = []

        for item in self.es_meta:
            if not item.searchable:
                continue

            if item.boost != 1.0:
                search_fields.append(
                    f"{item.name}^{item.boost:g}"
                )
            else:
                search_fields.append(item.name)

        return search_fields

    def build_es_mapping(self) -> dict[str, Any]:
        """Build Elasticsearch index mappings from configured ES fields."""
        properties: dict[str, Any] = {}

        for item in self.es_meta:
            properties[item.name] = self._build_es_field_mapping(item)

        return {
            "properties": properties,
        }

    def validate_es_record(
        self,
        record: dict[str, Any],
        *,
        reject_unknown: bool = True,
    ) -> dict[str, Any]:
        """
        Validate and normalize one Elasticsearch document.

        Required configured fields must be present unless they define a default.
        Optional fields with defaults are populated. Unknown fields are either
        rejected or removed depending on reject_unknown.
        """
        if not isinstance(record, dict):
            raise MetaConfigError(
                "Elasticsearch record must be a dictionary."
            )

        field_map = self.es_field_map()
        validated: dict[str, Any] = {}

        unknown_fields = sorted(
            name
            for name in record
            if name not in field_map
        )

        if unknown_fields and reject_unknown:
            raise MetaConfigError(
                "Unknown Elasticsearch record fields: "
                + ", ".join(unknown_fields)
            )

        for field_name, field_config in field_map.items():
            value = record.get(field_name)

            if value is None:
                if field_config.default is not None:
                    value = deepcopy(field_config.default)
                elif field_config.required:
                    raise MetaConfigError(
                        "Missing required Elasticsearch field: "
                        f"{field_name!r}"
                    )
                else:
                    continue

            self._validate_es_field_value(
                field_config,
                value,
            )
            validated[field_name] = value

        return validated

    def milvus_vector_fields(self) -> list[MilvusFieldConfig]:
        """Return all Milvus vector fields."""
        vector_types = {
            "FLOAT_VECTOR",
            "BINARY_VECTOR",
            "FLOAT16_VECTOR",
            "BFLOAT16_VECTOR",
            "SPARSE_FLOAT_VECTOR",
        }

        return [
            item
            for item in self.milvus_meta
            if item.data_type in vector_types
        ]

    def milvus_scalar_fields(self) -> list[MilvusFieldConfig]:
        """Return all non-vector Milvus fields."""
        vector_names = {
            item.name
            for item in self.milvus_vector_fields()
        }

        return [
            item
            for item in self.milvus_meta
            if item.name not in vector_names
        ]

    def validate_milvus_record(
        self,
        record: dict[str, Any],
        *,
        reject_unknown: bool = True,
    ) -> dict[str, Any]:
        """Validate and normalize one Milvus entity from configured fields."""
        if not isinstance(record, dict):
            raise MetaConfigError(
                "Milvus record must be a dictionary."
            )

        field_map = self.milvus_field_map()
        validated: dict[str, Any] = {}
        unknown_fields = sorted(
            name
            for name in record
            if name not in field_map
        )

        if unknown_fields and reject_unknown:
            raise MetaConfigError(
                "Unknown Milvus record fields: "
                + ", ".join(unknown_fields)
            )

        for field_name, field_config in field_map.items():
            value = record.get(field_name)

            if value is None:
                if field_config.default is not None:
                    value = deepcopy(field_config.default)
                elif field_config.required:
                    raise MetaConfigError(
                        "Missing required Milvus field: "
                        f"{field_name!r}"
                    )
                else:
                    continue

            self._validate_milvus_field_value(
                field_config,
                value,
            )
            validated[field_name] = value

        return validated

    def validate(self) -> None:
        """Validate the loaded metadata configuration."""
        self._validate_duplicate_names()
        self._validate_milvus_primary_key()
        self._validate_milvus_fields()

    def _validate_duplicate_names(self) -> None:
        es_names = [
            item.name
            for item in self.es_meta
        ]
        milvus_names = [
            item.name
            for item in self.milvus_meta
        ]

        duplicate_es = sorted({
            name
            for name in es_names
            if es_names.count(name) > 1
        })

        duplicate_milvus = sorted({
            name
            for name in milvus_names
            if milvus_names.count(name) > 1
        })

        if duplicate_es:
            raise MetaConfigError(
                "Duplicate Elasticsearch fields: "
                + ", ".join(duplicate_es)
            )

        if duplicate_milvus:
            raise MetaConfigError(
                "Duplicate Milvus fields: "
                + ", ".join(duplicate_milvus)
            )

    def _validate_milvus_primary_key(self) -> None:
        primary_keys = [
            item.name
            for item in self.milvus_meta
            if item.primary_key
        ]

        if len(primary_keys) != 1:
            raise MetaConfigError(
                "Exactly one Milvus field must have "
                "'primary_key: true'. "
                f"Found: {primary_keys}"
            )

    def _validate_milvus_fields(self) -> None:
        for item in self.milvus_meta:
            if (
                item.data_type == "VARCHAR"
                and item.max_length is None
            ):
                raise MetaConfigError(
                    f"Milvus VARCHAR field {item.name!r} "
                    "requires max_length."
                )

            if (
                item.data_type == "ARRAY_VARCHAR"
                and item.max_length is None
            ):
                raise MetaConfigError(
                    f"Milvus ARRAY_VARCHAR field {item.name!r} "
                    "requires max_length."
                )

            if (
                item.data_type == "ARRAY_VARCHAR"
                and item.max_capacity is None
            ):
                raise MetaConfigError(
                    f"Milvus ARRAY_VARCHAR field {item.name!r} "
                    "requires max_capacity."
                )

            if (
                item.data_type == "FLOAT_VECTOR"
                and item.dimension is None
            ):
                raise MetaConfigError(
                    f"Milvus FLOAT_VECTOR field {item.name!r} "
                    "requires dimension."
                )

    def _build_es_field_mapping(
        self,
        item: ESFieldConfig,
    ) -> dict[str, Any]:
        if item.data_type == "text_array":
            mapping: dict[str, Any] = {
                "type": "text",
            }
        elif item.data_type in {"keyword", "text"}:
            mapping = {
                "type": item.data_type,
            }
        else:
            raise MetaConfigError(
                f"Unsupported Elasticsearch data type "
                f"{item.data_type!r} for field {item.name!r}."
            )

        if item.keyword_subfield and mapping["type"] == "text":
            mapping["fields"] = {
                "keyword": {
                    "type": "keyword",
                },
            }

        return mapping

    def _validate_es_field_value(
        self,
        field_config: ESFieldConfig,
        value: Any,
    ) -> None:
        if field_config.allowed_values and value not in field_config.allowed_values:
            raise MetaConfigError(
                f"Invalid value for Elasticsearch field "
                f"{field_config.name!r}: {value!r}. "
                f"Allowed values: {field_config.allowed_values!r}"
            )

        if field_config.data_type == "text_array":
            if not isinstance(value, list):
                raise MetaConfigError(
                    f"Elasticsearch field {field_config.name!r} "
                    "must be a list."
                )

            invalid_items = [
                item
                for item in value
                if not isinstance(item, str)
            ]
            if invalid_items:
                raise MetaConfigError(
                    f"Elasticsearch field {field_config.name!r} "
                    "must contain only strings."
                )
            return

        if (
            field_config.data_type in {"keyword", "text"}
            and not isinstance(value, str)
        ):
            raise MetaConfigError(
                f"Elasticsearch field {field_config.name!r} "
                "must be a string."
            )

    def _validate_milvus_field_value(
        self,
        field_config: MilvusFieldConfig,
        value: Any,
    ) -> None:
        if (
            field_config.allowed_values
            and value not in field_config.allowed_values
        ):
            raise MetaConfigError(
                f"Invalid value for Milvus field "
                f"{field_config.name!r}: {value!r}. "
                f"Allowed values: {field_config.allowed_values!r}"
            )

        if field_config.data_type == "VARCHAR":
            if not isinstance(value, str):
                raise MetaConfigError(
                    f"Milvus field {field_config.name!r} "
                    "must be a string."
                )
            if (
                field_config.max_length is not None
                and len(value) > field_config.max_length
            ):
                raise MetaConfigError(
                    f"Milvus field {field_config.name!r} exceeds "
                    f"max_length={field_config.max_length}."
                )
            return

        if field_config.data_type == "ARRAY_VARCHAR":
            if not isinstance(value, list) or any(
                not isinstance(item, str)
                for item in value
            ):
                raise MetaConfigError(
                    f"Milvus field {field_config.name!r} "
                    "must be a list of strings."
                )
            if (
                field_config.max_capacity is not None
                and len(value) > field_config.max_capacity
            ):
                raise MetaConfigError(
                    f"Milvus field {field_config.name!r} exceeds "
                    f"max_capacity={field_config.max_capacity}."
                )
            if (
                field_config.max_length is not None
                and any(
                    len(item) > field_config.max_length
                    for item in value
                )
            ):
                raise MetaConfigError(
                    f"Milvus field {field_config.name!r} contains "
                    f"a value exceeding max_length="
                    f"{field_config.max_length}."
                )
            return

        if field_config.data_type == "FLOAT_VECTOR":
            if not isinstance(value, list) or any(
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                for item in value
            ):
                raise MetaConfigError(
                    f"Milvus field {field_config.name!r} "
                    "must be a list of numbers."
                )
            if (
                field_config.dimension is not None
                and len(value) != field_config.dimension
            ):
                raise MetaConfigError(
                    f"Milvus field {field_config.name!r} requires "
                    f"dimension={field_config.dimension}, "
                    f"got {len(value)}."
                )
            return

        raise MetaConfigError(
            f"Unsupported Milvus data type "
            f"{field_config.data_type!r} for field "
            f"{field_config.name!r}."
        )


config_file = (
    Path(__file__).parents[2]
    / "configs"
    / "meta_config.yml"
)

meta_config: MetaConfig = load_config(
    MetaConfig,
    config_file,
)

meta_config.validate()


if __name__ == "__main__":
    print("current path:", __file__)
    print("config path:", config_file)
    print("version:", meta_config.version)

    print(
        "all ES fields:",
        list(meta_config.es_field_map()),
    )

    print(
        "all Milvus fields:",
        list(meta_config.milvus_field_map()),
    )

    print(
        "ES searchable fields:",
        meta_config.es_search_fields(),
    )

    print(
        "Milvus vector fields:",
        [
            item.name
            for item in meta_config.milvus_vector_fields()
        ],
    )

    print(
        "Milvus scalar fields:",
        [
            item.name
            for item in meta_config.milvus_scalar_fields()
        ],
    )

    print(
        "dense vector config:",
        meta_config.get_milvus_field("dense_vector"),
    )

    print(meta_config.es_meta)

    """
        version: 1.0
        all ES fields: ['chunk_id',
         'parent_id', 'law_id', 
         'law_title', 
         'chapter_number',
          'chapter_title', 
          'article_number', 
          'article_title', 
          'paragraph_start', 
          'paragraph_end', 
          'record_type', 
          'content_type', 
          'key_terms', 
          'language', 'status', 'source_file', 'text']
        all Milvus fields: ['chunk_id', 'parent_id', 'law_id', 'chapter_number', 'article_number', 'record_type', 'content_type', 'key_terms', 'language', 'status', 'dense_vector']
        ES searchable fields: ['law_id', 'law_title^2', 'chapter_title^2', 'article_number^4', 'article_title^4', 'paragraph_start', 'paragraph_end', 'key_terms^2.5', 'text']
        Milvus vector fields: ['dense_vector']
        Milvus scalar fields: ['chunk_id', 'parent_id', 'law_id', 'chapter_number', 'article_number', 'record_type', 'content_type', 'key_terms', 'language', 'status']
        dense vector config: MilvusFieldConfig(name='dense_vector', description='Dense embedding generated from the child text and structural headings', data_type='FLOAT_VECTOR', required=True, default=None, allowed_values=[], primary_key=False, filterable=False, max_length=None, max_capacity=None, dimension=1024, metric_type='COSINE', index_type='HNSW', index_params={'M': 16, 'efConstruction': 200})
    
    """
