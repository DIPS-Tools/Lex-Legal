from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from app.config.meta_config import ESFieldConfig, meta_config


def _python_type(field_config: ESFieldConfig) -> type[Any]:
    """Map one Elasticsearch YAML data type to its Python value type."""
    if field_config.data_type in {"keyword", "text"}:
        return str
    if field_config.data_type == "text_array":
        return list[str]
    raise ValueError(
        f"Unsupported Elasticsearch entity type: "
        f"{field_config.data_type!r}"
    )


def _field_definition(
    field_config: ESFieldConfig,
) -> tuple[Any, Any]:
    """Convert one YAML field definition into a Pydantic model field."""
    python_type = _python_type(field_config)

    if field_config.default is not None:
        default_value = deepcopy(field_config.default)
        if isinstance(default_value, (list, dict, set)):
            return (
                python_type,
                Field(
                    default_factory=lambda value=default_value: deepcopy(value)
                ),
            )
        return python_type, default_value

    if field_config.required:
        return python_type, ...

    return python_type | None, None


class _ESRecordInfoBase(BaseModel):
    """Behavior shared by the YAML-generated Elasticsearch record model."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    @classmethod
    def from_chunk(cls, chunk: dict[str, Any]) -> Self:
        """Create a storage record from a pipeline chunk."""
        validated = meta_config.validate_es_record(
            chunk,
            reject_unknown=False,
        )
        return cls.model_validate(validated)

    @model_validator(mode="after")
    def validate_against_yaml(self) -> Self:
        """Enforce YAML defaults, allowed values, and runtime field types."""
        meta_config.validate_es_record(
            self.model_dump(exclude_none=True),
            reject_unknown=True,
        )
        return self

    def to_document(self) -> dict[str, Any]:
        """Return the exact dictionary written to Elasticsearch."""
        return self.model_dump(exclude_none=True)


ESRecordInfo = create_model(
    "ESRecordInfo",
    __base__=_ESRecordInfoBase,
    __module__=__name__,
    **{
        field_config.name: _field_definition(field_config)
        for field_config in meta_config.es_meta
    },
)


@dataclass(frozen=True)
class ESSearchResult:
    """One Elasticsearch record together with its BM25 relevance score."""

    record: _ESRecordInfoBase
    score: float | None


def print_es_record_info_fields() -> None:
    """Print all YAML-generated ESRecordInfo fields and their metadata."""
    print("ESRecordInfo fields:")
    for field_name, model_field in ESRecordInfo.model_fields.items():
        field_config = meta_config.get_es_field(field_name)
        default = (
            "<required>"
            if model_field.is_required()
            else model_field.get_default(call_default_factory=True)
        )
        print(
            f"- {field_name}: "
            f"type={model_field.annotation}, "
            f"required={model_field.is_required()}, "
            f"default={default!r}, "
            f"es_type={field_config.data_type}"
        )


if __name__ == "__main__":
    print_es_record_info_fields()
