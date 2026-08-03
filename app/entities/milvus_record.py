from __future__ import annotations

from copy import deepcopy
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from app.config.meta_config import MilvusFieldConfig, meta_config


def _python_type(field_config: MilvusFieldConfig) -> type[Any]:
    """Map one Milvus YAML data type to its Python value type."""
    if field_config.data_type == "VARCHAR":
        return str
    if field_config.data_type == "ARRAY_VARCHAR":
        return list[str]
    if field_config.data_type == "FLOAT_VECTOR":
        return list[float]
    raise ValueError(
        f"Unsupported Milvus entity type: "
        f"{field_config.data_type!r}"
    )


def _field_definition(
    field_config: MilvusFieldConfig,
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


class _MilvusRecordInfoBase(BaseModel):
    """Behavior shared by the YAML-generated Milvus record model."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    @classmethod
    def from_embedding_record(
        cls,
        record: dict[str, Any],
    ) -> Self:
        """Create a Milvus entity from an embedding pipeline record."""
        validated = meta_config.validate_milvus_record(
            record,
            reject_unknown=False,
        )
        return cls.model_validate(validated)

    @model_validator(mode="after")
    def validate_against_yaml(self) -> Self:
        """Enforce YAML defaults, constraints, and runtime field types."""
        meta_config.validate_milvus_record(
            self.model_dump(exclude_none=True),
            reject_unknown=True,
        )
        return self

    def to_entity(self) -> dict[str, Any]:
        """Return the exact dictionary written to Milvus."""
        return self.model_dump(exclude_none=True)


MilvusRecordInfo = create_model(
    "MilvusRecordInfo",
    __base__=_MilvusRecordInfoBase,
    __module__=__name__,
    **{
        field_config.name: _field_definition(field_config)
        for field_config in meta_config.milvus_meta
    },
)



def print_milvus_record_info_fields() -> None:
    """Print all YAML-generated MilvusRecordInfo fields and their metadata."""
    print("MilvusRecordInfo fields:")
    for field_name, model_field in MilvusRecordInfo.model_fields.items():
        field_config = meta_config.get_milvus_field(field_name)
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
            f"milvus_type={field_config.data_type}"
        )


if __name__ == "__main__":
    print_milvus_record_info_fields()

