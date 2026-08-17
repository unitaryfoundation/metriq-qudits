from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import validate
from pydantic import BaseModel, Field, create_model

from metriq_qudits.constants import SCHEMA_MAPPING, JobType

SCHEMA_DIR = Path(__file__).parent / "schemas"
BENCHMARK_NAME_KEY = "benchmark_name"

_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def load_json_file(path) -> dict:
    with open(path) as file:
        return json.load(file)


def load_schema(benchmark_name: str) -> dict:
    """Return the JSON schema for a benchmark name."""
    filename = SCHEMA_MAPPING.get(JobType(benchmark_name))
    if filename is None:
        raise ValueError(f"Unsupported benchmark: {benchmark_name}")
    return load_json_file(SCHEMA_DIR / filename)


def create_pydantic_model(schema: dict[str, Any]) -> type[BaseModel]:
    """Build a pydantic model from a JSON schema, carrying defaults and bounds."""
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for name, spec in schema["properties"].items():
        field_type = _TYPE_MAPPING[spec["type"]]
        params: dict[str, Any] = {}
        if "default" in spec:
            params["default"] = spec["default"]
        elif name not in required:
            params["default"] = None
        else:
            params["default"] = ...
        if "minimum" in spec:
            params["ge"] = spec["minimum"]
        if "maximum" in spec:
            params["le"] = spec["maximum"]
        optional = name not in required and "default" not in spec
        final_type = field_type | None if optional else field_type
        fields[name] = (final_type, Field(**params))
    model = create_model(schema["title"], **fields)
    model.model_rebuild()
    return model


def validate_and_create_model(config: dict[str, Any]) -> BaseModel:
    """Validate a config dict against its schema and build the params model."""
    if config.get(BENCHMARK_NAME_KEY) is None:
        raise ValueError(f"Missing '{BENCHMARK_NAME_KEY}' key in config.")
    schema = load_schema(config[BENCHMARK_NAME_KEY])
    validate(config, schema)
    return create_pydantic_model(schema)(**config)


def load_and_validate(path) -> BaseModel:
    """Load a benchmark config file and return its validated params model."""
    return validate_and_create_model(load_json_file(path))
