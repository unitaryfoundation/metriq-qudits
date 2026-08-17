"""Canonical benchmark identifiers and their schema files."""

from enum import StrEnum


class JobType(StrEnum):
    QUANTUM_VOLUME = "quantum_volume"


SCHEMA_MAPPING = {
    JobType.QUANTUM_VOLUME: "quantum_volume.schema.json",
}
