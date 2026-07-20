from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemConfig:
    """Single-qudit logical system.

    Physical parameters belong beside the numerical stage that uses them.
    this object only identifies the system whose results are being produced.
    """

    d: int

    def __post_init__(self) -> None:
        if self.d < 2:
            raise ValueError("d must be at least 2")

    @property
    def dimension(self) -> int:
        return self.d

    @property
    def key(self) -> str:
        return f"d{self.d}"
