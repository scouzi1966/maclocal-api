"""Data model for a single report record."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Row:
    """One record in a report.

    ``fields`` maps a column name to its value. The column order across a
    report is taken from the first row's insertion order.
    """

    fields: dict[str, Any]

    def get(self, column: str) -> Any:
        return self.fields.get(column)
