from __future__ import annotations

from typing import Protocol


class Expander(Protocol):
    """Protocol for an expander model."""

    def expand(self, document: str) -> list[str]:
        """Generate a query from the given document."""
        raise NotImplementedError

    @classmethod
    def from_default(cls: type[Expander]) -> Expander:
        """Load a default expander model."""
        raise NotImplementedError
