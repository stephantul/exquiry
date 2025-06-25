from __future__ import annotations

from typing import TypeVar, overload

from exquiry.types import ExpansionType


class Expander:
    """Protocol for an expander model."""

    expansion_type: ExpansionType

    @overload
    def expand(self, documents: list[str]) -> list[list[str]]: ...
    @overload
    def expand(self, documents: str) -> list[str]: ...

    def expand(self, documents: list[str] | str) -> list[list[str]] | list[str]:
        """Generate a query from the given document."""
        single_doc = False
        if isinstance(documents, str):
            single_doc = True
            documents = [documents]

        result = self._expand(documents)

        if single_doc:
            return result[0]
        return result

    @classmethod
    def from_default(cls: type[T]) -> T:
        """Load a default expander model."""
        raise NotImplementedError

    def _expand(self, documents: list[str]) -> list[list[str]]:
        """Internal method to expand documents."""
        raise NotImplementedError("This method should be implemented by subclasses.")


T = TypeVar("T", bound=Expander)
