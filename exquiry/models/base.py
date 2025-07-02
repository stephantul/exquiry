from __future__ import annotations

from typing import TypeVar, overload

from exquiry.types import ExpansionType


class Expander:
    """Protocol for an expander model."""

    expansion_type: ExpansionType

    @overload
    def expand(self, documents: list[str], k: int, show_progressbar: bool) -> list[list[str]]: ...
    @overload
    def expand(self, documents: str, k: int, show_progressbar: bool) -> list[str]: ...

    def expand(self, documents: list[str] | str, k: int, show_progressbar: bool = True) -> list[list[str]] | list[str]:
        """Generate a query from the given document."""
        single_doc = False
        if isinstance(documents, str):
            single_doc = True
            documents = [documents]

        result = self._expand(documents, k, show_progressbar)

        if single_doc:
            return result[0]
        return result

    @classmethod
    def from_default(cls: type[T]) -> T:
        """Load a default expander model."""
        raise NotImplementedError

    def _expand(self, documents: list[str], k: int, show_progressbar: bool) -> list[list[str]]:
        """Internal method to expand documents."""
        raise NotImplementedError("This method should be implemented by subclasses.")


T = TypeVar("T", bound=Expander)
