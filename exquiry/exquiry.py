from __future__ import annotations

from typing import Any, TypeVar

from exquiry.models import get_expander
from exquiry.models.base import Expander
from exquiry.types import ExpansionType


class DocumentExpander:
    def __init__(self, expanders: dict[str, Expander]) -> None:
        """Initialize the DocumentExpander with a model."""
        self.expanders = expanders

    def expand(self, document: str) -> dict[str, list[str]]:
        """Generate a query from the given document using the specified expansion type."""
        expansions = {}
        for name, expander in self.expanders.items():
            expansions[name] = expander.expand(document)

        return expansions

    @classmethod
    def from_expansion_types(cls: type[T], expansion_types: list[ExpansionType]) -> T:
        """Create a DocumentExpander from a list of expansion types."""
        models = {}
        expansion_types = [ExpansionType(t) for t in expansion_types]
        for expansion_type in expansion_types:
            model_class = get_expander(expansion_type)
            models[expansion_type.value] = model_class.from_default()

        return cls(models)


T = TypeVar("T", bound=DocumentExpander)
