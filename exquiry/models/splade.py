from __future__ import annotations

from typing import TypeVar, cast

import torch
from sentence_transformers import SparseEncoder

from exquiry.models.base import Expander
from exquiry.types import ExpansionType

_DEFAULT_MODEL = "naver/splade-v3"
_DEFAULT_REF = "refs/pr/6"


class SPLADE(Expander):
    expansion_type = ExpansionType.SPLADE

    def __init__(self, model: SparseEncoder, k: int | None = None) -> None:
        """Initialize the SPLADE model."""
        self.model = model
        self.k = k

    @classmethod
    def from_pretrained(
        cls: type[T],
        model_name: str,
        revision: str | None = None,
        device: str = "cpu",
        k: int | None = None,
    ) -> T:
        """Load a SPLADE model from a pretrained model name."""
        model = SparseEncoder(model_name, revision=revision, device=device)
        return cls(model, k)

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return self.model.device

    @classmethod
    def from_default(cls: type[T]) -> T:
        """Load a default Tilde model."""
        return cls.from_pretrained(_DEFAULT_MODEL, revision=_DEFAULT_REF, device="cpu")

    @torch.no_grad()
    def _expand(self, documents: list[str]) -> list[list[str]]:
        """Generate a query from the given document."""
        with torch.no_grad():
            result = cast(torch.Tensor, self.model.encode(documents))
            # Result is a sparse tensor. To get it, we need to coalesce

        out: list[list[str]] = []
        for x in result:
            indices = x.coalesce().indices().squeeze(0)
            # Convert to tokens
            tokens = [self.model.tokenizer.decode([i]) for i in indices.tolist()]
            out.append(tokens)
        return out


T = TypeVar("T", bound=SPLADE)
