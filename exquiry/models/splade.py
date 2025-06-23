from __future__ import annotations

from typing import Sequence, TypeVar, cast

import torch
from sentence_transformers import SparseEncoder

from exquiry.types import ExpansionType

_DEFAULT_MODEL = "naver/splade-v3"
_DEFAULT_REF = "refs/pr/6"


class SPLADE:
    expansion_type = ExpansionType.SPLADE

    def __init__(self, model: SparseEncoder, k: int | None = None, threshold: float | None = None) -> None:
        """Initialize the SPLADE model."""
        self.model = model
        self.k = k
        self.threshold = threshold

    @classmethod
    def from_pretrained(
        cls: type[T],
        model_name: str,
        revision: str | None = None,
        device: str = "cpu",
        k: int | None = None,
        threshold: float | None = None,
    ) -> T:
        """Load a SPLADE model from a pretrained model name."""
        model = SparseEncoder(model_name, revision=revision, device=device)
        return cls(model, k, threshold)

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return self.model.device

    @classmethod
    def from_default(cls: type[T]) -> T:
        """Load a default Tilde model."""
        return cls.from_pretrained(_DEFAULT_MODEL, revision=_DEFAULT_REF, device="cpu")

    @torch.no_grad()
    def expand(self, document: str) -> list[str]:
        """Generate a query from the given document."""
        input_ids = set(self.model.tokenizer.encode(document))
        with torch.no_grad():
            result = cast(torch.Tensor, self.model.encode(document))
            # Result is a sparse tensor. To get it, we need to coalesce
        result = result.coalesce()
        indices, values = result.indices()[0].tolist(), result.values().tolist()
        out = []
        for x, y in zip(indices, values):
            if x in input_ids:
                continue
            out.append((x, y))
        s, _ = zip(*sorted(out, key=lambda x: x[1], reverse=True)[: self.k])

        return [self.model.tokenizer.decode(x) for x in s]


T = TypeVar("T", bound=SPLADE)
