from __future__ import annotations

from typing import TypeVar

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from exquiry.models.base import Expander
from exquiry.types import ExpansionType

_DEFAULT_MODEL = "castorini/doc2query-t5-base-msmarco"


class T5Doc2Query(Expander):
    expansion_type = ExpansionType.T5DOC2QUERY

    def __init__(
        self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer, max_length: int = 64, top_k: int = 10
    ) -> None:
        """Initialize the T5Doc2Query model."""
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.top_k = top_k

    @classmethod
    def from_pretrained(cls: type[T], model_name: str, top_k: int = 10, max_length: int = 64, device: str = "cpu") -> T:
        """Load a T5Doc2Query model from a pretrained model name."""
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model = model.to(device)  # type: ignore  # invalid typing in transformers
        return cls(model, tokenizer, max_length, top_k)

    @classmethod
    def from_default(cls: type[T]) -> T:
        """Load a default T5Doc2Query model."""
        return cls.from_pretrained(_DEFAULT_MODEL, device="cpu", top_k=10, max_length=64)

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return self.model.device

    @torch.no_grad()
    def _expand(self, documents: list[str], k: int, show_progressbar: bool) -> list[list[str]]:
        """Generate a query from the given document."""
        out = []
        for document in tqdm(documents, disable=not show_progressbar):
            input_ids = self.tokenizer.encode(document, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=self.max_length,
                do_sample=True,
                top_k=self.top_k,
                num_return_sequences=k,
            )

            out_docs = []
            for output in outputs:
                out_docs.append(self.tokenizer.decode(output, skip_special_tokens=True))

            out.append(out_docs)

        return out


T = TypeVar("T", bound=T5Doc2Query)
