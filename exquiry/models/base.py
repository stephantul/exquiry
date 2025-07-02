from __future__ import annotations

from typing import TypeVar, overload

from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from exquiry.types import ExpansionType


class Expander:
    """Protocol for an expander model."""

    expansion_type: ExpansionType

    @overload
    def expand(self, documents: list[str], k: int, show_progressbar: bool) -> list[list[str]]: ...
    @overload
    def expand(self, documents: str, k: int, show_progressbar: bool) -> list[str]: ...

    def expand(self, documents: list[str] | str, k: int, show_progressbar: bool = True) -> list[list[str]] | list[str]:
        """
        Expands one or more documents into a list of generated queries.

        Given a document or a list of documents, this method generates `k` queries for each document using the underlying expansion logic.
        If a single document is provided, returns a list of queries for that document.
        If a list of documents is provided, returns a list of lists, where each sublist contains the queries for the corresponding document.

        :param documents: A single document as a string, or a list of documents to expand.
        :param k: The number of queries to generate per document.
        :param show_progressbar: Whether to display a progress bar during expansion. Defaults to True.

        :return: A list of generated queries for each document, or a list of queries if a single document was provided.
        """
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

    def expand_as_tokens(
        self, tokenizer: PreTrainedTokenizerBase, documents: list[str] | str, k: int, show_progressbar: bool = True
    ) -> BatchEncoding:
        """
        Generate tokenized inputs using the expander.

        The expansions are separated from the original documents by using a [SEP] token.

        :param tokenizer: The tokenizer to use for encoding the expansions.
            This should be the tokenizer you use for your model, not the tokenizer used in Tilde.
        :param documents: The documents to expand, either as a list of strings or a single string.
        :param k: The number of expansions to generate for each document.
        :param show_progressbar: Whether to show a progress bar during expansion.
        :return: A BatchEncoding object containing the tokenized expansions.
        """
        expansions: list[list[str]] | list[str]
        # This looks weird but makes the typing work correctly.
        if isinstance(documents, str):
            expansions = self.expand(documents, k, show_progressbar)
            pairs = [(documents, " ".join(expansions))]
        else:
            expansions = self.expand(documents, k, show_progressbar)
            pairs = [(doc, " ".join(exp)) for doc, exp in zip(documents, expansions, strict=True)]

        return tokenizer.batch_encode_plus(pairs, return_tensors="pt", padding=True, truncation=True)


T = TypeVar("T", bound=Expander)
