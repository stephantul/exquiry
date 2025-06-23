from __future__ import annotations

import re
from typing import TypeVar, cast

import torch
from transformers import BertLMHeadModel, BertTokenizerFast

from exquiry.types import ExpansionType

_NAME_TO_TOKENIZER = {"ielab/TILDE": "bert-base-uncased"}
_DEFAULT_MODEL = "ielab/TILDE"

# These are stopwords used in NLTK.
_STOPWORDS = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "you're",
    "you've",
    "you'll",
    "you'd",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "she's",
    "her",
    "hers",
    "herself",
    "it",
    "it's",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "that'll",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "don't",
    "should",
    "should've",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "aren't",
    "couldn",
    "couldn't",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "isn",
    "isn't",
    "ma",
    "mightn",
    "mightn't",
    "mustn",
    "mustn't",
    "needn",
    "needn't",
    "shan",
    "shan't",
    "shouldn",
    "shouldn't",
    "wasn",
    "wasn't",
    "weren",
    "weren't",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
}


def _find_stopword_ids(stopwords: set[str], tokenizer: BertTokenizerFast) -> set[int]:
    """Find stopword IDs in the tokenizer."""
    vocab = tokenizer.get_vocab()

    stop_ids = set()
    for stopword in stopwords:
        ids = tokenizer.encode(stopword, add_special_tokens=False)
        if len(ids) == 1:
            stop_ids.add(ids[0])

    for token, token_id in vocab.items():
        if token.startswith("##") and len(token) > 1:  # skip most of subtokens
            stop_ids.add(token_id)
        if not re.match("^[A-Za-z_-]*$", token):  # remove numbers, symbols, etc..
            stop_ids.add(token_id)
        if not re.match("[A-Za-z]", token):
            stop_ids.add(token_id)

    return stop_ids


class Tilde:
    expansion_type = ExpansionType.TILDE

    def __init__(self, model: BertLMHeadModel, tokenizer: BertTokenizerFast, k: int = 100) -> None:
        """Initialize the T5Doc2Query model."""
        self.tokenizer = tokenizer
        self.model = model
        self.stop_ids = _find_stopword_ids(_STOPWORDS, tokenizer)
        self.valid_ids = sorted(set(range(tokenizer.vocab_size)) - self.stop_ids)
        self.k = k

    @classmethod
    def from_pretrained(cls: type[T], model_name: str, device: str = "cpu") -> T:
        """Load a Tilde model from a pretrained model name."""
        tokenizer = BertTokenizerFast.from_pretrained(_NAME_TO_TOKENIZER.get(model_name, model_name))
        model = BertLMHeadModel.from_pretrained(model_name)
        model = model.to(device)  # type: ignore  # invalid typing in transformers
        return cls(model, tokenizer)

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return self.model.device

    @classmethod
    def from_default(cls: type[T]) -> T:
        """Load a default Tilde model."""
        return cls.from_pretrained(_DEFAULT_MODEL, device="cpu")

    @torch.no_grad()
    def expand(self, document: str) -> list[str]:
        """Generate a query from the given document."""
        input_ids = cast(torch.Tensor, self.tokenizer.encode(document, return_tensors="pt"))
        input_ids[:, 0] = 1  # type: ignore  # Set the first token to passage input
        all_valid_ids: list[int] = sorted(set(self.valid_ids) - set(input_ids[0].tolist()))
        with torch.no_grad():
            # Take logits of the CLS token.
            logits = self.model(input_ids).logits[0, 0][all_valid_ids]
            s = [all_valid_ids[x] for x in torch.topk(logits, k=self.k).indices]

        return [self.tokenizer.decode(x) for x in s]


T = TypeVar("T", bound=Tilde)
