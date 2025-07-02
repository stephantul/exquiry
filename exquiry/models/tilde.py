from __future__ import annotations

import re
from typing import Collection, TypeVar, cast

import torch
from tqdm import tqdm
from transformers import BertLMHeadModel, BertTokenizerFast

from exquiry.models.base import Expander
from exquiry.types import ExpansionType

_NAME_TO_TOKENIZER = {"ielab/TILDE": "bert-base-uncased"}
_DEFAULT_MODEL = "ielab/TILDE"

# These are stopwords used in NLTK.
_STOPWORDS = frozenset(
    {
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
)


def _find_stopword_ids(stopwords: Collection[str], tokenizer: BertTokenizerFast) -> set[int]:
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


class Tilde(Expander):
    expansion_type = ExpansionType.TILDE

    def __init__(
        self, model: BertLMHeadModel, tokenizer: BertTokenizerFast, stopwords: Collection[str] | None = _STOPWORDS
    ) -> None:
        """Initialize the T5Doc2Query model."""
        self.tokenizer = tokenizer
        self.model = model
        self.stop_ids: set[int]
        if stopwords is None:
            self.stop_ids = set()
        else:
            self.stop_ids = _find_stopword_ids(stopwords, tokenizer)
        # A priori valid ids are all IDs in the vocabulary minus the stopwords.
        self.valid_ids = sorted(set(range(tokenizer.vocab_size)) - self.stop_ids)

    @classmethod
    def from_pretrained(cls: type[T], model_name: str, device: str = "mps") -> T:
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
        return cls.from_pretrained(_DEFAULT_MODEL, device="mps")

    @torch.no_grad()
    def _expand(self, documents: list[str], k: int, show_progressbar: bool = True) -> list[list[str]]:
        """Generate a query from the given document."""
        out = []
        for batch_idx in tqdm(range(0, len(documents), 32), disable=not show_progressbar):
            docs = documents[batch_idx : batch_idx + 32]
            batch = self.tokenizer.batch_encode_plus(docs, return_tensors="pt", padding=True, truncation=True)
            batch = batch.to(self.device)  # type: ignore  # invalid typing in transformers
            batch.input_ids[:, 0] = 1  # type: ignore  # Set the first token to passage input
            with torch.no_grad():
                # Take logits of the CLS token.
                logits = self.model(**batch).logits[:, 0]
            # The top k tokens are the most probable next tokens
            # We take twice the number of tokens to ensure we have enough valid tokens
            for row, input_ids in zip(logits, batch.input_ids):
                # Remove stopwords and subtokens from the logits.
                stop_ids = list(self.stop_ids | set(cast(set[int], input_ids.tolist())))
                row[stop_ids] = float("-inf")
            top = torch.topk(logits, k=k, dim=1).indices
            for s, i in zip(top.tolist(), batch.input_ids):
                # Filter out stopwords and subtokens.
                i = set(i.tolist())
                s = [x for x in s if x in self.valid_ids and x not in i]
                out.append([self.tokenizer.decode(x) for x in s])

        return out


T = TypeVar("T", bound=Tilde)
