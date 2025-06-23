from enum import Enum


class ExpansionType(str, Enum):
    """Enum for different types of expansions."""

    T5DOC2QUERY = "t5doc2query"
    TILDE = "tilde"
    SPLADE = "splade"
