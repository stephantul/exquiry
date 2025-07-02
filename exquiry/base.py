from __future__ import annotations

from typing import Literal, overload

from exquiry.models.base import Expander
from exquiry.models.doc2query import T5Doc2Query
from exquiry.models.tilde import Tilde
from exquiry.types import ExpansionType

_EXPANDER_MAPPING: dict[ExpansionType, type[Expander]] = {
    ExpansionType.T5DOC2QUERY: T5Doc2Query,
    ExpansionType.TILDE: Tilde,
}


@overload
def get_expander_class(expansion_type: Literal[ExpansionType.T5DOC2QUERY, "t5doc2query"]) -> type[T5Doc2Query]: ...
@overload
def get_expander_class(expansion_type: Literal[ExpansionType.TILDE, "tilde"]) -> type[Tilde]: ...
@overload
def get_expander_class(expansion_type: ExpansionType | str) -> type[Expander]: ...


def get_expander_class(expansion_type: ExpansionType | str) -> type[Expander]:
    """
    Get the expander class for the given expansion type.

    You can then instantiate the expander, either using `from_default`,
    `from_pretrained`, or by initializing it directly with the required parameters.

    :param expansion_type: The type of expansion to get the class for.
        This can be an `ExpansionType` enum or a string representation of the type.
        The following inputs are accepted:
        - "t5doc2query" or `ExpansionType.T5DOC2QUERY`
        - "tilde" or `ExpansionType.TILDE`
    :return: The expander class corresponding to the expansion type.
    """
    expansion_type = ExpansionType(expansion_type)
    return _EXPANDER_MAPPING[expansion_type]


@overload
def get_expander(expansion_type: Literal[ExpansionType.T5DOC2QUERY, "t5doc2query"]) -> T5Doc2Query: ...
@overload
def get_expander(expansion_type: Literal[ExpansionType.TILDE, "tilde"]) -> Tilde: ...
@overload
def get_expander(expansion_type: ExpansionType | str) -> Expander: ...


def get_expander(expansion_type: ExpansionType | str) -> Expander:
    """
    Get a default expander for the given expansion type.

    This gets a good default expander for the given expansion type.
    For T5Doc2Query, it uses the `castorini/doc2query-t5-base-msmarco` model.
    For Tilde, it uses the `ielab/tilde` model.

    :param expansion_type: The type of expansion to get the class for.
        This can be an `ExpansionType` enum or a string representation of the type.
        The following inputs are accepted:
        - "t5doc2query" or `ExpansionType.T5DOC2QUERY`
        - "tilde" or `ExpansionType.TILDE`
    :return: The expander class corresponding to the expansion type.
    """
    expander_class = get_expander_class(expansion_type)
    return expander_class.from_default()
