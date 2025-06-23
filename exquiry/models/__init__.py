from exquiry.models.base import Expander
from exquiry.models.doc2query import T5Doc2Query
from exquiry.models.tilde import Tilde
from exquiry.types import ExpansionType

_EXPANDER_MAPPING: dict[ExpansionType, type[Expander]] = {
    ExpansionType.T5DOC2QUERY: T5Doc2Query,
    ExpansionType.TILDE: Tilde,
}


def get_expander(expansion_type: ExpansionType) -> type[Expander]:
    """Get the expander model for the given expansion type."""
    return _EXPANDER_MAPPING[expansion_type]
