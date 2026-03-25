from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

from ..models import Binding, Example, Frame, LexicalUnit, LocalID, Mapping, MetaphorEntry

if TYPE_CHECKING:
    from rdflib import Graph
    from ..parsers.rdf_extractors import _Extractor


class MetaNetRepository:
    def __init__(self, graph: "Graph", extractor_registry: "dict[str, type[_Extractor[Any]]]") -> None:
        self._graph = graph
        self._extractor_registry = extractor_registry

    @cached_property
    def metaphors(self) -> dict[LocalID, MetaphorEntry]:
        return self._extractor_registry["metaphors"]().extract(self._graph)

    @cached_property
    def frames(self) -> dict[LocalID, Frame]:
        return self._extractor_registry["frames"]().extract(self._graph)

    @cached_property
    def mappings(self) -> dict[LocalID, Mapping]:
        return self._extractor_registry["mappings"]().extract(self._graph)

    @cached_property
    def bindings(self) -> dict[LocalID, Binding]:
        return self._extractor_registry["bindings"]().extract(self._graph)

    @cached_property
    def examples(self) -> dict[LocalID, Example]:
        return self._extractor_registry["examples"]().extract(self._graph)

    @cached_property
    def lexical_units(self) -> dict[LocalID, LexicalUnit]:
        return self._extractor_registry["lexical_units"]().extract(self._graph)
