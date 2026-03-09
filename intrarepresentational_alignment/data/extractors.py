from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Generic, TypeVar

from rdflib import Graph, Literal, URIRef

from .metanet import (
    LocalID,
    MetaphorEntry, Frame, Mapping, Binding, Example, LexicalUnit,
)
from .namespaces import (
    _type, _label,
    _Metaphor, _Frame, _Mapping, _Binding, _Example, _LexicalUnit,
    _hasName, _hasDescription, _hasStatus, _hasCulturalScope, _wasInvestigatedFor,
    _hasSourceFrame, _hasTargetFrame, _hasMappings, _hasEntailment, _hasExample,
    _isEntailedByMetaphor, _isTargetSubcaseOfMetaphor, _isSourceSubcaseOfMetaphor,
    _isInMetaphorFamily, _hasAlias, _hasMetaphorType, _hasMetaphorLevel,
    _hasFrenchCorrespondent, _hasSpanishCorrespondent,
    _hasFrameType, _hasRoles, _hasLexicalUnit, _hasInference, _hasBindings,
    _isInFrameFamily, _correspondsToFrameNet, _makesUseOfFrame,
    _hasSourceRole, _hasTargetRole,
    _hasBoundRole1, _hasBoundRole2,
    _hasSentence, _hasAnnotation, _exampleConstruct, _exampleDialect,
    _isFromLanguage, _hasProvenance,
    _hasLemma, _LUs_Lemmas, _LUs_Language,
)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# RDF query helpers
# ---------------------------------------------------------------------------

def _local(uri: URIRef) -> LocalID:
    """Return the local fragment of a URI (everything after # or last /)."""
    s = str(uri)
    if "#" in s:
        return LocalID(s.split("#")[-1])
    return LocalID(s.split("/")[-1])


def _str(g: Graph, s: URIRef, p: URIRef) -> str | None:
    """Return the first literal value for a single-valued property, or None."""
    for o in g.objects(s, p):
        if isinstance(o, Literal):
            return str(o)
    return None


def _strs(g: Graph, s: URIRef, p: URIRef) -> list[str]:
    """Return all literal values for a multi-valued property."""
    return [str(o) for o in g.objects(s, p) if isinstance(o, Literal)]


def _refs(g: Graph, s: URIRef, p: URIRef) -> list[LocalID]:
    """Return local IDs of all URI-reference objects for a property."""
    return [_local(o) for o in g.objects(s, p) if isinstance(o, URIRef)]


def _ref(g: Graph, s: URIRef, p: URIRef) -> LocalID | None:
    """Return the local ID of the first URI-reference object, or None."""
    refs = _refs(g, s, p)
    return refs[0] if refs else None


# ---------------------------------------------------------------------------
# Abstract base extractor
# ---------------------------------------------------------------------------

class _Extractor(ABC, Generic[T]):
    """
    Base class for RDF entity extractors.

    Subclasses declare a `rdf_type` class attribute identifying the RDF type
    to query, and implement `_build()` to construct one domain entity from a
    single RDF subject. The `extract()` method handles the iteration and
    dict-building scaffold that would otherwise be repeated for every entity type.
    """

    rdf_type: URIRef

    def extract(self, g: Graph) -> dict[LocalID, T]:
        """Return a dict mapping local ID to domain entity for every subject of this type."""
        result: dict[LocalID, T] = {}
        for s in g.subjects(_type, self.rdf_type):
            eid = _local(s)
            result[eid] = self._build(g, s, eid)
        return result

    @abstractmethod
    def _build(self, g: Graph, s: URIRef, eid: LocalID) -> T:
        """Construct a domain entity from a single RDF subject."""
        ...


# ---------------------------------------------------------------------------
# Concrete extractors
# ---------------------------------------------------------------------------

class _MetaphorExtractor(_Extractor[MetaphorEntry]):
    rdf_type = _Metaphor

    def _build(self, g: Graph, s: URIRef, eid: LocalID) -> MetaphorEntry:
        return MetaphorEntry(
            id                    = eid,
            name                  = _str(g, s, _hasName),
            source_frame          = _ref(g, s, _hasSourceFrame),
            target_frame          = _ref(g, s, _hasTargetFrame),
            mappings              = _refs(g, s, _hasMappings),
            entailments           = _refs(g, s, _hasEntailment),
            examples              = _refs(g, s, _hasExample),
            is_entailed_by        = _refs(g, s, _isEntailedByMetaphor),
            is_target_subcase_of  = _refs(g, s, _isTargetSubcaseOfMetaphor),
            is_source_subcase_of  = _refs(g, s, _isSourceSubcaseOfMetaphor),
            metaphor_families     = _refs(g, s, _isInMetaphorFamily),
            alias                 = _str(g, s, _hasAlias),
            cultural_scope        = _strs(g, s, _hasCulturalScope),
            metaphor_type         = _strs(g, s, _hasMetaphorType),
            metaphor_level        = _str(g, s, _hasMetaphorLevel),
            status                = _str(g, s, _hasStatus),
            description           = _str(g, s, _hasDescription),
            french_correspondent  = _str(g, s, _hasFrenchCorrespondent),
            spanish_correspondent = _str(g, s, _hasSpanishCorrespondent),
            was_investigated_for  = _strs(g, s, _wasInvestigatedFor),
        )


class _FrameExtractor(_Extractor[Frame]):
    rdf_type = _Frame

    def _build(self, g: Graph, s: URIRef, eid: LocalID) -> Frame:
        return Frame(
            id                      = eid,
            name                    = _str(g, s, _hasName),
            description             = _str(g, s, _hasDescription),
            frame_type              = _str(g, s, _hasFrameType),
            roles                   = _refs(g, s, _hasRoles),
            lexical_units           = _refs(g, s, _hasLexicalUnit),
            inferences              = _refs(g, s, _hasInference),
            bindings                = _refs(g, s, _hasBindings),
            frame_families          = _refs(g, s, _isInFrameFamily),
            corresponds_to_framenet = _str(g, s, _correspondsToFrameNet),
            makes_use_of_frames     = _refs(g, s, _makesUseOfFrame),
            cultural_scope          = _strs(g, s, _hasCulturalScope),
            status                  = _str(g, s, _hasStatus),
        )


class _MappingExtractor(_Extractor[Mapping]):
    rdf_type = _Mapping

    def _build(self, g: Graph, s: URIRef, eid: LocalID) ->Mapping:
        return Mapping(
            id          = eid,
            source_role = _ref(g, s, _hasSourceRole),
            target_role = _ref(g, s, _hasTargetRole),
            label       = _str(g, s, _label),
        )


class _BindingExtractor(_Extractor[Binding]):
    rdf_type = _Binding

    def _build(self, g: Graph, s: URIRef, eid: LocalID) ->Binding:
        return Binding(
            id           = eid,
            bound_role_1 = _ref(g, s, _hasBoundRole1),
            bound_role_2 = _ref(g, s, _hasBoundRole2),
            label        = _str(g, s, _label),
        )


class _ExampleExtractor(_Extractor[Example]):
    rdf_type = _Example

    def _build(self, g: Graph, s: URIRef, eid: LocalID) ->Example:
        return Example(
            id           = eid,
            sentence     = _str(g, s, _hasSentence),
            annotation   = _str(g, s, _hasAnnotation),
            construction = _str(g, s, _exampleConstruct),
            dialect      = _str(g, s, _exampleDialect),
            language     = _str(g, s, _isFromLanguage),
            provenance   = _str(g, s, _hasProvenance),
        )


class _LexicalUnitExtractor(_Extractor[LexicalUnit]):
    rdf_type = _LexicalUnit

    def _build(self, g: Graph, s: URIRef, eid: LocalID) ->LexicalUnit:
        raw_lemmas = _str(g, s, _LUs_Lemmas)
        return LexicalUnit(
            id       = eid,
            lemma    = _str(g, s, _hasLemma),
            lemmas   = raw_lemmas.split("|") if raw_lemmas else [],
            language = _str(g, s, _LUs_Language),
        )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class MetaNetRepository:
    """
    Lazily-loaded container for all MetaNet entity types.

    Each entity collection is extracted from the RDF graph on first access,
    then cached for subsequent calls. Moving the class here (alongside the
    extractors) avoids a circular import between metanet.py and extractors.py.
    """

    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        self._metaphors:     dict[LocalID, MetaphorEntry] | None = None
        self._frames:        dict[LocalID, Frame] | None         = None
        self._mappings:      dict[LocalID, Mapping] | None       = None
        self._bindings:      dict[LocalID, Binding] | None       = None
        self._examples:      dict[LocalID, Example] | None       = None
        self._lexical_units: dict[LocalID, LexicalUnit] | None   = None

    def _get(self, attr: str, extractor_cls: type[_Extractor]) -> dict:
        """Lazily extract and cache an entity collection by attribute name."""
        if getattr(self, attr) is None:
            setattr(self, attr, extractor_cls().extract(self._graph))
        return getattr(self, attr)

    @property
    def metaphors(self) -> dict[LocalID, MetaphorEntry]:
        return self._get("_metaphors", _MetaphorExtractor)

    @property
    def frames(self) -> dict[LocalID, Frame]:
        return self._get("_frames", _FrameExtractor)

    @property
    def mappings(self) -> dict[LocalID, Mapping]:
        return self._get("_mappings", _MappingExtractor)

    @property
    def bindings(self) -> dict[LocalID, Binding]:
        return self._get("_bindings", _BindingExtractor)

    @property
    def examples(self) -> dict[LocalID, Example]:
        return self._get("_examples", _ExampleExtractor)

    @property
    def lexical_units(self) -> dict[LocalID, LexicalUnit]:
        return self._get("_lexical_units", _LexicalUnitExtractor)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_metanet_cached(path_str: str) -> MetaNetRepository:
    """Parse an RDF file and return a cached MetaNetRepository."""
    g = Graph()
    try:
        g.parse(path_str)
    except FileNotFoundError:
        raise FileNotFoundError(f"MetaNet RDF file not found: '{path_str}'")
    except Exception as e:
        raise ValueError(f"Could not parse MetaNet RDF at '{path_str}': {e}") from e
    return MetaNetRepository(g)


def load_metanet(path: str | Path) -> MetaNetRepository:
    """
    Load a MetaNet RDF file and return a MetaNetRepository.
    Results are cached by resolved path.
    """
    return _load_metanet_cached(str(Path(path).resolve()))
