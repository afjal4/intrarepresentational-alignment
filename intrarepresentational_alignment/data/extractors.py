from __future__ import annotations

from pathlib import Path

from rdflib import Graph, Literal, URIRef

from .metanet import (
    MetaNetRepository,
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


# ---------------------------------------------------------------------------
# RDF query helpers
# ---------------------------------------------------------------------------

def _local(uri: URIRef) -> str:
    """Return the local fragment of a URI (everything after # or last /)."""
    s = str(uri)
    if "#" in s:
        return s.split("#")[-1]
    return s.split("/")[-1]


def _str(g: Graph, s: URIRef, p: URIRef) -> str | None:
    """Return the first literal value for a single-valued property, or None."""
    for o in g.objects(s, p):
        if isinstance(o, Literal):
            return str(o)
    return None


def _strs(g: Graph, s: URIRef, p: URIRef) -> list[str]:
    """Return all literal values for a multi-valued property."""
    return [str(o) for o in g.objects(s, p) if isinstance(o, Literal)]


def _refs(g: Graph, s: URIRef, p: URIRef) -> list[str]:
    """Return local IDs of all URI-reference objects for a property."""
    return [_local(o) for o in g.objects(s, p) if isinstance(o, URIRef)]


def _ref(g: Graph, s: URIRef, p: URIRef) -> str | None:
    """Return the local ID of the first URI-reference object, or None."""
    refs = _refs(g, s, p)
    return refs[0] if refs else None


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _extract_metaphors(g: Graph) -> dict[str, MetaphorEntry]:
    result: dict[str, MetaphorEntry] = {}
    for s in g.subjects(_type, _Metaphor):
        eid = _local(s)
        result[eid] = MetaphorEntry(
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
            description           = _strs(g, s, _hasDescription),
            french_correspondent  = _str(g, s, _hasFrenchCorrespondent),
            spanish_correspondent = _str(g, s, _hasSpanishCorrespondent),
            was_investigated_for  = _strs(g, s, _wasInvestigatedFor),
        )
    return result


def _extract_frames(g: Graph) -> dict[str, Frame]:
    result: dict[str, Frame] = {}
    for s in g.subjects(_type, _Frame):
        eid = _local(s)
        result[eid] = Frame(
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
    return result


def _extract_mappings(g: Graph) -> dict[str, Mapping]:
    result: dict[str, Mapping] = {}
    for s in g.subjects(_type, _Mapping):
        eid = _local(s)
        result[eid] = Mapping(
            id          = eid,
            source_role = _ref(g, s, _hasSourceRole),
            target_role = _ref(g, s, _hasTargetRole),
            label       = _str(g, s, _label),
        )
    return result


def _extract_bindings(g: Graph) -> dict[str, Binding]:
    result: dict[str, Binding] = {}
    for s in g.subjects(_type, _Binding):
        eid = _local(s)
        result[eid] = Binding(
            id           = eid,
            bound_role_1 = _ref(g, s, _hasBoundRole1),
            bound_role_2 = _ref(g, s, _hasBoundRole2),
            label        = _str(g, s, _label),
        )
    return result


def _extract_examples(g: Graph) -> dict[str, Example]:
    result: dict[str, Example] = {}
    for s in g.subjects(_type, _Example):
        eid = _local(s)
        result[eid] = Example(
            id           = eid,
            sentence     = _str(g, s, _hasSentence),
            annotation   = _str(g, s, _hasAnnotation),
            construction = _str(g, s, _exampleConstruct),
            dialect      = _str(g, s, _exampleDialect),
            language     = _str(g, s, _isFromLanguage),
            provenance   = _str(g, s, _hasProvenance),
        )
    return result


def _extract_lexical_units(g: Graph) -> dict[str, LexicalUnit]:
    result: dict[str, LexicalUnit] = {}
    for s in g.subjects(_type, _LexicalUnit):
        eid = _local(s)
        result[eid] = LexicalUnit(
            id       = eid,
            lemma    = _str(g, s, _hasLemma),
            lemmas   = _str(g, s, _LUs_Lemmas),
            language = _str(g, s, _LUs_Language),
        )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_metanet(path: str | Path) -> MetaNetRepository:
    g = Graph()
    g.parse(str(path))
    return MetaNetRepository(
        metaphors     = _extract_metaphors(g),
        frames        = _extract_frames(g),
        mappings      = _extract_mappings(g),
        bindings      = _extract_bindings(g),
        examples      = _extract_examples(g),
        lexical_units = _extract_lexical_units(g),
    )
