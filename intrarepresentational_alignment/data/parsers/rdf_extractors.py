from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from rdflib import Graph, Literal, URIRef

from ..models import Binding, Example, Frame, LexicalUnit, LocalID, Mapping, MetaphorEntry
from ..rdf.namespaces import (
    _Binding,
    _Example,
    _Frame,
    _LexicalUnit,
    _Mapping,
    _Metaphor,
    _LUs_Language,
    _LUs_Lemmas,
    _correspondsToFrameNet,
    _exampleConstruct,
    _exampleDialect,
    _hasAlias,
    _hasAnnotation,
    _hasBindings,
    _hasBoundRole1,
    _hasBoundRole2,
    _hasCulturalScope,
    _hasDescription,
    _hasEntailment,
    _hasExample,
    _hasFrameType,
    _hasFrenchCorrespondent,
    _hasInference,
    _hasLemma,
    _hasLexicalUnit,
    _hasMappings,
    _hasMetaphorLevel,
    _hasMetaphorType,
    _hasName,
    _hasProvenance,
    _hasRoles,
    _hasSentence,
    _hasSourceFrame,
    _hasSourceRole,
    _hasSpanishCorrespondent,
    _hasStatus,
    _hasTargetFrame,
    _hasTargetRole,
    _isEntailedByMetaphor,
    _isFromLanguage,
    _isInFrameFamily,
    _isInMetaphorFamily,
    _isSourceSubcaseOfMetaphor,
    _isTargetSubcaseOfMetaphor,
    _label,
    _makesUseOfFrame,
    _type,
    _wasInvestigatedFor,
)

T = TypeVar("T")


def _local(uri: URIRef) -> LocalID:
    s = str(uri)
    if "#" in s:
        return LocalID(s.split("#")[-1])
    return LocalID(s.split("/")[-1])


def _str(g: Graph, s: URIRef, p: URIRef) -> str | None:
    for o in g.objects(s, p):
        if isinstance(o, Literal):
            return str(o)
    return None


def _strs(g: Graph, s: URIRef, p: URIRef) -> list[str]:
    return [str(o) for o in g.objects(s, p) if isinstance(o, Literal)]


def _refs(g: Graph, s: URIRef, p: URIRef) -> list[LocalID]:
    return [_local(o) for o in g.objects(s, p) if isinstance(o, URIRef)]


def _ref(g: Graph, s: URIRef, p: URIRef) -> LocalID | None:
    refs = _refs(g, s, p)
    return refs[0] if refs else None


class _Extractor(ABC, Generic[T]):
    rdf_type: URIRef

    def extract(self, g: Graph) -> dict[LocalID, T]:
        result: dict[LocalID, T] = {}
        for s in g.subjects(_type, self.rdf_type):
            eid = _local(s)
            result[eid] = self._build(g, s, eid)
        return result

    @abstractmethod
    def _build(self, g: Graph, s: URIRef, eid: LocalID) -> T:
        ...


class _MetaphorExtractor(_Extractor[MetaphorEntry]):
    rdf_type = _Metaphor

    def _build(self, g: Graph, s: URIRef, eid: LocalID) -> MetaphorEntry:
        return MetaphorEntry(
            id=eid,
            name=_str(g, s, _hasName),
            source_frame=_ref(g, s, _hasSourceFrame),
            target_frame=_ref(g, s, _hasTargetFrame),
            mappings=_refs(g, s, _hasMappings),
            entailments=_refs(g, s, _hasEntailment),
            examples=_refs(g, s, _hasExample),
            is_entailed_by=_refs(g, s, _isEntailedByMetaphor),
            is_target_subcase_of=_refs(g, s, _isTargetSubcaseOfMetaphor),
            is_source_subcase_of=_refs(g, s, _isSourceSubcaseOfMetaphor),
            metaphor_families=_refs(g, s, _isInMetaphorFamily),
            alias=_str(g, s, _hasAlias),
            cultural_scope=_strs(g, s, _hasCulturalScope),
            metaphor_type=_strs(g, s, _hasMetaphorType),
            metaphor_level=_str(g, s, _hasMetaphorLevel),
            status=_str(g, s, _hasStatus),
            description=_str(g, s, _hasDescription),
            french_correspondent=_str(g, s, _hasFrenchCorrespondent),
            spanish_correspondent=_str(g, s, _hasSpanishCorrespondent),
            was_investigated_for=_strs(g, s, _wasInvestigatedFor),
        )


class _FrameExtractor(_Extractor[Frame]):
    rdf_type = _Frame

    def _build(self, g: Graph, s: URIRef, eid: LocalID) -> Frame:
        return Frame(
            id=eid,
            name=_str(g, s, _hasName),
            description=_str(g, s, _hasDescription),
            frame_type=_str(g, s, _hasFrameType),
            roles=_refs(g, s, _hasRoles),
            lexical_units=_refs(g, s, _hasLexicalUnit),
            inferences=_refs(g, s, _hasInference),
            bindings=_refs(g, s, _hasBindings),
            frame_families=_refs(g, s, _isInFrameFamily),
            corresponds_to_framenet=_str(g, s, _correspondsToFrameNet),
            makes_use_of_frames=_refs(g, s, _makesUseOfFrame),
            cultural_scope=_strs(g, s, _hasCulturalScope),
            status=_str(g, s, _hasStatus),
        )


class _MappingExtractor(_Extractor[Mapping]):
    rdf_type = _Mapping

    def _build(self, g: Graph, s: URIRef, eid: LocalID) -> Mapping:
        return Mapping(
            id=eid,
            source_role=_ref(g, s, _hasSourceRole),
            target_role=_ref(g, s, _hasTargetRole),
            label=_str(g, s, _label),
        )


class _BindingExtractor(_Extractor[Binding]):
    rdf_type = _Binding

    def _build(self, g: Graph, s: URIRef, eid: LocalID) -> Binding:
        return Binding(
            id=eid,
            bound_role_1=_ref(g, s, _hasBoundRole1),
            bound_role_2=_ref(g, s, _hasBoundRole2),
            label=_str(g, s, _label),
        )


class _ExampleExtractor(_Extractor[Example]):
    rdf_type = _Example

    def _build(self, g: Graph, s: URIRef, eid: LocalID) -> Example:
        return Example(
            id=eid,
            sentence=_str(g, s, _hasSentence),
            annotation=_str(g, s, _hasAnnotation),
            construction=_str(g, s, _exampleConstruct),
            dialect=_str(g, s, _exampleDialect),
            language=_str(g, s, _isFromLanguage),
            provenance=_str(g, s, _hasProvenance),
        )


class _LexicalUnitExtractor(_Extractor[LexicalUnit]):
    rdf_type = _LexicalUnit

    def _build(self, g: Graph, s: URIRef, eid: LocalID) -> LexicalUnit:
        raw_lemmas = _str(g, s, _LUs_Lemmas)
        return LexicalUnit(
            id=eid,
            lemma=_str(g, s, _hasLemma),
            lemmas=raw_lemmas.split("|") if raw_lemmas else [],
            language=_str(g, s, _LUs_Language),
        )


def default_extractor_registry() -> dict[str, type[_Extractor]]:
    return {
        "metaphors": _MetaphorExtractor,
        "frames": _FrameExtractor,
        "mappings": _MappingExtractor,
        "bindings": _BindingExtractor,
        "examples": _ExampleExtractor,
        "lexical_units": _LexicalUnitExtractor,
    }
