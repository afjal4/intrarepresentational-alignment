from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS

# ---------------------------------------------------------------------------
# Namespace URIs
# ---------------------------------------------------------------------------

_MN = "http://metanet.english.ubc.ca/metaphor/MetaphorOntology.owl#"
_MR = "https://metaphor.icsi.berkeley.edu/en/MetaphorRepository.owl#"

# rdflib URIRef shorthands for every property we read
_type              = RDF.type
_label             = RDFS.label

_Metaphor          = URIRef(_MN + "Metaphor")
_Frame             = URIRef(_MN + "Frame")
_Mapping           = URIRef(_MN + "Mapping")
_Binding           = URIRef(_MN + "Binding")
_Example           = URIRef(_MN + "Example")
_LexicalUnit       = URIRef(_MN + "LexicalUnit")

_hasName           = URIRef(_MN + "hasName")
_hasDescription    = URIRef(_MN + "hasDescription")
_hasStatus         = URIRef(_MN + "hasStatus")
_hasCulturalScope  = URIRef(_MN + "hasCulturalScope")
_wasInvestigatedFor= URIRef(_MN + "wasInvestigatedFor")

_hasSourceFrame    = URIRef(_MN + "hasSourceFrame")
_hasTargetFrame    = URIRef(_MN + "hasTargetFrame")
_hasMappings       = URIRef(_MN + "hasMappings")
_hasEntailment     = URIRef(_MN + "hasEntailment")
_hasExample        = URIRef(_MN + "hasExample")
_isEntailedByMetaphor       = URIRef(_MN + "isEntailedByMetaphor")
_isTargetSubcaseOfMetaphor  = URIRef(_MN + "isTargetSubcaseOfMetaphor")
_isSourceSubcaseOfMetaphor  = URIRef(_MN + "isSourceSubcaseOfMetaphor")
_isInMetaphorFamily         = URIRef(_MN + "isInMetaphorFamily")
_hasAlias          = URIRef(_MN + "hasAlias")
_hasMetaphorType   = URIRef(_MN + "hasMetaphorType")
_hasMetaphorLevel  = URIRef(_MN + "hasMetaphorLevel")
_hasFrenchCorrespondent  = URIRef(_MN + "hasFrenchCorrespondent")
_hasSpanishCorrespondent = URIRef(_MN + "hasSpanishCorrespondent")

_hasFrameType          = URIRef(_MN + "hasFrameType")
_hasRoles              = URIRef(_MN + "hasRoles")
_hasLexicalUnit        = URIRef(_MN + "hasLexicalUnit")
_hasInference          = URIRef(_MN + "hasInference")
_hasBindings           = URIRef(_MN + "hasBindings")
_isInFrameFamily       = URIRef(_MN + "isInFrameFamily")
_correspondsToFrameNet = URIRef(_MN + "correspondsToFrameNet")
_makesUseOfFrame       = URIRef(_MN + "makesUseOfFrame")

_hasSourceRole = URIRef(_MN + "hasSourceRole")
_hasTargetRole = URIRef(_MN + "hasTargetRole")

_hasBoundRole1 = URIRef(_MN + "hasBoundRole1")
_hasBoundRole2 = URIRef(_MN + "hasBoundRole2")

_hasSentence       = URIRef(_MN + "hasSentence")
_hasAnnotation     = URIRef(_MN + "hasAnnotation")
_exampleConstruct  = URIRef(_MN + "Example.Construction")
_exampleDialect    = URIRef(_MN + "Example.Dialect")
_isFromLanguage    = URIRef(_MN + "isFromLanguage")
_hasProvenance     = URIRef(_MN + "hasProvenance")

_hasLemma    = URIRef(_MN + "hasLemma")
_LUs_Lemmas  = URIRef(_MN + "LUs_Lemmas")
_LUs_Language= URIRef(_MN + "LUs_Language")


# ---------------------------------------------------------------------------
# Helpers
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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MetaphorEntry:
    # Unique local identifier derived from the RDF URI
    id: str

    # Canonical all-caps name of the conceptual metaphor (e.g. "KNOWING IS SEEING")
    name: str | None = None

    # The conceptual domain being mapped FROM (concrete / physical / experiential)
    source_frame: str | None = None

    # The conceptual domain being mapped TO (abstract / target of understanding)
    target_frame: str | None = None

    # Explicit role-level correspondences between source and target frame participants
    mappings: list[str] = field(default_factory=list)

    # Logical consequences of the mapping — inferences that follow from applying it
    entailments: list[str] = field(default_factory=list)

    # Attested corpus sentences that instantiate this metaphor
    examples: list[str] = field(default_factory=list)

    # More general metaphor(s) that logically give rise to this one
    is_entailed_by: list[str] = field(default_factory=list)

    # More specific metaphors that share this metaphor's target domain
    is_target_subcase_of: list[str] = field(default_factory=list)

    # More specific metaphors that share this metaphor's source domain
    is_source_subcase_of: list[str] = field(default_factory=list)

    # Named groupings of thematically related metaphors in the repository
    metaphor_families: list[str] = field(default_factory=list)

    # Alternative name string used in the repository's alias system
    alias: str | None = None

    # Cultural/dialectal communities for which this metaphor has been verified
    cultural_scope: list[str] = field(default_factory=list)

    # Classification of the metaphor (e.g. "Primary", "Entailed", "Composed/complex")
    metaphor_type: list[str] = field(default_factory=list)

    # Degree of specificity in the metaphor hierarchy ("General" or "Specific")
    metaphor_level: str | None = None

    # Editorial status of the entry ("reviewed", "in development", etc.)
    status: str | None = None

    # Free-text prose explanation of the mapping and its motivation
    description: list[str] = field(default_factory=list)

    # French-language equivalent conceptual metaphor
    french_correspondent: str | None = None

    # Spanish-language equivalent conceptual metaphor
    spanish_correspondent: str | None = None

    # Languages for which this metaphor has been empirically investigated
    was_investigated_for: list[str] = field(default_factory=list)


@dataclass
class Frame:
    # Unique local identifier derived from the RDF URI
    id: str

    # Human-readable name of the semantic frame
    name: str | None = None

    # Prose definition of the scene or situation the frame describes
    description: str | None = None

    # Structural category of the frame ("Frame" for full frames, "Image Schema" for basic
    # embodied schemas like PATH or CONTAINER)
    frame_type: str | None = None

    # Semantic roles (participants / props) defined within this frame
    roles: list[str] = field(default_factory=list)

    # Words and phrases (lemmas + POS) that linguistically evoke this frame
    lexical_units: list[str] = field(default_factory=list)

    # Statements about what is typically true or implied within this frame's scenario
    inferences: list[str] = field(default_factory=list)

    # Role-pair links connecting this frame's roles to roles in a related frame
    bindings: list[str] = field(default_factory=list)

    # Named thematic groupings that this frame belongs to
    frame_families: list[str] = field(default_factory=list)

    # Corresponding FrameNet frame name, enabling cross-resource alignment
    corresponds_to_framenet: str | None = None

    # Other frames whose structure is reused as sub-structure within this frame
    makes_use_of_frames: list[str] = field(default_factory=list)

    # Cultural/dialectal communities for which this frame has been verified
    cultural_scope: list[str] = field(default_factory=list)

    # Editorial status of the entry
    status: str | None = None


@dataclass
class Mapping:
    # Unique local identifier derived from the RDF URI
    id: str

    # The participant role in the source (concrete) frame
    source_role: str | None = None

    # The participant role in the target (abstract) frame that the source role maps onto
    target_role: str | None = None

    # Human-readable summary of the role correspondence (e.g. "Target.role<=Source.role")
    label: str | None = None


@dataclass
class Binding:
    # Unique local identifier derived from the RDF URI
    id: str

    # First of the two frame roles being bound together
    bound_role_1: str | None = None

    # Second of the two frame roles being bound together
    bound_role_2: str | None = None

    # Human-readable summary of the binding (e.g. "Frame:A.roleX=Frame:B.roleY")
    label: str | None = None


@dataclass
class Example:
    # Unique local identifier derived from the RDF URI
    id: str

    # Full sentence from a real text corpus that instantiates the metaphor
    sentence: str | None = None

    # The specific phrase within the sentence that was annotated as metaphorical
    annotation: str | None = None

    # Syntactic construction pattern used (e.g. "Noun(S) preposition Noun(T)")
    construction: str | None = None

    # Regional or social dialect of the example (e.g. "American English")
    dialect: str | None = None

    # Natural language of the example sentence
    language: str | None = None

    # URL or bibliographic reference identifying the source of the sentence
    provenance: str | None = None


@dataclass
class LexicalUnit:
    # Unique local identifier derived from the RDF URI
    id: str

    # Single base form (lemma) that evokes the associated frame
    lemma: str | None = None

    # Pipe-separated list of lemmas when multiple forms are grouped together
    lemmas: str | None = None

    # Natural language of the lexical unit
    language: str | None = None


@dataclass
class MetaNetRepository:
    metaphors:     dict[str, MetaphorEntry]
    frames:        dict[str, Frame]
    mappings:      dict[str, Mapping]
    bindings:      dict[str, Binding]
    examples:      dict[str, Example]
    lexical_units: dict[str, LexicalUnit]


# ---------------------------------------------------------------------------
# Extractors (one per entity type)
# ---------------------------------------------------------------------------

def _extract_metaphors(g: Graph) -> dict[str, MetaphorEntry]:
    result: dict[str, MetaphorEntry] = {}
    for s in g.subjects(_type, _Metaphor):
        eid = _local(s)
        result[eid] = MetaphorEntry(
            id                  = eid,
            name                = _str(g, s, _hasName),
            source_frame        = _ref(g, s, _hasSourceFrame),
            target_frame        = _ref(g, s, _hasTargetFrame),
            mappings            = _refs(g, s, _hasMappings),
            entailments         = _refs(g, s, _hasEntailment),
            examples            = _refs(g, s, _hasExample),
            is_entailed_by      = _refs(g, s, _isEntailedByMetaphor),
            is_target_subcase_of= _refs(g, s, _isTargetSubcaseOfMetaphor),
            is_source_subcase_of= _refs(g, s, _isSourceSubcaseOfMetaphor),
            metaphor_families   = _refs(g, s, _isInMetaphorFamily),
            alias               = _str(g, s, _hasAlias),
            cultural_scope      = _strs(g, s, _hasCulturalScope),
            metaphor_type       = _strs(g, s, _hasMetaphorType),
            metaphor_level      = _str(g, s, _hasMetaphorLevel),
            status              = _str(g, s, _hasStatus),
            description         = _strs(g, s, _hasDescription),
            french_correspondent= _str(g, s, _hasFrenchCorrespondent),
            spanish_correspondent=_str(g, s, _hasSpanishCorrespondent),
            was_investigated_for= _strs(g, s, _wasInvestigatedFor),
        )
    return result


def _extract_frames(g: Graph) -> dict[str, Frame]:
    result: dict[str, Frame] = {}
    for s in g.subjects(_type, _Frame):
        eid = _local(s)
        result[eid] = Frame(
            id                    = eid,
            name                  = _str(g, s, _hasName),
            description           = _str(g, s, _hasDescription),
            frame_type            = _str(g, s, _hasFrameType),
            roles                 = _refs(g, s, _hasRoles),
            lexical_units         = _refs(g, s, _hasLexicalUnit),
            inferences            = _refs(g, s, _hasInference),
            bindings              = _refs(g, s, _hasBindings),
            frame_families        = _refs(g, s, _isInFrameFamily),
            corresponds_to_framenet = _str(g, s, _correspondsToFrameNet),
            makes_use_of_frames   = _refs(g, s, _makesUseOfFrame),
            cultural_scope        = _strs(g, s, _hasCulturalScope),
            status                = _str(g, s, _hasStatus),
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
            id          = eid,
            bound_role_1= _ref(g, s, _hasBoundRole1),
            bound_role_2= _ref(g, s, _hasBoundRole2),
            label       = _str(g, s, _label),
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
