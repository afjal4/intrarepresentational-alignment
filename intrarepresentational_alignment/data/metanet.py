from __future__ import annotations

from dataclasses import dataclass, field
from typing import NewType

# Local RDF fragment ID (after # or last / in a URI).
LocalID = NewType("LocalID", str)


@dataclass
class MetaphorEntry:
    # Unique ID from the RDF URI
    id: LocalID

    # All-caps name of the conceptual metaphor
    name: str | None = None

    # Alternative name string
    alias: str | None = None

    # The source conceptual domain (KNOWLEDGE in KNOWING IS SEEING)
    source_frame: LocalID | None = None

    # The target conceptual domain (SEER in KNOWING IS SEEING)
    target_frame: LocalID | None = None

    # Mappings between source and target frame participants (KNOWER <-> SEER)
    mappings: list[LocalID] = field(default_factory=list)

    # 'Second-order' inferences that logically follow from the mappings
    entailments: list[LocalID] = field(default_factory=list)

    # Attested corpus sentences that instantiate this metaphor
    examples: list[LocalID] = field(default_factory=list)

    # General metaphors that logically give rise to this one
    is_entailed_by: list[LocalID] = field(default_factory=list)
    # Specific metaphors that share this metaphor's target domain
    is_target_subcase_of: list[LocalID] = field(default_factory=list)
    # Specific metaphors that share this metaphor's source domain
    is_source_subcase_of: list[LocalID] = field(default_factory=list)
    # Named groupings of thematically related metaphors in the repository
    metaphor_families: list[LocalID] = field(default_factory=list)

    # Cultural/dialectal communities for which this metaphor has been verified
    cultural_scope: list[str] = field(default_factory=list)

    # Classification of the metaphor (e.g. "Primary", "Entailed", "Composed/complex")
    metaphor_type: list[str] = field(default_factory=list)
    # Degree of specificity in the metaphor hierarchy ("General" or "Specific")
    metaphor_level: str | None = None

    # Explanation of the mapping and its motivation
    description: str | None = None

    # French-language equivalent
    french_correspondent: str | None = None
    # Spanish-language equivalent
    spanish_correspondent: str | None = None
    # Languages for which this metaphor has been empirically investigated
    was_investigated_for: list[str] = field(default_factory=list)

    # Editorial status of the entry
    status: str | None = None


@dataclass
class Frame:
    # Unique ID from the RDF URI
    id: LocalID

    # Human-readable name of the semantic frame
    name: str | None = None

    # Definition of the scene or situation the frame describes
    description: str | None = None

    # Structural category of the frame
    #  "Frame" for full frames,
    #  "Image Schema" for basic embodied schemas like PATH or CONTAINER
    frame_type: str | None = None

    # Semantic roles (participants / props) defined within this frame
    roles: list[LocalID] = field(default_factory=list)

    # Pairing of a lemma with a Part-Of-Speech (POS) tag
    lexical_units: list[LocalID] = field(default_factory=list)

    # Statements about what is implied within this frame's scenario
    inferences: list[LocalID] = field(default_factory=list)

    # Role-pair links connecting this frame's roles to roles in a related frame
    bindings: list[LocalID] = field(default_factory=list)

    # Named thematic groupings that this frame belongs to
    frame_families: list[LocalID] = field(default_factory=list)

    # Corresponding FrameNet frame name (external reference, not a LocalID)
    corresponds_to_framenet: str | None = None

    # Other frames whose structure is reused as sub-structure within this frame
    makes_use_of_frames: list[LocalID] = field(default_factory=list)

    # Cultural/dialectal communities for which this frame has been verified
    cultural_scope: list[str] = field(default_factory=list)

    # Editorial status of the entry
    status: str | None = None


@dataclass
class Mapping:
    # Unique ID from the RDF URI
    id: LocalID

    # The participant role in the source (concrete) frame
    source_role: LocalID | None = None
    # The participant role in the target (abstract) frame that the source role maps onto
    target_role: LocalID | None = None

    # Human-readable summary of the role correspondence (e.g. "Target.role<=Source.role")
    label: str | None = None


@dataclass
class Binding:
    # Unique ID from the RDF URI
    id: LocalID

    # First and second of bound frame roles
    bound_role_1: LocalID | None = None
    bound_role_2: LocalID | None = None

    # Human-readable summary of the binding (e.g. "Frame:A.roleX=Frame:B.roleY")
    label: str | None = None


@dataclass
class Example:
    # Unique ID from the RDF URI
    id: LocalID

    # Full sentence that instantiates the metaphor
    sentence: str | None = None

    # Language of the example sentence
    language: str | None = None
    # Regional or social dialect of the example (e.g. "American English")
    dialect: str | None = None

    # The specific phrase within the sentence that was annotated as metaphorical
    annotation: str | None = None

    # Syntactic construction pattern used (e.g. "Noun(S) preposition Noun(T)")
    construction: str | None = None

    # URL or bibliographic reference
    provenance: str | None = None


@dataclass
class LexicalUnit:
    # Unique ID from the RDF URI
    id: LocalID

    # Canonical form (lemma)
    lemma: str | None = None
    # All lemma forms
    lemmas: list[str] = field(default_factory=list)

    # Language of the lexical unit
    language: str | None = None


@dataclass
class LccInstance:
    # Unique instance ID from the corpus
    id: str

    # Document ID
    doc_id: str

    # Abstract target concept category (e.g. "POVERTY", "TAXATION")
    target_concept: str

    # Corpus validation type ("RECALL_VALIDATIONS" or "SYSTEM_VALIDATIONS")
    instance_type: str

    # Syntactic dependency chain pattern (e.g. "*:prep_on")
    chain: str

    # Surrounding sentence context
    prev_sentence: str | None = None
    current_sentence: str | None = None
    next_sentence: str | None = None

    # Source domain expressions: the concrete/vehicle phrases from <LmSource> tags
    source_expressions: list[str] = field(default_factory=list)

    # Target domain expressions: the abstract/topic phrases from <LmTarget> tags
    target_expressions: list[str] = field(default_factory=list)

    # Conceptual source domain label from <CMSourceAnnotation> (e.g. "DISEASE",
    # "STRUGGLE").  Only present on a subset of instances; None if unannotated.
    source_concept: str | None = None

    # Metaphoricity scores (1.0 = literal, 3.0 = highly metaphorical)
    metaphoricity_scores: list[float] = field(default_factory=list)

    # Polarity labels per annotator ("POSITIVE", "NEGATIVE", "NEUTRAL")
    polarity_labels: list[str] = field(default_factory=list)

    # Intensity scores per annotator (0.0–3.0)
    intensity_scores: list[float] = field(default_factory=list)
