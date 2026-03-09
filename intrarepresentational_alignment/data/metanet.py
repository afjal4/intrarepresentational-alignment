from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetaphorEntry:
    # Unique ID from the RDF URI
    id: str

    # All-caps name of the conceptual metaphor
    name: str | None = None

    # Alternative name string
    alias: str | None = None

    # The source conceptual domain (concrete / physical / experiential)
    source_frame: str | None = None

    # The target conceptual domain (abstract / target of understanding)
    target_frame: str | None = None

    # Explicit role-level correspondences between source and target frame participants
    mappings: list[str] = field(default_factory=list)

    # Logical consequences of the mapping - inferences that follow from applying it
    entailments: list[str] = field(default_factory=list)

    # Attested corpus sentences that instantiate this metaphor
    examples: list[str] = field(default_factory=list)

    # General metaphors that logically give rise to this one
    is_entailed_by: list[str] = field(default_factory=list)
    # Specific metaphors that share this metaphor's target domain
    is_target_subcase_of: list[str] = field(default_factory=list)
    # Specific metaphors that share this metaphor's source domain
    is_source_subcase_of: list[str] = field(default_factory=list)
    # Named groupings of thematically related metaphors in the repository
    metaphor_families: list[str] = field(default_factory=list)

    # Cultural/dialectal communities for which this metaphor has been verified
    cultural_scope: list[str] = field(default_factory=list)

    # Classification of the metaphor (e.g. "Primary", "Entailed", "Composed/complex")
    metaphor_type: list[str] = field(default_factory=list)
    # Degree of specificity in the metaphor hierarchy ("General" or "Specific")
    metaphor_level: str | None = None

    # Explanation of the mapping and its motivation
    description: list[str] = field(default_factory=list)

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
    id: str

    # Human-readable name of the semantic frame
    name: str | None = None

    # Definition of the scene or situation the frame describes
    description: str | None = None

    # Structural category of the frame
    #  "Frame" for full frames,
    #  "Image Schema" for basic embodied schemas like PATH or CONTAINER
    frame_type: str | None = None

    # Semantic roles (participants / props) defined within this frame
    roles: list[str] = field(default_factory=list)

    # Words and phrases (lemmas + POS) that linguistically evoke this frame
    lexical_units: list[str] = field(default_factory=list)

    # Statements about what is implied within this frame's scenario
    inferences: list[str] = field(default_factory=list)

    # Role-pair links connecting this frame's roles to roles in a related frame
    bindings: list[str] = field(default_factory=list)

    # Named thematic groupings that this frame belongs to
    frame_families: list[str] = field(default_factory=list)

    # Corresponding FrameNet frame name
    corresponds_to_framenet: str | None = None

    # Other frames whose structure is reused as sub-structure within this frame
    makes_use_of_frames: list[str] = field(default_factory=list)

    # Cultural/dialectal communities for which this frame has been verified
    cultural_scope: list[str] = field(default_factory=list)

    # Editorial status of the entry
    status: str | None = None


@dataclass
class Mapping:
    # Unique ID from the RDF URI
    id: str

    # The participant role in the source (concrete) frame
    source_role: str | None = None
    # The participant role in the target (abstract) frame that the source role maps onto
    target_role: str | None = None

    # Human-readable summary of the role correspondence (e.g. "Target.role<=Source.role")
    label: str | None = None


@dataclass
class Binding:
    # Unique ID from the RDF URI
    id: str

    # First and second of bound frame roles
    bound_role_1: str | None = None
    bound_role_2: str | None = None

    # Human-readable summary of the binding (e.g. "Frame:A.roleX=Frame:B.roleY")
    label: str | None = None


@dataclass
class Example:
    # Unique ID from the RDF URI
    id: str

    # Full sentence that instantiates the metaphor
    sentence: str | None = None
    # Language of the example sentence
    language: str | None = None

    # The specific phrase within the sentence that was annotated as metaphorical
    annotation: str | None = None

    # Syntactic construction pattern used (e.g. "Noun(S) preposition Noun(T)")
    construction: str | None = None

    # Regional or social dialect of the example (e.g. "American English")
    dialect: str | None = None

    # URL or bibliographic reference identifying the source of the sentence
    provenance: str | None = None


@dataclass
class LexicalUnit:
    # Unique ID from the RDF URI
    id: str

    # Canonical form (lemma)
    lemma: str | None = None
    # Pipe-separated list of lemmas when multiple forms are grouped together
    lemmas: str | None = None

    # Language of the lexical unit
    language: str | None = None


@dataclass
class MetaNetRepository:
    metaphors:     dict[str, MetaphorEntry]
    frames:        dict[str, Frame]
    mappings:      dict[str, Mapping]
    bindings:      dict[str, Binding]
    examples:      dict[str, Example]
    lexical_units: dict[str, LexicalUnit]
