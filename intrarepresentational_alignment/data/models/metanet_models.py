from __future__ import annotations

from dataclasses import dataclass, field
from typing import NewType

LocalID = NewType("LocalID", str)


@dataclass
class MetaphorEntry:
    id: LocalID
    name: str | None = None
    alias: str | None = None
    source_frame: LocalID | None = None
    target_frame: LocalID | None = None
    mappings: list[LocalID] = field(default_factory=list)
    entailments: list[LocalID] = field(default_factory=list)
    examples: list[LocalID] = field(default_factory=list)
    is_entailed_by: list[LocalID] = field(default_factory=list)
    is_target_subcase_of: list[LocalID] = field(default_factory=list)
    is_source_subcase_of: list[LocalID] = field(default_factory=list)
    metaphor_families: list[LocalID] = field(default_factory=list)
    cultural_scope: list[str] = field(default_factory=list)
    metaphor_type: list[str] = field(default_factory=list)
    metaphor_level: str | None = None
    description: str | None = None
    french_correspondent: str | None = None
    spanish_correspondent: str | None = None
    was_investigated_for: list[str] = field(default_factory=list)
    status: str | None = None


@dataclass
class Frame:
    id: LocalID
    name: str | None = None
    description: str | None = None
    frame_type: str | None = None
    roles: list[LocalID] = field(default_factory=list)
    lexical_units: list[LocalID] = field(default_factory=list)
    inferences: list[LocalID] = field(default_factory=list)
    bindings: list[LocalID] = field(default_factory=list)
    frame_families: list[LocalID] = field(default_factory=list)
    corresponds_to_framenet: str | None = None
    makes_use_of_frames: list[LocalID] = field(default_factory=list)
    cultural_scope: list[str] = field(default_factory=list)
    status: str | None = None


@dataclass
class Mapping:
    id: LocalID
    source_role: LocalID | None = None
    target_role: LocalID | None = None
    label: str | None = None


@dataclass
class Binding:
    id: LocalID
    bound_role_1: LocalID | None = None
    bound_role_2: LocalID | None = None
    label: str | None = None


@dataclass
class Example:
    id: LocalID
    sentence: str | None = None
    language: str | None = None
    dialect: str | None = None
    annotation: str | None = None
    construction: str | None = None
    provenance: str | None = None


@dataclass
class LexicalUnit:
    id: LocalID
    lemma: str | None = None
    lemmas: list[str] = field(default_factory=list)
    language: str | None = None
