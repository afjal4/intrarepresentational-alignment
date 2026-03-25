from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LccInstance:
    id: str
    doc_id: str
    target_concept: str
    instance_type: str
    chain: str
    prev_sentence: str | None = None
    current_sentence: str | None = None
    next_sentence: str | None = None
    source_expressions: list[str] = field(default_factory=list)
    target_expressions: list[str] = field(default_factory=list)
    source_concept: str | None = None
    metaphoricity_scores: list[float] = field(default_factory=list)
    polarity_labels: list[str] = field(default_factory=list)
    intensity_scores: list[float] = field(default_factory=list)
