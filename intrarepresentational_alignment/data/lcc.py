from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

from .models import LccInstance

# ---------------------------------------------------------------------------
# XML schema constants
# ---------------------------------------------------------------------------

_TAG_LM_INSTANCE   = "LmInstance"
_TAG_TEXT_CONTENT  = "TextContent"
_TAG_PREV          = "Prev"
_TAG_CURRENT       = "Current"
_TAG_NEXT          = "Next"
_TAG_LM_SOURCE     = "LmSource"
_TAG_LM_TARGET     = "LmTarget"
_TAG_ANNOTATIONS   = "Annotations"
_TAG_METAPHORICITY = "MetaphoricityAnnotation"
_TAG_POLARITY      = "PolarityAnnotation"
_TAG_INTENSITY     = "IntensityAnnotation"
_TAG_CM_SOURCE     = "CMSourceAnnotation"

_ATTR_ID             = "id"
_ATTR_DOC_ID         = "docid"
_ATTR_TARGET_CONCEPT = "targetConcept"
_ATTR_TYPE           = "type"
_ATTR_CHAIN          = "chain"
_ATTR_SCORE          = "score"
_ATTR_POLARITY       = "polarity"
_ATTR_INTENSITY      = "intensity"
_ATTR_SOURCE_CONCEPT = "sourceConcept"


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def _element_full_text(elem: ET.Element | None) -> str | None:
    """Return all text within *elem*, collapsing inline child elements.

    Handles mixed content like:
        <Current>The <LmSource>War</LmSource> on <LmTarget>Poverty</LmTarget>.</Current>
    → "The War on Poverty."
    """
    if elem is None:
        return None
    parts = [elem.text or ""]
    for child in elem:
        parts.append(child.text or "")
        parts.append(child.tail or "")
    text = "".join(parts).strip()
    return text or None


def _child_texts(elem: ET.Element, tag: str) -> list[str]:
    """Return stripped text content of every descendant with the given *tag*."""
    return [
        c.text.strip()
        for c in elem.iter(tag)
        if c.text and c.text.strip()
    ]


def _float_attrs(elem: ET.Element, child_tag: str, attr: str) -> list[float]:
    """Return float values of *attr* from every *child_tag* descendant."""
    results: list[float] = []
    for child in elem.iter(child_tag):
        raw = child.get(attr)
        if raw is not None:
            try:
                results.append(float(raw))
            except ValueError:
                pass
    return results


def _str_attrs(elem: ET.Element, child_tag: str, attr: str) -> list[str]:
    """Return string values of *attr* from every *child_tag* descendant."""
    return [
        child.get(attr)
        for child in elem.iter(child_tag)
        if child.get(attr) is not None
    ]


# ---------------------------------------------------------------------------
# Instance builder
# ---------------------------------------------------------------------------

def _build_instance(elem: ET.Element) -> LccInstance:
    """Parse one <LmInstance> element into an LccInstance."""
    text_content = elem.find(_TAG_TEXT_CONTENT)
    annotations = elem.find(_TAG_ANNOTATIONS)

    prev = _element_full_text(text_content.find(_TAG_PREV)) if text_content is not None else None
    current = _element_full_text(text_content.find(_TAG_CURRENT)) if text_content is not None else None
    nxt = _element_full_text(text_content.find(_TAG_NEXT)) if text_content is not None else None

    source_exprs: list[str] = []
    target_exprs: list[str] = []
    if text_content is not None:
        current_elem = text_content.find(_TAG_CURRENT)
        if current_elem is not None:
            source_exprs = _child_texts(current_elem, _TAG_LM_SOURCE)
            target_exprs = _child_texts(current_elem, _TAG_LM_TARGET)

    metaphoricity: list[float] = []
    polarity: list[str] = []
    intensity: list[float] = []
    source_concept: str | None = None
    if annotations is not None:
        metaphoricity = _float_attrs(annotations, _TAG_METAPHORICITY, _ATTR_SCORE)
        polarity = _str_attrs(annotations, _TAG_POLARITY, _ATTR_POLARITY)
        intensity = _float_attrs(annotations, _TAG_INTENSITY, _ATTR_INTENSITY)
        # Pick the highest-scored CMSourceAnnotation (score > 0) as the
        # conceptual source domain label (e.g. "DISEASE", "STRUGGLE").
        best = max(
            (a for a in annotations.iter(_TAG_CM_SOURCE)
             if float(a.get(_ATTR_SCORE, 0)) > 0),
            key=lambda a: float(a.get(_ATTR_SCORE, 0)),
            default=None,
        )
        if best is not None:
            source_concept = best.get(_ATTR_SOURCE_CONCEPT)

    return LccInstance(
        id                   = elem.get(_ATTR_ID, ""),
        doc_id               = elem.get(_ATTR_DOC_ID, ""),
        target_concept       = elem.get(_ATTR_TARGET_CONCEPT, ""),
        instance_type        = elem.get(_ATTR_TYPE, ""),
        chain                = elem.get(_ATTR_CHAIN, ""),
        prev_sentence        = prev,
        current_sentence     = current,
        next_sentence        = nxt,
        source_expressions   = source_exprs,
        target_expressions   = target_exprs,
        source_concept       = source_concept,
        metaphoricity_scores = metaphoricity,
        polarity_labels      = polarity,
        intensity_scores     = intensity,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_lcc_cached(path_str: str) -> list[LccInstance]:
    """Parse an LCC Metaphor XML file and return a cached list of instances."""
    try:
        tree = ET.parse(path_str)
    except FileNotFoundError:
        raise FileNotFoundError(f"LCC Metaphor XML file not found: '{path_str}'")
    except ET.ParseError as e:
        raise ValueError(f"Could not parse LCC XML at '{path_str}': {e}") from e

    root = tree.getroot()
    return [_build_instance(elem) for elem in root.iter(_TAG_LM_INSTANCE)]


def load_lcc(path: str | Path) -> list[LccInstance]:
    """Load an LCC Metaphor XML file and return all instances.

    Each :class:`LccInstance` exposes the source domain expressions
    (``source_expressions``) and target domain expressions
    (``target_expressions``) found within the annotated sentence, along
    with the abstract ``target_concept`` label, surrounding context
    sentences, and metaphoricity / polarity / intensity annotations.

    Results are cached by resolved path so repeated calls are free.
    """
    return _load_lcc_cached(str(Path(path).resolve()))
