from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

try:
    from rdflib import Graph

    _RDFLIB_AVAILABLE = True
    _rdflib_import_error: Exception | None = None
except ImportError as e:
    _RDFLIB_AVAILABLE = False
    _rdflib_import_error = e

from ..repositories.metanet_repository import MetaNetRepository

MAX_RDF_BYTES = 100 * 1024 * 1024


def _sanitize_rdf_xml(content: str) -> str:
    return re.sub(
        r'(xmlns:\w+="[^"]*")',
        lambda m: m.group(1).replace(" ", "%20"),
        content,
    )


@lru_cache(maxsize=None)
def _load_metanet_cached(path_str: str) -> MetaNetRepository:
    if not _RDFLIB_AVAILABLE:
        raise ImportError("load_metanet requires rdflib: pip install rdflib") from _rdflib_import_error

    from .rdf_extractors import default_extractor_registry

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"MetaNet RDF file not found: '{path_str}'")
    if path.stat().st_size > MAX_RDF_BYTES:
        raise ValueError(f"MetaNet RDF file exceeds size limit: '{path_str}'")

    g = Graph()
    try:
        raw = path.read_text(encoding="utf-8")
        g.parse(data=_sanitize_rdf_xml(raw), format="xml")
    except Exception as e:
        raise ValueError(f"Could not parse MetaNet RDF at '{path_str}': {e}") from e

    return MetaNetRepository(g, default_extractor_registry())


def load_metanet(path: str | Path) -> MetaNetRepository:
    return _load_metanet_cached(str(Path(path).resolve()))
