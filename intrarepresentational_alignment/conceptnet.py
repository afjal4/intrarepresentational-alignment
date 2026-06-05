from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import requests

_DEFAULT_CACHE_PATH = Path("data/conceptnet_cache.json")
_RATE_LIMIT_DELAY = 0.5
_SAVE_EVERY = 50
_API_URL = "https://api.conceptnet.io/relatedness"


def _cn_node(word: str) -> str:
    return f"/c/en/{word.lower().replace(' ', '_')}"


def _cache_key(word1: str, word2: str) -> str:
    a, b = sorted([word1.lower(), word2.lower()])
    return f"{a}|{b}"


def _load_cache(cache_path: Path) -> dict[str, float]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict[str, float], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _fetch_relatedness(
    word1: str,
    word2: str,
    cache: dict[str, float],
    cache_path: Path,
    new_call_count: list[int],
) -> float:
    key = _cache_key(word1, word2)
    if key in cache:
        return cache[key]

    try:
        resp = requests.get(
            _API_URL,
            params={"node1": _cn_node(word1), "node2": _cn_node(word2)},
            timeout=15,
        )
        score = float(resp.json()["value"]) if resp.ok else 0.0
    except Exception:
        score = 0.0

    cache[key] = score
    new_call_count[0] += 1
    time.sleep(_RATE_LIMIT_DELAY)

    if new_call_count[0] % _SAVE_EVERY == 0:
        _save_cache(cache, cache_path)

    return score


def build_conceptnet_kernel(
    words: list[str],
    cache_path: Path = _DEFAULT_CACHE_PATH,
) -> np.ndarray:
    """Build an NxN symmetric kernel from ConceptNet relatedness scores.

    Entry (i, j) is the relatedness score between words[i] and words[j].
    Missing or unknown pairs get score 0.0. The diagonal is 0.0.
    Results are fetched from the ConceptNet REST API with a persistent
    disk cache at *cache_path* to avoid redundant calls.
    """
    n = len(words)
    K = np.zeros((n, n), dtype=float)
    cache = _load_cache(cache_path)
    new_call_count = [0]

    for i in range(n):
        for j in range(i + 1, n):
            score = _fetch_relatedness(
                words[i], words[j], cache, cache_path, new_call_count
            )
            K[i, j] = score
            K[j, i] = score

    if new_call_count[0] > 0:
        _save_cache(cache, cache_path)

    return K
