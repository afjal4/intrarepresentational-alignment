from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import requests

_DEFAULT_CACHE_PATH = Path("data/conceptnet_cache.json")
_RATE_LIMIT_DELAY = 1.0
_SAVE_EVERY = 50
_API_URL = "https://api.conceptnet.io/relatedness"
_MIRROR_BASE = "https://cstr-conceptnet-normalized.hf.space/gradio_api/call/run_raw_query"


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


def _query_mirror(word1: str, word2: str) -> float:
    """Query the HuggingFace ConceptNet mirror (Gradio 5 SSE API).

    Returns the max edge weight between word1 and word2 in ConceptNet,
    or 0.0 if no direct edge exists.  Note: weights are raw ConceptNet
    values (typically 0–10) rather than the 0–1 relatedness scores from
    the official API.
    """
    # Sanitise to alphanumeric + underscore only (avoids SQL injection)
    w1 = re.sub(r"[^a-z0-9_]", "", word1.lower().replace(" ", "_"))
    w2 = re.sub(r"[^a-z0-9_]", "", word2.lower().replace(" ", "_"))
    sql = (
        "SELECT MAX(e.weight) FROM edge_norm e "
        "JOIN node_norm n_start ON e.start_fk = n_start.node_pk "
        "JOIN node_norm n_end ON e.end_fk = n_end.node_pk "
        f"WHERE (n_start.node_url LIKE '%/c/en/{w1}' AND n_end.node_url LIKE '%/c/en/{w2}') "
        f"OR (n_start.node_url LIKE '%/c/en/{w2}' AND n_end.node_url LIKE '%/c/en/{w1}')"
    )

    # Step 1: submit job, get event_id
    submit = requests.post(_MIRROR_BASE, json={"data": [sql]}, timeout=30)
    submit.raise_for_status()
    event_id = submit.json()["event_id"]

    # Step 2: stream SSE response until event: complete
    with requests.get(f"{_MIRROR_BASE}/{event_id}", stream=True, timeout=120) as stream:
        stream.raise_for_status()
        last_data: str | None = None
        for line in stream.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                last_data = line[5:].strip()

    if last_data is None:
        return 0.0

    # payload = [{"headers": [...], "data": [[val]], ...}, "message string"]
    payload = json.loads(last_data)
    df = payload[0]
    if isinstance(df, dict):
        rows = df.get("data", [[None]])
        val = rows[0][0] if rows and rows[0] else None
    else:
        val = None
    return float(val) if val is not None else 0.0


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

    # Try official API first; fall back to HuggingFace mirror
    try:
        resp = requests.get(
            _API_URL,
            params={"node1": _cn_node(word1), "node2": _cn_node(word2)},
            timeout=10,
        )
        resp.raise_for_status()
        score = float(resp.json()["value"])
        time.sleep(_RATE_LIMIT_DELAY)
    except Exception:
        score = _query_mirror(word1, word2)  # raises if mirror also fails

    cache[key] = score
    new_call_count[0] += 1

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
