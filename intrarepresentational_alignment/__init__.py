from __future__ import annotations


def __getattr__(name: str):
    if name in ("EmbeddingModel",):
        from .models import EmbeddingModel
        globals()["EmbeddingModel"] = EmbeddingModel
        return globals()[name]
    if name in ("Embedder",):
        from .embedding import Embedder
        globals()["Embedder"] = Embedder
        return globals()[name]
    if name in ("SparsificationStrategy", "KNN", "EpsilonThreshold", "SparseGraph"):
        from .graph import SparsificationStrategy, KNN, EpsilonThreshold, SparseGraph
        globals().update(dict(
            SparsificationStrategy=SparsificationStrategy,
            KNN=KNN,
            EpsilonThreshold=EpsilonThreshold,
            SparseGraph=SparseGraph,
        ))
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
