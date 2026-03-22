from .lcc import load_lcc
from .metanet import (
    LocalID,
    MetaphorEntry,
    Frame,
    Mapping,
    Binding,
    Example,
    LexicalUnit,
    LccInstance,
)


def __getattr__(name: str):
    if name in ("load_metanet", "MetaNetRepository"):
        from .extractors import load_metanet, MetaNetRepository
        globals()["load_metanet"] = load_metanet
        globals()["MetaNetRepository"] = MetaNetRepository
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
