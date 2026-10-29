"""AHCAL stable intervals, trigger rates, and effective luminosity."""

__version__ = "0.1.0"
__all__ = ["StableRunListReader"]


def __getattr__(name):
    if name == "StableRunListReader":
        from .stable_runlist_reader import StableRunListReader
        return StableRunListReader
    raise AttributeError(name)
