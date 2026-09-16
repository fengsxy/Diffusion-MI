"""Public estimator classes, imported only when requested.

Example: ``from dmi.estimators import MMGEstimator``.
"""
from importlib import import_module

_MODULES = {
    "CPCEstimator": "CPC",
    "DoEEstimator": "DOE",
    "MINEEstimator": "MINE",
    "NWJEstimator": "NWJ",
    "SMILEEstimator": "SMILE",
    "DIMEEstimator": "DIME",
    "MINDEEstimator": "MINDE",
    "MMGEstimator": "MMG",
}
__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    cls = getattr(import_module(f"estimators.neural.{_MODULES[name]}"), name)
    globals()[name] = cls
    return cls


def __dir__():
    return sorted(set(globals()) | set(__all__))
