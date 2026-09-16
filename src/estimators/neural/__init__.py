"""Lazy exports; importing one estimator does not load every backend."""
from importlib import import_module

_ESTIMATORS = {
    "CPCEstimator": "CPC", "DoEEstimator": "DOE", "MINEEstimator": "MINE",
    "NWJEstimator": "NWJ", "SMILEEstimator": "SMILE", "DIMEEstimator": "DIME",
    "MINDEEstimator": "MINDE", "MMGEstimator": "MMG",
}
_LEGACY = {
    **{name: "_critic" for name in ("MLP", "ConvCritic", "ConcatCritic", "SeparableCritic", "Discriminator", "CombinedArchitecture", "ConvolutionalCritic", "UnetMLP")},
    "sample_vp_truncated_q": "libs.importance", "get_normalizing_constant": "libs.importance",
    "VP_SDE": "libs.SDE", "EMA": "libs.util", "concat_vect": "libs.util", "deconcat": "libs.util",
}
__all__ = list(_ESTIMATORS) + list(_LEGACY)


def __getattr__(name):
    module = {**_ESTIMATORS, **_LEGACY}.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value
