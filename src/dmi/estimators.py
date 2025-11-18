"""Public estimators API for :mod:`dmi`.

This module re-exports the neural mutual information estimators from
``estimators.neural`` under the :mod:`dmi.estimators` namespace so that
users can simply do::

    from dmi import estimators
    mi_estimator = estimators.DIMEEstimator(...)
"""

from __future__ import annotations

try:  # pragma: no cover - import-time wiring
    # The concrete implementations live in the internal
    # ``estimators.neural`` package shipped with this distribution.
    from estimators.neural import *  # type: ignore
except Exception as exc:  # pragma: no cover - defensive fallback
    # If the internal package is unavailable, expose a helpful error
    # message when any estimator is accessed.
    raise ImportError(
        "dmi.estimators requires the internal 'estimators.neural' "
        "package to be importable. Make sure diffusion-mi is installed "
        "correctly."
    ) from exc

