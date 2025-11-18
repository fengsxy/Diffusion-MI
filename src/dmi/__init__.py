"""Top-level package for Diffusion Mutual Information (DMI).

This package provides a convenient namespace for neural mutual
information estimators under :mod:`dmi.estimators`.

The underlying implementations are provided by the internal
``estimators.neural`` package shipped with this distribution.
"""

from __future__ import annotations

from . import estimators

__all__ = ["estimators"]
