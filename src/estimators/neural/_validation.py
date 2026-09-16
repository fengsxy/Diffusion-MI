"""Shared input checks for the fit/estimate interface."""
import numpy as np


def validate_pair(X, Y):
    """Return finite float32 arrays with paired rows and feature axes."""
    arrays = []
    for name, value in (("X", X), ("Y", Y)):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.ndim < 2 or any(size == 0 for size in value.shape[1:]):
            raise ValueError(f"{name} must have shape (n_samples, n_features); use reshape(-1, 1) for a scalar variable.")
        if value.dtype.kind not in "biuf":
            raise ValueError(f"{name} must contain real numeric values.")
        with np.errstate(over="ignore", invalid="ignore"):
            value = value.astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite float32 values (no NaN or infinity).")
        arrays.append(value)
    X, Y = arrays
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same number of paired samples.")
    if len(X) < 2:
        raise ValueError("At least two paired samples are required.")
    return X, Y


def validate_inputs(estimator, X, Y, *, fitting=False):
    X, Y = validate_pair(X, Y)
    params = getattr(estimator, "hparams", estimator)
    for name, array in (("x_shape", X), ("y_shape", Y)):
        expected = getattr(params, name, None)
        if expected is not None and tuple(expected) != array.shape[1:]:
            raise ValueError(f"{name}: expected {tuple(expected)}, got {array.shape[1:]}.")
    shapes = (X.shape[1:], Y.shape[1:])
    if hasattr(estimator, "_input_shapes") and shapes != estimator._input_shapes:
        raise ValueError(f"Feature shapes must match training data: expected {estimator._input_shapes}, got {shapes}.")
    if fitting:
        for key in ("batch_size", "max_n_steps", "max_epochs"):
            value = getattr(params, key, None)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < (2 if key == "batch_size" else 1)):
                raise ValueError(f"{key} must be an integer >= {2 if key == 'batch_size' else 1}.")
        estimator._input_shapes = shapes
    elif not getattr(estimator, "_is_fitted", False):
        raise RuntimeError("Call fit(X, Y) before estimate(X, Y).")
    return X, Y


def validate_validation(X_val, Y_val, X, Y):
    if (X_val is None) != (Y_val is None):
        raise ValueError("Provide both X_val and Y_val, or neither.")
    if X_val is not None:
        X_val, Y_val = validate_pair(X_val, Y_val)
        if X_val.shape[1:] != X.shape[1:] or Y_val.shape[1:] != Y.shape[1:]:
            raise ValueError("Validation feature shapes must match training data.")
    return X_val, Y_val


def training_limits(max_steps, max_epochs):
    """Finite default; an explicit step budget must not get an epoch cap."""
    if max_steps is None and max_epochs is None:
        return {"max_steps": 1000, "max_epochs": -1}
    return {"max_steps": max_steps if max_steps is not None else -1,
            "max_epochs": max_epochs if max_epochs is not None else -1}
