import os
import sys
import numpy as np


# Ensure the project root (containing the `src` package) is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class _SimpleGaussianTask:
    """Simple synthetic correlated Gaussian task used for testing.

    This avoids importing bmi / JAX while providing a task-like API
    with `.sample`, `.dim_x`, `.dim_y`, and `.mutual_information`.
    """

    def __init__(self, dim_x=1, dim_y=1, rho=0.75):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.rho = rho
        # Mutual information of a 1D correlated Gaussian (in nats).
        self.mutual_information = 0.5 * np.log(1.0 / (1.0 - rho ** 2))
        self.name = f"{dim_x}v{dim_y}-gaussian-{rho}"

    def sample(self, n, seed=None):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, self.dim_x))
        eps = rng.normal(size=(n, self.dim_y))
        y = self.rho * x + np.sqrt(1.0 - self.rho ** 2) * eps
        return x, y


def _standardize_train_test(x_train, y_train, x_test, y_test):
    """Simple standardization helper used in tests.

    We mimic the preprocessing used in the MIND main script
    (standardizing X and Y separately on the training set).
    """

    from sklearn.preprocessing import StandardScaler

    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)

    x_train_s = x_scaler.transform(x_train)
    y_train_s = y_scaler.transform(y_train)
    x_test_s = x_scaler.transform(x_test)
    y_test_s = y_scaler.transform(y_test)

    return x_train_s, y_train_s, x_test_s, y_test_s


def test_mind_estimator_smoke_on_bmi_gaussian():
    """Smoke test: MMGEstimator (formerly MIND) runs on a simple task.

    This test is intentionally light-weight: it only checks that
    - we can construct an MMGEstimator,
    - run a very short training via `.fit`, and
    - call `.estimate` to obtain a finite MI estimate.

    It does *not* aim to verify high-accuracy MI estimation, to keep
    runtime and flakiness under control.
    """

    # Import the diffusion-based estimator defined in MMG.py
    from src.estimators.neural.MMG import MMGEstimator

    # Use a simple 1D correlated Gaussian synthetic task with known MI.
    task = _SimpleGaussianTask(dim_x=1, dim_y=1, rho=0.75)
    task_name = task.name

    # Use relatively small sample sizes and epochs so the test
    # stays quick, while still exercising the full training pipeline.
    train_samples = 2000
    val_samples = 500

    x_train, y_train = task.sample(train_samples, seed=0)
    x_val, y_val = task.sample(val_samples, seed=1)

    # Match the preprocessing used in the MIND script: rescale X and Y.
    x_train_s, y_train_s, x_val_s, y_val_s = _standardize_train_test(
        x_train, y_train, x_val, y_val
    )

    # Construct a small MMGEstimator. We keep max_epochs low so that
    # the test is fast; this is a functional smoke test, not a benchmark.
    model = MMGEstimator(
        x_shape=(task.dim_x,),
        y_shape=(task.dim_y,),
        learning_rate=1e-3,
        batch_size=128,
        max_epochs=2,
        seed=0,
        task_name=task_name,
        task_gt=task.mutual_information,
        mi_estimation_interval=200,
        use_ema=False,
    )

    # Fit on the small training set, validate on the held-out set.
    model.fit(x_train_s, y_train_s, x_val_s, y_val_s)

    # Estimate MI on the validation set.
    mi_estimate, _ = model.estimate(x_val_s, y_val_s)

    # Basic sanity checks: value should be finite and not explode.
    assert np.isfinite(mi_estimate), "MI estimate should be finite"

    # Ground-truth MI for this synthetic task (in nats).
    mi_gt = task.mutual_information
    abs_err = abs(mi_estimate - mi_gt)

    # With very few epochs the estimate can be rough, so we only
    # require it to be non-negative, not astronomically large, and
    # within a loose band of the ground truth (~0.41 nats).
    assert mi_estimate >= 0.0, "MI estimate should be non-negative"
    assert mi_estimate < 10.0, "MI estimate is unreasonably large"
    assert abs_err < 2.0, (
        f"MIND.MINDEstimator: MI estimate too far from ground truth. "
        f"estimate={mi_estimate:.3f}, gt={mi_gt:.3f}, abs_err={abs_err:.3f}"
    )
