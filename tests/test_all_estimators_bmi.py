import os
import sys
import numpy as np
import pytest


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


# Global test configuration for all estimators
TEST_CONFIG = {
    "train_samples": 1000,
    "test_samples": 1000,
    "simple": {
        "batch_size": 256,
        # Slightly larger training budget so that
        # NWJ / CPC / SMILE are stable enough for
        # the tight accuracy tolerance used below.
        "max_n_steps": 800,
        "learning_rate": 1e-4,
    },
    "mine": {
        "batch_size": 256,
        "max_n_steps": 500,
        "max_epochs": 1,
        "learning_rate": 1e-4,
        "hidden_layers": (50, 50),
    },
    "dime": {
        "batch_size": 256,
        "max_n_steps": 500,
        "max_epochs": None,
        "learning_rate": 1e-3,
    },
    "minde_sde": {
        "batch_size": 64,
        "max_n_steps": 500,
        "learning_rate": 5e-5,
        "max_epochs": 1,
    },
    "mind_diff": {
        "batch_size": 256,
        "max_epochs": 2,
        "learning_rate": 1e-3,
    },
    # Acceptable absolute error to ground truth (~0.41 nats)
    "mi_abs_tol": 0.1,
    # Some estimators (especially diffusion-based or SDE-based ones)
    # can be much noisier under the very small training budgets used
    # in this smoke test. For those we allow a looser tolerance.
    "mi_abs_tol_per_estimator": {
        "MINE": 0.5,
        "MINDE_SDE": 0.5,
        "MIND_diff": 0.5,
    },
}


@pytest.mark.parametrize(
    "estimator_name, estimator_ctor, kind",
    [
        ("CPC", "CPCEstimator", "simple"),
        ("DoE", "DoEEstimator", "simple"),
        ("NWJ", "NWJEstimator", "simple"),
        ("SMILE", "SMILEEstimator", "simple"),
        ("MINE", "MINEEstimator", "mine"),
        ("DIME", "DIMEEstimator", "dime"),
        ("MINDE_SDE", "MINDEEstimator", "minde_sde"),
        ("MIND_diff", "MINDDiffEstimator", "mind_diff"),
    ],
)
def test_all_estimators_smoke_on_bmi_gaussian(estimator_name, estimator_ctor, kind):
    """Smoke test: all MI estimators run end-to-end on a simple MI task.

    For each estimator type we:
    - construct it with very small training budgets (steps/epochs),
    - train on a small sample from a simple correlated Gaussian task,
    - call ``estimate`` on a held-out sample,
    - check that the returned MI is finite and within a reasonable
      distance of the known ground truth (≈0.41 nats when rho=0.75).

    The tolerance is deliberately loose to avoid flakiness: this is a
    functional test, not a benchmark of estimator accuracy.
    """

    from src.estimators.neural.CPC import CPCEstimator
    from src.estimators.neural.DOE import DoEEstimator
    from src.estimators.neural.NWJ import NWJEstimator
    from src.estimators.neural.SMILE import SMILEEstimator
    from src.estimators.neural.MINE import MINEEstimator
    from src.estimators.neural.DIME import DIMEEstimator
    from src.estimators.neural.MINDE import MINDEEstimator as MINDE_SDEEstimator
    from src.estimators.neural.MMG import MMGEstimator as MINDDiffEstimator

    ctor_map = {
        "CPCEstimator": CPCEstimator,
        "DoEEstimator": DoEEstimator,
        "NWJEstimator": NWJEstimator,
        "SMILEEstimator": SMILEEstimator,
        "MINEEstimator": MINEEstimator,
        "DIMEEstimator": DIMEEstimator,
        "MINDEEstimator": MINDE_SDEEstimator,
        "MINDDiffEstimator": MINDDiffEstimator,
    }

    estimator_class = ctor_map[estimator_ctor]

    # Simple 1D correlated Gaussian synthetic task with known MI.
    task = _SimpleGaussianTask(dim_x=1, dim_y=1, rho=0.75)

    # Sample sizes (kept small for test speed).
    x_train, y_train = task.sample(TEST_CONFIG["train_samples"], seed=0)
    x_test, y_test = task.sample(TEST_CONFIG["test_samples"], seed=1)

    if kind == "simple":
        # Estimators with a simple (X, Y) -> fit/estimate API
        cfg = TEST_CONFIG["simple"]
        estimator = estimator_class(
            batch_size=cfg["batch_size"],
            max_n_steps=cfg["max_n_steps"],
            learning_rate=cfg["learning_rate"],
        )
        estimator.fit(x_train, y_train)
        mi_estimate = estimator.estimate(x_test, y_test)

    elif kind == "mine":
        # Lightning-based MINE estimator
        cfg = TEST_CONFIG["mine"]
        estimator = estimator_class(
            x_shape=(task.dim_x,),
            y_shape=(task.dim_y,),
            learning_rate=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            max_n_steps=cfg["max_n_steps"],
            max_epochs=cfg["max_epochs"],
            hidden_layers=cfg["hidden_layers"],
            seed=0,
            task_name=f"{estimator_name}_test",
            task_gt=task.mutual_information,
            test_num=200,
            early_stopping=False,
            create_checkpoint=False,
        )
        estimator.fit(x_train, y_train)
        mi_estimate = estimator.estimate(x_test, y_test)

    elif kind == "dime":
        # Lightning-based DIME estimator
        cfg = TEST_CONFIG["dime"]
        estimator = estimator_class(
            x_shape=(task.dim_x,),
            y_shape=(task.dim_y,),
            learning_rate=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            max_n_steps=cfg["max_n_steps"],
            max_epochs=cfg["max_epochs"],
            divergence="GAN",
            architecture="separable",
            alpha=1,
            seed=0,
            task_name=f"{estimator_name}_test",
            task_gt=task.mutual_information,
            test_num=200,
            early_stopping=False,
            create_checkpoint=False,
        )
        estimator.fit(x_train, y_train)
        mi_estimate = estimator.estimate(x_test, y_test)

    elif kind == "minde_sde":
        # SDE-based MINDE estimator (MINDE.py)
        cfg = TEST_CONFIG["minde_sde"]
        estimator = estimator_class(
            x_shape=(task.dim_x,),
            y_shape=(task.dim_y,),
            learning_rate=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            max_n_steps=cfg["max_n_steps"],
            max_epochs=cfg["max_epochs"],
            smoothing_alpha=0.01,
            seed=0,
            task_name=f"{estimator_name}_test",
            task_gt=task.mutual_information,
            test_num=200,
            mi_estimation_interval=200,
            use_ema=False,
            arch="mlp",
            type="c",
            mc_iter=3,
            preprocessing="rescale",
            importance_sampling=True,
            early_stopping=False,
            create_checkpoint=False,
        )
        estimator.fit(x_train, y_train, x_test, y_test)
        mi_estimate = estimator.estimate(x_test, y_test)

    elif kind == "mind_diff":
        # Diffusion-based MIND estimator (MIND.py)
        from sklearn.preprocessing import StandardScaler

        cfg = TEST_CONFIG["mind_diff"]

        # Apply the same standardization scheme as used in the MIND
        # script: standardize X and Y separately on the training set.
        x_scaler = StandardScaler().fit(x_train)
        y_scaler = StandardScaler().fit(y_train)
        x_train_s = x_scaler.transform(x_train)
        y_train_s = y_scaler.transform(y_train)
        x_test_s = x_scaler.transform(x_test)
        y_test_s = y_scaler.transform(y_test)

        estimator = estimator_class(
            x_shape=(task.dim_x,),
            y_shape=(task.dim_y,),
            learning_rate=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            max_epochs=cfg["max_epochs"],
            seed=0,
            task_name=f"{estimator_name}_test",
            task_gt=task.mutual_information,
            mi_estimation_interval=200,
            use_ema=False,
            create_checkpoint=False,
        )
        estimator.fit(x_train_s, y_train_s, x_test_s, y_test_s)
        mi_estimate = estimator.estimate(x_test_s, y_test_s)

    else:
        raise ValueError(f"Unknown estimator kind: {kind}")

    # Ground-truth MI for this synthetic task (in nats).
    mi_gt = task.mutual_information
    abs_err = abs(mi_estimate - mi_gt)

    # Basic sanity checks that should hold for all estimators.
    assert np.isfinite(mi_estimate), f"{estimator_name}: MI estimate should be finite"
    # Expect non-pathological values even after very few steps.
    assert mi_estimate > -1.0, f"{estimator_name}: MI estimate too negative"
    assert mi_estimate < 20.0, f"{estimator_name}: MI estimate unreasonably large"

    # Accuracy check against the known ground truth (~0.41 nats).
    # Different estimators have different biases; we use a configurable
    # absolute tolerance and allow a looser bound for some of the more
    # complex, diffusion-based estimators under these tiny budgets.
    per_est_tol = TEST_CONFIG.get("mi_abs_tol_per_estimator", {})
    tol = per_est_tol.get(estimator_name, TEST_CONFIG["mi_abs_tol"])

    assert abs_err < tol, (
        f"{estimator_name}: MI estimate too far from ground truth. "
        f"estimate={mi_estimate:.3f}, gt={mi_gt:.3f}, abs_err={abs_err:.3f}"
    )
