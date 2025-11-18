## Diffusion Mutual Information (DMI)

`diffusion-mi` (short for **Diffusion Mutual Information**, DMI) is a
PyTorch-based toolbox for estimating mutual information (MI), with a
focus on diffusion-style and neural estimators.

It provides:
- A collection of neural MI estimators (DIME, MINE, NWJ, SMILE, MINDE,
  MMG, etc.) exposed under the `dmi.estimators` namespace.
- Lightweight synthetic benchmarks (e.g. correlated Gaussians and
  self-consistency-style image tasks) to sanity-check estimator
  behavior.

The core estimators are implemented in PyTorch / PyTorch Lightning and
are intended for both research experiments and practical use.

---

### Installation

#### 1. From PyPI

The recommended way to install DMI is:

```bash
pip install diffusion-mi
```

This installs:
- The `diffusion-mi` package (imported as `dmi`), and
- Required dependencies such as `torch` and `pytorch-lightning`.

#### 2. From source

From the repository root:

```bash
# Using poetry (recommended for development)
poetry install

# Or using pip
pip install -e .
```

After installation you can import the package via:

```python
import dmi
```

---

### Quick Start: Fit, Then Estimate MI on Your Own Data

You can apply any estimator to your own data `(X, Y)` as long as:
- `X` and `Y` are arrays of shape `(n_samples, dim_x)` and
  `(n_samples, dim_y)` respectively (NumPy arrays or similar), and
- Their shapes are compatible with the estimator you choose.

The typical workflow is always:

1. **Construct** an estimator with your desired hyperparameters.
2. **Fit** it on training data via `.fit(X_train, Y_train, X_val=None, Y_val=None)`.
3. **Estimate** MI on new data via `.estimate(X_test, Y_test)`.

```python
import numpy as np
import dmi

# Example: a simple 1D correlated Gaussian
n_train, n_test = 2000, 1000
dim_x, dim_y = 1, 1
rho = 0.75
rng = np.random.default_rng(0)

def sample_gaussian(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim_x))
    eps = rng.normal(size=(n, dim_y))
    y = rho * x + np.sqrt(1.0 - rho ** 2) * eps
    return x, y

X_train, Y_train = sample_gaussian(n_train, seed=0)
X_test, Y_test = sample_gaussian(n_test, seed=1)

# Construct a neural MI estimator
estimator = dmi.estimators.MINEEstimator(
    learning_rate=1e-4,
    batch_size=256,
    max_n_steps=3000,
)

# 1) Fit on training data
estimator.fit(X_train, Y_train)

# 2) Estimate MI on held-out data
mi_hat = estimator.estimate(X_test, Y_test)
print("Estimated MI (MINE):", float(mi_hat))
```

Notes on shapes and initialization:

- Most estimators (CPC/DoE/NWJ/SMILE/MINE/DIME) can **infer**
  `x_shape` / `y_shape` automatically from the first call to
  `.fit(X, Y)`.
- Diffusion/SDE-based estimators (`MINDEEstimator`, `MMGEstimator`)
  also support lazy initialization: if you omit `x_shape` / `y_shape`
  in the constructor, they will infer the shapes at the first
  `.fit(X, Y, ...)` call and build their internal U-Net architectures
  accordingly.

In all cases, `.estimate(X, Y)` performs **no further training**: it
uses the current model parameters to compute an MI estimate on the
provided data.

---

### Benchmarks and Synthetic Tasks

This repository includes a few lightweight benchmark tasks that are
useful for sanity-checking estimator behavior. They are not yet
exposed as a stable public API, but can be imported directly from the
source tree for experiments.

- **Correlated Gaussian (stair benchmark)**
  - File: `src/benchmark/stair/stair_benchmark.py`
  - Class: `CorrelatedGaussianTask`
  - Generates pairs `(X, Y)` of correlated Gaussians with known
    ground-truth MI:

    ```python
    from benchmark.stair.stair_benchmark import CorrelatedGaussianTask

    task = CorrelatedGaussianTask(name="gaussian-20d-0.5", dim=20, rho=0.5)
    X, Y = task.sample(1000, seed=0)

    print("Ground truth MI:", task.mutual_information)
    ```

- **Self-consistency image tasks**
  - Files: under `src/benchmark/self-consistency/`
  - Provide simple image-based tasks (MNIST/Fashion-MNIST/CIFAR10) in
    which MI should satisfy known monotonicity / additivity patterns
    when only parts of the image are revealed.

These benchmarks are primarily intended for internal evaluation and
may change; the core public surface of the package is
`dmi.estimators`.

---

### Available Neural MI Estimators

The main estimators exposed via `dmi.estimators` include:

- `CPCEstimator` — Contrastive Predictive Coding (CPC)-style
  estimator.
- `DoEEstimator` — Density-of-States style estimator.
- `MINEEstimator` — Mutual Information Neural Estimator.
- `NWJEstimator` — NWJ lower bound (an f-divergence based estimator).
- `SMILEEstimator` — A stabilized MI lower-bound estimator.
- `DIMEEstimator` — Diffusion-based MI estimator.
- `MINDEEstimator` — SDE-based MI estimator, based on the MINDE
  method (see arXiv:2310.09031).
- `MMGEstimator` — Diffusion-based multi-modal MI estimator, based on
  the MMG method (see arXiv:2509.20609).

You can inspect the available estimators, for example, with:

```python
import dmi
dir(dmi.estimators)
```

or import them directly:

```python
from dmi import estimators
est = estimators.DIMEEstimator(...)
```

---

### Environment and Dependencies

Typical requirements are:

- Python >= 3.9
- PyTorch (`torch`)
- PyTorch Lightning (`pytorch-lightning` / `lightning`)

These are installed automatically when using `pip install diffusion-mi`.

---

### Citation and Acknowledgements

The design of the estimators in this repository is inspired by
several existing mutual-information estimators and diffusion-based
methods. If you use this repository in academic work, please consider
citing it as "Diffusion Mutual Information (DMI)" and, where
appropriate, the original works listed below.

**Estimator References (non-exhaustive)**

- `CPCEstimator` — Contrastive Predictive Coding (CPC) estimator,
  based on:
  - A. van den Oord, Y. Li, O. Vinyals, "Representation Learning with
    Contrastive Predictive Coding", arXiv:1807.03748.

- `MINEEstimator` — Mutual Information Neural Estimation, based on:
  - M. I. Belghazi et al., "Mutual Information Neural Estimation",
    ICML 2018, arXiv:1801.04062.

- `NWJEstimator` — NWJ lower-bound estimator, using the Nguyen–
  Wainwright–Jordan f-divergence bound:
  - X. Nguyen, M. J. Wainwright, M. I. Jordan, "Estimating divergence
    functionals and the likelihood ratio by convex risk minimization",
    Annals of Statistics, 2010.

- `SMILEEstimator` — Implementation of the SMILE mutual information
  estimator; see the original SMILE MI paper for details.

- `DoEEstimator` — Density-of-States style estimator; this
  implementation follows the DoE estimator found in MI benchmarking
  codebases (e.g. `benchmark-mi`) and related literature.

- `DIMEEstimator` — Diffusion-based MI estimator; this implementation
  follows the DIME-style neural estimator from the diffusion MI
  literature.

- `MINDEEstimator` — SDE-based MI estimator, based on the MINDE
  method:
  - MINDE: see arXiv:2310.09031.

- `MMGEstimator` — Diffusion-based multi-modal MI estimator, based on
  the MMG method:
  - MMG: see arXiv:2509.20609.
