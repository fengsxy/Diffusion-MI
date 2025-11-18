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

### Quick Start: Estimate MI on Your Own Data

You can apply any estimator to your own data `(X, Y)` as long as:
- `X` and `Y` are arrays of shape `(n_samples, dim_x)` and
  `(n_samples, dim_y)` respectively (NumPy arrays or similar), and
- The shapes are consistent with the `x_shape` / `y_shape` you pass
  when constructing the estimator.

```python
import numpy as np
import dmi

# Example: a simple 1D correlated Gaussian
n = 2000
dim_x, dim_y = 1, 1
rho = 0.75
rng = np.random.default_rng(0)
x = rng.normal(size=(n, dim_x))
eps = rng.normal(size=(n, dim_y))
y = rho * x + np.sqrt(1.0 - rho ** 2) * eps

estimator = dmi.estimators.MINEEstimator(
    x_shape=(dim_x,),
    y_shape=(dim_y,),
    max_n_steps=500,
    batch_size=256,
)

mi_hat = estimator.estimate(x, y)
print("Estimated MI (MINE):", float(mi_hat))
```

Different estimators have slightly different initialization options,
but the overall pattern is always:
- Provide `x_shape` / `y_shape` (typically `(dim_x,)` / `(dim_y,)`),
- Configure training hyperparameters such as `max_n_steps`,
  `max_epochs`, learning rate, and batch size, and
- Call `.estimate(X, Y)` to obtain an MI estimate.

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
- `MINDEEstimator` — SDE-based MI estimator.
- `MMGEstimator` — Diffusion-based multi-modal MI estimator (from
  diffusion-style MIND variants).

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

The design of some estimators and benchmarks is inspired by existing
papers and open-source projects. If you use this repository in
academic work, please consider citing it as
"Diffusion Mutual Information (DMI)".

