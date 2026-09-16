<p align="center">
  <img src="docs/assets/diffusion-mi-logo.png" width="600" alt="Diffusion-MI logo">
</p>

<h1 align="center">Diffusion-MI</h1>
<p align="center"><strong>Mutual information estimation. One fit/estimate interface.</strong></p>
<p align="center">
  <a href="https://pypi.org/project/diffusion-mi/">PyPI</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#choose-an-estimator">Estimators</a> ·
  <a href="https://arxiv.org/abs/2509.20609">MMG paper</a> ·
  <a href="https://github.com/fengsxy/Diffusion-MI/issues">Issues</a>
</p>

Estimate how much information two continuous random variables share, directly from paired samples. **Diffusion-MI** brings diffusion estimators and neural baselines behind the same Python interface, so you can change methods without rewriting your data pipeline.

- **Eight estimators:** MMG, MINDE, MINE, NWJ, SMILE, CPC, DoE, and DIME.
- **Your arrays in, MI in nats out:** `fit(X_train, Y_train)` → `estimate(X_test, Y_test)`.
- **Independent feature dimensions:** X and Y need paired rows, not identical widths.
- **PyTorch on CPU or CUDA:** no dataset downloads or external experiment accounts needed for the quick start.

**Names:** Diffusion-MI is the toolbox; **MMG is a method within it**. Install `diffusion-mi`, import `dmi`.

## Install

Requires Python **3.9–3.12**. In a virtual environment:

```bash
python -m pip install diffusion-mi
```

For the source version described by this README, including unreleased fixes:

```bash
git clone https://github.com/fengsxy/Diffusion-MI.git
cd Diffusion-MI
python -m pip install -e .
```

PyTorch and the core scientific dependencies are installed automatically. For a CPU-only environment, install CPU PyTorch first:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

Image benchmark scripts have optional dependencies: `python -m pip install -e '.[images]'`.

## Quick start

A complete example with separate training and test data, no downloads, and a known ground truth:

```python
import numpy as np
from dmi.estimators import MINEEstimator


def sample(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    y = 0.75 * x + np.sqrt(1 - 0.75**2) * rng.normal(size=(n, 1))
    return x, y


X_train, Y_train = sample(2000, seed=0)
X_test, Y_test = sample(1000, seed=1)

estimator = MINEEstimator(
    max_n_steps=500, batch_size=128, learning_rate=1e-3,
    seed=42, early_stopping=False,
)
estimator.fit(X_train, Y_train)
mi = estimator.estimate(X_test, Y_test)

print(f"Estimated: {mi:.3f} nats")
print(f"Analytic:  {-0.5 * np.log(1 - 0.75**2):.3f} nats")  # 0.413
```

The estimate is stochastic and will not exactly equal the analytic value. This is an onboarding example, not a benchmark claim. Run the same example from the repository with `python examples/gaussian.py`.

### Try MMG with the same interface

```python
from sklearn.preprocessing import StandardScaler
from dmi.estimators import MMGEstimator

# Fit preprocessing only on training data; reuse it for held-out data.
sx, sy = StandardScaler().fit(X_train), StandardScaler().fit(Y_train)
mmg = MMGEstimator(max_n_steps=1000, batch_size=256, seed=42)
mmg.fit(sx.transform(X_train), sy.transform(Y_train))
mi_mmg = mmg.estimate(sx.transform(X_test), sy.transform(Y_test))
```

MMG supports optional validation data via `fit(X, Y, X_val, Y_val)`. MINDE standardizes using training statistics by default; MMG expects you to handle preprocessing. Diffusion estimators may need larger training and Monte Carlo budgets for useful accuracy.

## Choose an estimator

These descriptions refer to the implementations in this repository; a common API does not imply identical objectives, hyperparameters, or statistical guarantees.

| Class in `dmi.estimators` | Approach | Practical consideration |
| --- | --- | --- |
| `MMGEstimator` | Diffusion / conditional–unconditional MMSE gap | This project's MMG method; standardize training and held-out data consistently. |
| `MINDEEstimator` | Score-based diffusion / SDE | Built-in standardization; `mc_iter` controls Monte Carlo averaging. |
| `MINEEstimator` | Neural Donsker–Varadhan objective | A compact starting point for checking your data pipeline. |
| `NWJEstimator` | Variational f-divergence lower bound | Exponentials can produce high variance. |
| `SMILEEstimator` | Clipped density-ratio objective | Clipping trades bias against variance. |
| `CPCEstimator` | Contrastive / InfoNCE objective | Bound depends on negative count; pairwise scoring scales quadratically. |
| `DoEEstimator` | Difference of marginal and conditional entropy estimates | Parametric density assumptions matter. |
| `DIMEEstimator` | Discriminator-based density-ratio estimation | Supports GAN/KL/HD objectives; default pairwise scoring is quadratic. |

## API and data contract

```python
from dmi import estimators

model = estimators.NWJEstimator(max_n_steps=500, batch_size=128)
model.fit(X_train, Y_train)               # returns model
mi_nats = model.estimate(X_test, Y_test)  # returns a float; no training
mi_bits = mi_nats / np.log(2)
```

- The shared, tested path is **continuous vector data**: finite arrays of shape `(n_samples, dim_x)` and `(n_samples, dim_y)`. Convert a scalar variable with `x.reshape(-1, 1)`.
- Row `i` of X and row `i` of Y must describe the same observation. Keep feature shapes fixed after fitting. At least two rows are required; very small datasets are suitable only for smoke tests.
- `batch_size >= 2`; `max_n_steps` is the optimizer-step budget. For Lightning estimators, if both step and epoch limits are supplied, training stops at the first limit. With neither, the default is 1,000 steps.
- `estimate()` requires a successful `fit()` and never updates learned weights. Sampling can make repeated calls return different estimates.
- For plain PyTorch estimators, use `device="cpu"` or `device="cuda"`. Lightning estimators select one CUDA device when available, otherwise CPU. Device controls are not yet fully unified.
- Validation arrays must be supplied together. MMG uses them for validation; MINDE uses them for monitoring. Other estimators currently accept these arguments but do not use them for validation-based early stopping.
- Constructors infer feature shapes at first fit. Existing class names and direct imports remain available.

### Interpreting results

Use held-out samples, compare to synthetic cases with known MI, and repeat with several seeds. Finite-sample neural estimates can be negative or biased; this library does not clip them to zero. Larger training budgets alone do not guarantee accuracy. Compare methods on the same split and report sample count, preprocessing, training budget, seeds, and evaluation budget.

The unified interface is an **estimation API**, not a differentiable MI loss for end-to-end training. The common path does not automatically flatten images, encode categorical variables, or handle missing values. CPC and default DIME evaluation allocate an `n × n` score matrix; use a manageable held-out subset for large datasets.

## Development and tests

```bash
python -m pip install -e . pytest build
python -m pytest tests/test_public_api.py -q
python -m build
```

The regression suite covers all eight public imports, small datasets, unequal feature dimensions, invalid inputs, step limits, checkpoint creation, and training-only preprocessing. The older stochastic Gaussian accuracy experiments are separate:

```bash
python -m pytest -m benchmark -q
```

See [the usability audit](docs/usability-audit.md) for reproduced failures, fixes, local verification, and remaining limitations. Research benchmark scripts live under `src/benchmark/` and are not a stable installed API.

Bug reports are most useful with a minimal example, array shapes, estimator parameters, Python/PyTorch versions, and a traceback. Contributions that add an estimator should implement the same `fit`/`estimate` contract and include a small public-API test.

## Citation and acknowledgements

If you use **MMG**, cite its method paper:

```bibtex
@misc{yu2025mmg,
  title={MMG: Mutual Information Estimation via the MMSE Gap in Diffusion},
  author={Longxuan Yu and Xing Shi and Xianghao Kong and Tong Jia and Greg Ver Steeg},
  year={2025},
  eprint={2509.20609},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

Please also credit the original methods you evaluate, including [MINDE](https://arxiv.org/abs/2310.09031), [MINE](https://arxiv.org/abs/1801.04062), [CPC](https://arxiv.org/abs/1807.03748), and the authors of the other estimators. [Benchmarking Mutual Information](https://github.com/cbg-ethz/bmi) is an acknowledged source of inspiration for the benchmark workflow.
