# Diffusion-MI usability audit

Baseline: `e7eb358184fed02491ca6255bbf312a76ee202e6`.

## Assessment

The unified estimator interface is a useful product direction. The first priority is reliability at the entry points, followed by evidence that users can choose and tune a method successfully. A large rewrite is not necessary to address the reproduced onboarding failures. Repository traffic and download analytics were not available in this audit, so these findings do not establish why adoption is low.

Diffusion-MI is the toolbox brand; MMG is one estimator. The distribution remains `diffusion-mi`, and imports remain `dmi`.

## Reproduced failures and changes

| Finding | Evidence / impact | Change |
| --- | --- | --- |
| Small data can hang training | Original CPC with 8 rows, batch size 32, and one training step timed out after 20 seconds. CPC/DoE/NWJ/SMILE used `drop_last=True`, leaving an empty loader inside an unbounded outer loop. | Cap effective batch size at the number of samples; reject fewer than two paired rows. |
| DoE assumes equal feature widths | Original `(8, 1)` X and `(8, 2)` Y failed with a matrix multiplication shape error. | Give the conditional density network separate input and output dimensions. |
| MMG contradicts the common API | Original `MMGEstimator(...).fit(X, Y)` immediately raised a validation-data error. | Validation is optional; a partial validation pair still raises a clear error. |
| MINE fails with an epoch-only budget | Original `MINEEstimator(max_epochs=1).fit(...)` passed `max_steps=None` to Lightning and raised a TypeError. | Normalize omitted training limits before constructing a trainer. |
| MINDE ignores step budget | Its trainer never received `max_n_steps`. | Pass the actual step limit; regression checks stop at two optimizer steps. |
| MINDE refits preprocessing on test data | Source used fresh `StandardScaler.fit_transform` calls during validation and estimation. | Store training scalers and reuse `transform`; tests inspect shifted held-out data. |
| CPC temperature scaling is inconsistent | Constant score matrix at temperature 0.5 returned −1 nat instead of zero. | Scale both positive logits and normalization logits. |
| Explicit checkpointing conflicts with trainer settings | TrainerFactory created a ModelCheckpoint but always disabled checkpointing. | Enable checkpointing when requested; actual file creation is tested. |
| DIME default budget and singleton batches | One-epoch default silently capped step budgets; objectives divide by `n*(n-1)`. | Remove default epoch cap and omit incomplete training batches. |
| Poor error messages and inconsistent return values | Unfitted, mismatched, and non-finite inputs reached lower-level tensor operations; most `fit` calls returned None. | Shared validation, fitted-state checks, and chainable `fit` across eight methods. |
| Import and installation overhead | Public wildcard imports loaded all estimators. Audio and image dependencies were mandatory. | Lazy explicit public exports, remove unused audio/TensorBoardX dependencies, optional image extra. |
| Conflicting packaging and stale metadata | A second setup.py duplicated packaging; project links referenced another repo. | Keep Poetry as the packaging source, regenerate its lockfile, fix URLs, and package only API/implementation namespaces. |
| Documentation conflates methods | DIME described as diffusion; DoE described as density-of-states; MMG lacked full citation. | Describe the code's density-ratio and entropy-difference approaches; add method-selection table and verified MMG citation. |

The MINDE duplicate `score_inference` definition was removed. Explicit MINDE callback configurations are retained correctly; its existing `hidden_dim` and importance-sampling settings now reach the model/SDE. MMG and MINDE now honor their existing seed arguments at initialization.

## Local verification

Environment: Linux, Python 3.12.14, PyTorch 2.14.0+cpu, Lightning / PyTorch Lightning 2.6.6, NumPy 2.3.5. No CUDA device. A task-specific virtual environment reused existing scientific packages; this was not a fresh operating-system installation.

- **25 regression tests passed**, including all eight estimators on small datasets with unequal feature widths, inference without learned-weight updates, early stopping returns, input errors, CPC temperature, MINE checkpoint creation, MINDE training statistics, DIME singleton tails, and lazy imports.
- Built both sdist and wheel; inspected the wheel for the new validation module and exclusion of research benchmark scripts.
- Installed the wheel, then ran the 25 regression tests from outside the repository. Imports resolved to `site-packages/dmi`, not the source checkout.
- Executed all Python blocks in README order. MINE: **0.4433 nats**, analytic value **0.4133 nats**. MMG with the demonstrated 1,000-step/default-EMA budget: **0.0111 nats**. NWJ: **0.3599 nats**. These are one-run observations, not accuracy guarantees. In particular, MMG's result shows that its short onboarding budget is insufficient to demonstrate reliable accuracy on this case.
- Existing stochastic accuracy suite: **8 passed, 1 failed** in 141.57 seconds. The MMG/MIND_diff case returned 0.955 nats versus 0.413 ground truth (absolute error 0.542, tolerance 0.5) after only two epochs. Its tolerance was not loosened. These experiments are now explicitly marked `benchmark` and remain runnable separately; the failure is unresolved.
- Poetry lock validation passed with metadata-deprecation warnings. Library warnings about logging without a logger and small DataLoader worker counts remain.
- CPU-only local coverage does not validate CUDA, other Python versions, image architectures, or every DIME objective. The updated CI matrix targets Python 3.9–3.12; remote CI results must be checked separately.

## Compatibility notes

- Existing estimator names and the `dmi.estimators` namespace remain unchanged. Legacy internal exports are retained lazily.
- `fit` returns `self`. Invalid inputs now fail early. The common supported path is continuous vector arrays; do not interpret the API as generic image/categorical preprocessing.
- DIME no longer silently stops after one epoch by default. Supply `max_epochs=1` to reproduce that prior budget. MINE/MINDE/MMG with no explicit limit use a finite 1,000-step default.
- CPC/DoE/NWJ/SMILE retain their existing full-batch training policy, with a reduced batch size only when the dataset itself is smaller. DIME now drops incomplete batches to avoid singleton divisions.
- Correcting MINDE preprocessing, CPC temperature, and previously unused seeds can change numerical results. This is intentional and should be recorded when comparing older experiments.
- Research benchmark scripts remain in the checkout rather than being installed as a public `benchmark` namespace.

## Next priorities

1. **Accuracy and calibration before promotion.** Add reproducible multi-seed Gaussian sweeps over dimension, correlation, sample count, training budget, and MC budget. MMG's short-run result above needs a documented training recipe, including EMA convergence and estimator variance, before making accuracy claims.
2. **Checkpoint round trips.** Existing `load_model` paths use lazy network initialization and lack a tested unified serialization contract; MINDE additionally needs persisted scaler state. Checkpoint *creation* passing does not establish save/reload correctness. Provide explicit save/load with fitted metadata, preprocessing, architecture, and EMA state.
3. **Bounded-memory evaluation.** CPC and default DIME form pairwise score matrices. Chunking must preserve the original objective rather than silently average estimates over smaller negative sets.
4. **Consistent controls.** Standardize device selection, verbosity, local RNG handling, validation behavior, and evaluation budgets. `MMGEstimator.estimate(n_samples=...)` currently does not use that argument.
5. **Separate research artifacts from the library.** Keep experiment scripts outside the installed package and gradually move implementations into a private `dmi` namespace with compatibility shims. Avoid a namespace migration in the same change as correctness fixes.
6. **Release readiness.** Metadata declares MIT, but this checkout has no LICENSE text. The maintainers should confirm licensing/provenance of incorporated implementations before adding the correct license file. Publish a new version only after reviewing these behavioral changes; this branch does not publish to PyPI.

## Visual identity

`docs/assets/diffusion-mi-logo.png` was generated with the built-in image-generation tool. Prompt: a compact horizontal Diffusion-MI wordmark, paired interlocking loops with a highlighted shared region, subtle diffusion particles, navy/teal/turquoise palette, transparent background, no extra text. The first MMG-branded draft was superseded after the project/method naming distinction was clarified.
