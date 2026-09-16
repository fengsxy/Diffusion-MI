"""Run the README Gaussian example with the installed dmi package."""
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
