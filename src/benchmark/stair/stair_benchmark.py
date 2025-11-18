import numpy as np
import torch
from typing import Tuple


class CorrelatedGaussianTask:
    """Correlated Gaussian benchmark task.

    This task mirrors the simple correlated Gaussian setup used in the
    original stair-case MI demos: ``X`` and ``Y`` are jointly Gaussian
    with correlation ``rho`` in each coordinate and known closed-form
    mutual information

        I(X; Y) = -0.5 * dim * log(1 - rho**2).

    Parameters
    ----------
    name:
        Human-readable name for the task.
    dim:
        Dimensionality of ``X`` and ``Y``.
    rho:
        Correlation coefficient between corresponding coordinates of
        ``X`` and ``Y`` (|rho| < 1).
    cubic:
        If ``True``, apply a cubic non-linearity to ``Y`` to obtain a
        non-Gaussian but still MI-preserving pair.
    """

    def __init__(self, name: str, dim: int, rho: float, cubic: bool = False):
        self.name = name
        self.dim_x = dim
        self.dim_y = dim
        self.rho = float(rho)
        self.cubic = bool(cubic)
        self.mutual_information = self.compute_mutual_information()

    def compute_mutual_information(self) -> float:
        """Closed-form mutual information (in nats)."""

        if not (-1.0 < self.rho < 1.0):
            raise ValueError("rho must lie in (-1, 1) for a valid covariance.")
        return -0.5 * np.log(1.0 - self.rho**2) * self.dim_x

    def _sample_correlated_gaussian(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from a correlated Gaussian distribution.

        The implementation follows the standard construction used in
        the stair-case demo notebooks: draw ``(X, eps) ~ N(0, I)`` and
        set::

            Y = rho * X + sqrt(1 - rho**2) * eps.

        When ``self.cubic`` is ``True`` a cubic non-linearity is
        applied to ``Y``.
        """

        dim = self.dim_x
        rho = self.rho

        x, eps = torch.chunk(torch.randn(batch_size, 2 * dim), 2, dim=1)
        y = rho * x + torch.sqrt(torch.tensor(1.0 - rho**2, dtype=x.dtype)) * eps
        if self.cubic:
            y = y ** 3
        return x, y

    def sample(self, n: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Sample ``n`` pairs ``(X, Y)`` as NumPy arrays.

        Parameters
        ----------
        n:
            Number of samples to draw.
        seed:
            Optional Torch random seed for reproducibility.
        """

        if seed is not None:
            torch.manual_seed(seed)

        x, y = self._sample_correlated_gaussian(batch_size=n)
        return x.numpy(), y.numpy()
    
if __name__ == "__main__":
    task = CorrelatedGaussianTask("correlated_gaussian", 20, 0.5)
    x, y = task.sample(128)
    print(x.shape, y.shape)
    print(task.mutual_information)
