from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from ._critic import MLP, ConvCritic

class CPCEstimator:
    def __init__(
        self,
        batch_size=128,
        max_n_steps=500,
        learning_rate=1e-3,
        hidden_layers=(100, 100),
        temperature=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        x_shape=None,
        y_shape=None
    ):
        self.batch_size = batch_size
        self.max_n_steps = max_n_steps
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.temperature = temperature
        self.device = device
        self.critic = None
        self.x_shape = x_shape
        self.y_shape = y_shape

    def _create_critic(self, input_dim):
        return MLP(input_dim, self.hidden_layers).to(self.device)

    def _create_image_critic(self, x_channels, y_channels):
        return ConvCritic(x_channels, y_channels).to(self.device)

    @staticmethod
    def _infonce_lower_bound(scores, temperature):
        positive_samples = torch.diag(scores)
        nll = -positive_samples + torch.logsumexp(scores / temperature, dim=1)
        mi = torch.mean(-nll) + torch.log(torch.tensor(scores.shape[0]))
        return mi

    def fit(self, X: np.ndarray, Y: np.ndarray, X_val=None, Y_val=None, early_stopping: bool = False,
            early_stopping_patience: int = 10, early_stopping_min_delta: float = 0.0):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        if self.critic is None:
            input_dim = X.shape[1] + Y.shape[1]
            self.critic = self._create_critic(input_dim)

        optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        steps = 0
        best_loss = float('inf')
        no_improve = 0

        pbar = tqdm(total=self.max_n_steps, desc="Training CPC")
        while steps < self.max_n_steps:
            for x_batch, y_batch in dataloader:
                if steps >= self.max_n_steps:
                    break

                scores = torch.zeros(self.batch_size, self.batch_size).to(self.device)
                for i in range(self.batch_size):
                    scores[i] = self.critic(x_batch[i].unsqueeze(0).repeat(self.batch_size, 1), y_batch).squeeze()

                mi_estimate = self._infonce_lower_bound(scores, self.temperature)
                loss = -mi_estimate

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                steps += 1
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})

                if early_stopping:
                    if loss.item() < best_loss - early_stopping_min_delta:
                        best_loss = loss.item()
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= early_stopping_patience:
                            pbar.set_postfix({'loss': loss.item(), 'early_stop': True})
                            pbar.close()
                            return

                if steps >= self.max_n_steps:
                    break

        pbar.close()

    def estimate(self, X: np.ndarray, Y: np.ndarray) -> float:
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            n_samples = X.shape[0]
            scores = torch.zeros(n_samples, n_samples).to(self.device)
            for i in range(n_samples):
                scores[i] = self.critic(X[i].unsqueeze(0).repeat(n_samples, 1), Y).squeeze()
            final_mi = self._infonce_lower_bound(scores, self.temperature).item()

        print(f"Final estimate - MI: {final_mi}")
        return final_mi

if __name__ == '__main__':
    """Simple demo: CPC on a synthetic 1D Gaussian task.

    This avoids any external BMI or dataset dependencies and mirrors
    the small Gaussian task used in our unit tests.
    """

    import numpy as np

    class _SimpleGaussianTask:
        def __init__(self, dim_x=1, dim_y=1, rho=0.75):
            self.dim_x = dim_x
            self.dim_y = dim_y
            self.rho = rho
            self.mutual_information = 0.5 * np.log(1.0 / (1.0 - rho ** 2))

        def sample(self, n, seed=None):
            rng = np.random.default_rng(seed)
            x = rng.normal(size=(n, self.dim_x))
            eps = rng.normal(size=(n, self.dim_y))
            y = self.rho * x + np.sqrt(1.0 - self.rho ** 2) * eps
            return x, y

    task = _SimpleGaussianTask(dim_x=1, dim_y=1, rho=0.75)
    x_train, y_train = task.sample(2000, seed=0)
    x_test, y_test = task.sample(500, seed=1)

    cpc = CPCEstimator(batch_size=128, max_n_steps=500, learning_rate=1e-3)
    cpc.fit(x_train, y_train)
    mi_estimate = cpc.estimate(x_test, y_test)

    print(
        f"CPC estimated MI: {mi_estimate:.4f}, "
        f"gt={task.mutual_information:.4f}, "
        f"abs_err={abs(mi_estimate - task.mutual_information):.4f}"
    )
