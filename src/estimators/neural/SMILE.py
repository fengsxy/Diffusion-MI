from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from ._critic import MLP, ConvCritic

class SMILEEstimator:
    def __init__(
        self,
        batch_size=256,
        max_n_steps=3000,
        learning_rate=1e-4,
        hidden_layers=(100, 100),
        clip_value=20.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        x_shape=None,
        y_shape=None
    ):
        self.batch_size = batch_size
        self.max_n_steps = max_n_steps
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.clip_value = clip_value
        self.device = device
        self.critic = None
        self.x_shape = x_shape
        self.y_shape = y_shape

    def _create_critic(self, input_dim):
        return MLP(input_dim, self.hidden_layers).to(self.device)

    def _create_image_critic(self, x_channels, y_channels):
        return ConvCritic(x_channels, y_channels).to(self.device)

    @staticmethod
    def _smile_lower_bound(t, t_shuffle, clip_value):
        t_clip = t
        t_shuffle_clip = torch.clamp(t_shuffle, -clip_value, clip_value)
        js = torch.mean(-torch.nn.functional.softplus(-t_clip)) - \
            torch.mean(torch.nn.functional.softplus(t_shuffle_clip))
        dv = torch.mean(t) - torch.log(torch.mean(torch.exp(t_shuffle)))
        return js + (dv - js).detach()

    def fit(self, X: np.ndarray, Y: np.ndarray, X_val=None, Y_val=None, early_stopping: bool = False,
            early_stopping_patience: int = 10, early_stopping_min_delta: float = 0.0):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        if self.x_shape is None:
            self.x_shape = X.shape[1:]
        if self.y_shape is None:
            self.y_shape = Y.shape[1:]

        if self.critic is None:
            input_dim = np.prod(self.x_shape) + np.prod(self.y_shape)
            self.critic = self._create_critic(input_dim)

        optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        steps = 0
        best_loss = float('inf')
        no_improve = 0

        pbar = tqdm(total=self.max_n_steps, desc="Training SMILE")
        while steps < self.max_n_steps:
            for x_batch, y_batch in dataloader:
                if steps >= self.max_n_steps:
                    break

                y_shuffle = y_batch[torch.randperm(y_batch.size(0))]
                t = self.critic(x_batch, y_batch)
                t_shuffle = self.critic(x_batch, y_shuffle)
                mi_estimate = self._smile_lower_bound(t, t_shuffle, self.clip_value)

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
            y_shuffle = Y[torch.randperm(Y.size(0))]
            t = self.critic(X, Y)
            t_shuffle = self.critic(X, y_shuffle)
            mi_estimate = self._smile_lower_bound(t, t_shuffle, self.clip_value)

        print(f"Final estimate - MI: {mi_estimate.item()}")
        return mi_estimate.item()

if __name__ == '__main__':
    """Simple demo: SMILE on a synthetic 1D Gaussian task."""

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
    X, Y = task.sample(5000, seed=42)

    smile = SMILEEstimator(max_n_steps=3000, batch_size=256, learning_rate=1e-4)
    smile.fit(X, Y)
    mi_estimate = smile.estimate(X, Y)

    print(
        f"SMILE estimated MI: {mi_estimate:.4f}, "
        f"gt={task.mutual_information:.4f}, "
        f"abs_err={abs(mi_estimate - task.mutual_information):.4f}"
    )
