from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from ._critic import MLP, ConvCritic

class NWJEstimator:
    def __init__(
        self,
        batch_size=256,
        max_n_steps=3000,
        learning_rate=1e-4,
        hidden_layers=(100, 100),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        x_shape=None,
        y_shape=None
    ):
        self.batch_size = batch_size
        self.max_n_steps = max_n_steps
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.device = device
        self.critic = None
        self.x_shape = x_shape
        self.y_shape = y_shape

    def _create_critic(self, input_dim):
        return MLP(input_dim, self.hidden_layers).to(self.device)

    def _create_image_critic(self, x_channels, y_channels):
        return ConvCritic(x_channels, y_channels).to(self.device)
    

    '''
    @staticmethod
    def _nwj_lower_bound(joint_scores, marginal_scores):
        joint_term = torch.mean(joint_scores)
        marginal_term = torch.mean(torch.exp(marginal_scores - 1))
        return joint_term - marginal_term
    '''
    
    @staticmethod
    def _nwj_lower_bound(joint_scores, marginal_scores):
        # numerically stable version
        max_marginal = torch.max(marginal_scores)
        marginal_shift = marginal_scores - max_marginal
        joint_term = torch.mean(joint_scores)
        log_marginal_term = torch.log(torch.mean(torch.exp(marginal_shift))) + max_marginal - 1
        marginal_term = torch.exp(log_marginal_term)

        return joint_term - marginal_term

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
        pbar = tqdm(total=self.max_n_steps, desc="Training NWJ")
        while steps < self.max_n_steps:
            for x_batch, y_batch in dataloader:
                if steps >= self.max_n_steps:
                    break
                
                y_shuffle = y_batch[torch.randperm(y_batch.size(0))]
                joint_scores = self.critic(x_batch, y_batch)
                marginal_scores = self.critic(x_batch, y_shuffle)
                
                mi_estimate = self._nwj_lower_bound(joint_scores, marginal_scores)
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
            joint_scores = self.critic(X, Y)
            marginal_scores = self.critic(X, y_shuffle)
            mi_estimate = self._nwj_lower_bound(joint_scores, marginal_scores)

        print(f"Final estimate - MI: {mi_estimate.item()}")
        return mi_estimate.item()

if __name__ == '__main__':
    """Simple demo: NWJ on a synthetic 1D Gaussian task."""

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

    nwj = NWJEstimator(max_n_steps=3000, batch_size=256, learning_rate=1e-4)
    nwj.fit(X, Y)
    mi_estimate = nwj.estimate(X, Y)

    print(
        f"NWJ estimated MI: {mi_estimate:.4f}, "
        f"gt={task.mutual_information:.4f}, "
        f"abs_err={abs(mi_estimate - task.mutual_information):.4f}"
    )
