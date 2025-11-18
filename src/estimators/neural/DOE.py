from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import random


class DoEEstimator:
    def __init__(
        self,
        batch_size=256,
        max_n_steps=3000,
        learning_rate=1e-4,
        hidden_layers=(100, 100),
        pdf='gauss',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        x_shape=None,
        y_shape=None,
        seed: int | None = 42,
    ):
        self.batch_size = batch_size
        self.max_n_steps = max_n_steps
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.pdf = pdf
        self.device = device
        self.model = None
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.seed = seed

    def _create_model(self, dim_x, dim_y):
        return DoE(dim_y, self.hidden_layers[0], len(self.hidden_layers), self.pdf).to(self.device)

    def fit(self, X: np.ndarray, Y: np.ndarray, X_val=None, Y_val=None, early_stopping: bool = False,
            early_stopping_patience: int = 10, early_stopping_min_delta: float = 0.0):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        # Save RNG state so we can restore it after training and avoid
        # affecting global randomness used elsewhere in the codebase.
        torch_rng_state = torch.get_rng_state()
        np_rng_state = np.random.get_state()
        python_rng_state = random.getstate()
        cuda_rng_states = None
        if torch.cuda.is_available():
            try:
                cuda_rng_states = torch.cuda.get_rng_state_all()
            except Exception:
                cuda_rng_states = None

        # Deterministic initialization for stability across runs.
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            if torch.cuda.is_available():
                try:
                    torch.cuda.manual_seed_all(self.seed)
                except Exception:
                    pass

        try:
            if self.model is None:
                self.model = self._create_model(X.shape[1], Y.shape[1])

            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            dataset = TensorDataset(X, Y)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

            steps = 0
            best_loss = float('inf')
            no_improve = 0
            pbar = tqdm(total=self.max_n_steps, desc="Training DoE")
            while steps < self.max_n_steps:
                for x_batch, y_batch in dataloader:
                    if steps >= self.max_n_steps:
                        break

                    XY_package = torch.cat([x_batch, y_batch], dim=1)
                    loss = self.model(x_batch, y_batch, XY_package)

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
        finally:
            # Restore RNG state to avoid side effects for callers.
            torch.set_rng_state(torch_rng_state)
            np.random.set_state(np_rng_state)
            random.setstate(python_rng_state)
            if cuda_rng_states is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
                except Exception:
                    pass

    def estimate(self, X: np.ndarray, Y: np.ndarray) -> float:
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            XY_package = torch.cat([X, Y], dim=1)
            mi_estimate = -self.model(X, Y, XY_package).item()

        return mi_estimate

class DoE(nn.Module):
    def __init__(self, dim, hidden, layers, pdf):
        super(DoE, self).__init__()
        self.qY = PDF(dim, pdf)
        self.qY_X = ConditionalPDF(dim, hidden, layers, pdf)

    def forward(self, X, Y, XY_package):
        hY = self.qY(Y)
        hY_X = self.qY_X(Y, X)

        loss = hY + hY_X
        mi_loss = hY_X - hY
        return (mi_loss - loss).detach() + loss

class ConditionalPDF(nn.Module):
    def __init__(self, dim, hidden, layers, pdf):
        super(ConditionalPDF, self).__init__()
        assert pdf in {'gauss', 'logistic'}
        self.dim = dim
        self.pdf = pdf
        self.X2Y = FF(dim, hidden, 2 * dim, layers)

    def forward(self, Y, X):
        mu, ln_var = torch.split(self.X2Y(X), self.dim, dim=1)
        cross_entropy = compute_negative_ln_prob(Y, mu, ln_var, self.pdf)
        return cross_entropy

class PDF(nn.Module):
    def __init__(self, dim, pdf):
        super(PDF, self).__init__()
        assert pdf in {'gauss', 'logistic'}
        self.dim = dim
        self.pdf = pdf
        self.mu = nn.Embedding(1, self.dim)
        self.ln_var = nn.Embedding(1, self.dim)  # ln(s) in logistic

    def forward(self, Y):
        cross_entropy = compute_negative_ln_prob(Y, self.mu.weight,
                                                 self.ln_var.weight, self.pdf)
        return cross_entropy

class FF(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, num_layers,
                 activation='tanh', dropout_rate=0, layer_norm=False,
                 residual_connection=False):
        super(FF, self).__init__()
        assert (not residual_connection) or (dim_hidden == dim_input)
        self.residual_connection = residual_connection

        self.stack = nn.ModuleList()
        for l in range(num_layers):
            layer = []

            if layer_norm:
                layer.append(nn.LayerNorm(dim_input if l == 0 else dim_hidden))

            layer.append(nn.Linear(dim_input if l == 0 else dim_hidden,
                                   dim_hidden))
            layer.append({'tanh': nn.Tanh(), 'relu': nn.ReLU()}[activation])
            layer.append(nn.Dropout(dropout_rate))

            self.stack.append(nn.Sequential(*layer))

        self.out = nn.Linear(dim_input if num_layers < 1 else dim_hidden,
                             dim_output)

    def forward(self, x):
        for layer in self.stack:
            x = x + layer(x) if self.residual_connection else layer(x)
        return self.out(x)

def compute_negative_ln_prob(Y, mu, ln_var, pdf):
    var = ln_var.exp()

    if pdf == 'gauss':
        negative_ln_prob = 0.5 * ((Y - mu) ** 2 / var).sum(1).mean() + \
                           0.5 * Y.size(1) * np.log(2 * np.pi) + \
                           0.5 * ln_var.sum(1).mean()

    elif pdf == 'logistic':
        whitened = (Y - mu) / var
        adjust = torch.logsumexp(
            torch.stack([torch.zeros(Y.size()).to(Y.device), -whitened]), 0)
        negative_ln_prob = whitened.sum(1).mean() + \
                           2 * adjust.sum(1).mean() + \
                           ln_var.sum(1).mean()

    else:
        raise ValueError('Unknown PDF: %s' % (pdf))

    return negative_ln_prob

if __name__ == '__main__':
    """Simple demo: DoE on a synthetic 1D Gaussian task."""

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
    X, Y = task.sample(50000, seed=42)

    doe = DoEEstimator(max_n_steps=3000, learning_rate=1e-4, batch_size=256)
    doe.fit(X, Y)
    mi_estimate = doe.estimate(X, Y)

    print(
        f"DoE estimated MI: {mi_estimate:.4f}, "
        f"gt={task.mutual_information:.4f}, "
        f"abs_err={abs(mi_estimate - task.mutual_information):.4f}"
    )
