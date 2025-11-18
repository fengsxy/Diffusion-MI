from abc import ABC, abstractmethod
import numpy as np

class MIEstimator(ABC):
    @abstractmethod
    def estimate(self, X: np.ndarray, Y: np.ndarray) -> float:
        pass

class NeuralMIEstimator(MIEstimator):
    def __init__(self, input_dim: int, hidden_dim: int, learning_rate: float = 0.001, n_epochs: int = 1000, batch_size: int = 256):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.network = self._build_network()
        self.optimizer = None  # To be initialized in subclasses

    def _build_network(self):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(2 * self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    @abstractmethod
    def compute_mi(self, scores):
        pass

    def estimate(self, X: np.ndarray, Y: np.ndarray) -> float:
        import torch
        import torch.optim as optim
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
        
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

        for _ in range(self.n_epochs):
            indices = torch.randperm(X.shape[0])[:self.batch_size]
            x_batch, y_batch = X[indices], Y[indices]
            
            self.optimizer.zero_grad()
            scores = self.network(torch.cat([x_batch, y_batch], dim=1))
            mi = self.compute_mi(scores)  
            loss = -mi
            loss.backward()
            
            self.optimizer.step()
        
        with torch.no_grad():
            scores = self.network(torch.cat([X, Y], dim=1))
            mi = self.compute_mi(scores)
        
        return mi.item()