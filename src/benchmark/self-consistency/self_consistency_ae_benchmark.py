import torch
import torchvision
from torchvision import transforms
from typing import Tuple, Literal
import os
import sys

# Ensure the local directory (containing ``huggingface_ae.py``) is on
# the import path so that ``from huggingface_ae import AE`` works both
# when this file is executed as a script and when it is imported
# programmatically.
_CURRENT_DIR = os.path.dirname(__file__)
if _CURRENT_DIR not in sys.path:
    sys.path.append(_CURRENT_DIR)

from huggingface_ae import AE

DatasetType = Literal['mnist', 'fashionmnist', 'cifar10']
TaskType = Literal['baseline', 'data_processing', 'additivity']

class Self_Consistency_Benchmark:
    def __init__(self, task_type: TaskType, dataset: DatasetType, rows: int, k: int = 4):
        self.task_type = task_type
        self.dataset = dataset
        self.rows = rows
        self.k = k

        if dataset == 'mnist' or dataset == 'fashionmnist':
            self.dim_x = (1, 28, 28)
            self.dim_y = (1, 28, 28)
            self.model = AE.from_pretrained(f"liddlefish/mnist_auto_encoder_crop_{rows}")
        elif dataset == 'cifar10':
            self.dim_x = (3, 32, 32)
            self.dim_y = (3, 32, 32)
            raise NotImplementedError("CIFAR10 autoencoder not implemented")

        self.name = f"{task_type}_{dataset}_{rows}"
        if task_type == 'data_processing':
            self.name += f"_{k}"

    def sample(self, n: int, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)
        
        train_dataset = self.load_dataset()
        
        sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=n, replacement=True)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=n, sampler=sampler)
        if self.task_type == 'baseline':
            X, _ = next(iter(dataloader))
            X, Y = self.transform(X)
        elif self.task_type == 'data_processing':
            X, _ = next(iter(dataloader))
            X, Y = self.transform(X)
        elif self.task_type == 'additivity':
            X_1, _ = next(iter(dataloader))
            X_2, _ = next(iter(dataloader))
            X, Y = self.transform(X_1, X_2)

        return X, Y

    def load_dataset(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) if self.dataset != 'cifar10' else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        if self.dataset == 'mnist':
            return torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
        elif self.dataset == 'cifar10':
            return torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
        elif self.dataset == 'fashionmnist':
            return torchvision.datasets.FashionMNIST('data', train=True, download=True, transform=transform)

    def standardize(self, latent: tuple) -> torch.Tensor:
        latent = latent[0]
        mean = latent.mean(dim=0)
        std = latent.std(dim=0)
        std[std == 0] = 1  # Avoid division by zero
        return (latent - mean) / std
    
    def transform(self, X_1: torch.Tensor, X_2: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.task_type == 'baseline':
            Y = X_1.clone()
            Y[:, :, self.rows:, :] = 0.0
            self.model.eval()
            with torch.no_grad():
                X_latent = self.model.encode(X_1)
                Y_latent = self.model.encode(Y)
                X_latent = self.standardize(X_latent)
                Y_latent = self.standardize(Y_latent)
            return X_latent, Y_latent
        elif self.task_type == 'data_processing':
            Y1 = X_1.clone()
            Y1[:, :, self.rows:, :] = 0.0
            Y2 = X_1.clone()
            Y2[:, :, self.rows-self.k:, :] = 0.0
            with torch.no_grad():
                X_latent = self.model.encode(X_1)
                Y1_latent = self.model.encode(Y1)
                Y2_latent = self.model.encode(Y2)
                X_latent = self.standardize(X_latent)
                Y1_latent = self.standardize(Y1_latent)
                Y2_latent = self.standardize(Y2_latent)
            X_latent = torch.cat((X_latent, X_latent), dim=1)
            Y_latent = torch.cat((Y1_latent, Y2_latent), dim=1)
            return X_latent, Y_latent
        elif self.task_type == 'additivity':
            with torch.no_grad():
                Y1 = X_1.clone()
                Y2 = X_2.clone()
                Y1[:, :, self.rows:, :] = 0.0
                Y2[:, :, self.rows:, :] = 0.0
                X_1_latent = self.model.encode(X_1)
                X_2_latent = self.model.encode(X_2)
                Y_1_latent = self.model.encode(Y1)
                Y_2_latent = self.model.encode(Y2)
                X_1_latent = self.standardize(X_1_latent)
                X_2_latent = self.standardize(X_2_latent)
                Y_1_latent = self.standardize(Y_1_latent)
                Y_2_latent = self.standardize(Y_2_latent)
            X_latent = torch.cat((X_1_latent, X_2_latent), dim=1)
            Y_latent = torch.cat((Y_1_latent, Y_2_latent), dim=1)
            return X_latent, Y_latent

    @property
    def mutual_information(self) -> float:
        if self.task_type == 'baseline':
            return self.rows / 28  # Adjusted MI
        elif self.task_type == 'data_processing':
            return 1.0  # Ideal value
        elif self.task_type == 'additivity':
            return 2.0  # Ideal value

if __name__ == '__main__':
    # Simple demo: sample latent pairs from a self-consistency task
    task = Self_Consistency_Benchmark(
        task_type='baseline',
        dataset='mnist',
        rows=14,
    )

    X_latent, Y_latent = task.sample(1000, seed=42)
    print(f"Task name: {task.name}")
    print(f"Latent X shape: {X_latent.shape}")
    print(f"Latent Y shape: {Y_latent.shape}")
    print(f"Theoretical self-consistency MI (normalized): {task.mutual_information}")
