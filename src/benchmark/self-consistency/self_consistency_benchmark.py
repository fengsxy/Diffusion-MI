import torch
import torchvision
from torchvision import transforms
from typing import Tuple, Literal

DatasetType = Literal['mnist', 'fashionmnist', 'cifar10']
TaskType = Literal['baseline', 'data_processing', 'additivity']

class Self_Consistency_Benchmark:
    def __init__(self, task_type: TaskType, dataset: DatasetType, rows: int, k: int = None):
        self.task_type = task_type
        self.dataset = dataset
        self.rows = rows
        self.k = k

        if dataset == 'mnist' or dataset == 'fashionmnist':
            self.dim_x = (1, 28, 28)
            self.dim_y = (1, 28, 28)
        elif dataset == 'cifar10':
            self.dim_x = (3, 32, 32)
            self.dim_y = (3, 32, 32)

        self.name = f"{task_type}_{dataset}_{rows}"
        if task_type == 'data_processing':
            self.name += f"_{k}"

    def sample(self, n: int, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)
        
        # Load dataset
        train_dataset = self.load_dataset()
        
        # Sample n images
        sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=n, replacement=True)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=n, sampler=sampler)
        X, _ = next(iter(dataloader))
        
        # Apply transformation to get Y
        Y = self.transform(X)
        
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

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.task_type == 'baseline':
            Y = X.clone()
            Y[:, :, self.rows:, :] = 0.0
            return Y
        elif self.task_type == 'data_processing':
            Y1 = X.clone()
            Y1[:, :, self.rows:, :] = 0.0
            Y2 = X.clone()
            Y2[:, :, self.rows+self.k:, :] = 0.0
            Y = torch.cat((Y1, Y2), dim=1)
            X = torch.cat((X, X), dim=1)
            return Y
        elif self.task_type == 'additivity':
            n = X.shape[0] // 2
            X1, X2 = X[:n], X[n:]
            Y1 = X1.clone()
            Y1[:, :, self.rows:, :] = 0.0
            Y2 = X2.clone()
            Y2[:, :, self.rows:, :] = 0.0
            Y = torch.cat((Y1, Y2), dim=0)
            return Y

    @property
    def mutual_information(self) -> float:
        if self.task_type == 'baseline':
            return self.rows / self.dim_x[1]  # Theoretical MI
        elif self.task_type == 'data_processing':
            return 1.0  # Ideal value
        elif self.task_type == 'additivity':
            return 2.0  # Ideal value

if __name__ == '__main__':
    # 创建基线测试任务
    baseline_task = Self_Consistency_Benchmark(task_type='baseline', dataset='mnist', rows=14)
    print(f"Task: {baseline_task.name}")
    print(f"Task {baseline_task.name} with dimensions {baseline_task.dim_x} and {baseline_task.dim_y}")
    print(f"Ground truth mutual information: {baseline_task.mutual_information:.2f}")
    X, Y = baseline_task.sample(1000)

    # 创建数据处理测试任务
    dp_task = Self_Consistency_Benchmark(task_type='data_processing', dataset='cifar10', rows=16, k=4)
    print(f"Task: {dp_task.name}")
    print(f"Task {dp_task.name} with dimensions {dp_task.dim_x} and {dp_task.dim_y}")
    print(f"Ground truth mutual information: {dp_task.mutual_information:.2f}")
    X, Y = dp_task.sample(1000)

    # 创建可加性测试任务
    additivity_task = Self_Consistency_Benchmark(task_type='additivity', dataset='fashionmnist', rows=14)
    print(f"Task: {additivity_task.name}")
    print(f"Task {additivity_task.name} with dimensions {additivity_task.dim_x} and {additivity_task.dim_y}")
    print(f"Ground truth mutual information: {additivity_task.mutual_information:.2f}")
    X, Y = additivity_task.sample(1000)

    # 获取互信息
    mi = baseline_task.mutual_information
