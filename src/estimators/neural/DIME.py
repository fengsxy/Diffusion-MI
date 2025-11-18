from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from .TrainerFactory import TrainerFactory
from ._critic import (
    ConcatCritic, SeparableCritic, ConvolutionalCritic
)


class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        '''
        Initialize the discriminator.
        '''
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, input_tensor):
        output_tensor = self.main(input_tensor)
        return output_tensor

class CombinedArchitecture(nn.Module):
    """
    Class combining two equal neural network architectures.
    """
    def __init__(self, single_architecture, divergence):
        super(CombinedArchitecture, self).__init__()
        self.divergence = divergence
        self.single_architecture = single_architecture
        if self.divergence == "GAN":
            self.final_activation = nn.Sigmoid()
        elif self.divergence == "KL" or self.divergence == "HD":
            self.final_activation = nn.Softplus()
        else:
            self.final_activation = nn.Identity()

    def forward(self, input_tensor_1, input_tensor_2):
        intermediate_1 = self.single_architecture(input_tensor_1)
        output_tensor_1 = self.final_activation(intermediate_1)
        intermediate_2 = self.single_architecture(input_tensor_2)
        output_tensor_2 = self.final_activation(intermediate_2)
        return output_tensor_1, output_tensor_2

class DIMEEstimator(L.LightningModule):
    def __init__(
        self,
        x_shape=None,  # Optional; inferred from data if not provided.
        y_shape=None,
        learning_rate=1e-4,
        batch_size=256,
        max_n_steps=1000,
        max_epochs=1,
        hidden_layers=(256, 256),
        divergence='GAN',
        architecture='separable',
        alpha=1,
        seed=42,
        task_name="default",
        task_gt=-1,
        test_num=10000,
        early_stopping=False,
        create_checkpoint=False,
        # Backward compatibility: accepted but unused.
        log_to_tensorboard: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.critic = None
        self.task_name = task_name
        self.model_name = 'dime_estimator'
        self.alpha = alpha
        #self.automatic_optimization = False
        
        self.logger_name = f"{self.model_name}_{task_name}_seed_{seed}_lr_{learning_rate}"
        if max_n_steps:
            self.logger_name += f"_max_steps_{max_n_steps}"
        if max_epochs:
            self.logger_name += f"_max_epochs_{max_epochs}"
            
        self.task_gt = task_gt
        
        self.smoothed_mi_history = []
        if seed is not None:
            L.seed_everything(seed, workers=True)
        
        self.trainer_config = {
            'trainer': {
                **({'max_steps': max_n_steps} if max_n_steps is not None else {}),
                **({'max_epochs': max_epochs} if max_epochs is not None else {}),
                'log_every_n_steps': 10,
            },
            'early_stopping': {
                'monitor': 'train_loss',
                'mode': 'min',
                'patience': 5,
                'min_delta': 0.00
            } if early_stopping else None,
            'checkpoint': {
                'dirpath': 'checkpoints',
                'filename': '{model_name}-{logger_name}-{train_sample_num}',
                'save_last': True,
                'save_top_k': 1,
                'monitor': 'train_loss',
                'mode': 'min'
            } if create_checkpoint else None,
            # Logger is entirely disabled in this setup.
            'logger': None,
        }

    def _create_critic(self):
        """Create the critic network based on the chosen architecture.

        Note: ``x_shape`` and ``y_shape`` in the hyperparameters are tuples, so
        we need to convert them to integers (flattened dimensions) before
        constructing MLP-based critics. Passing the raw tuple directly to
        ``nn.Linear`` leads to errors like
        ``empty(): argument 'size' failed to unpack``.
        """

        x_dim = int(np.prod(self.hparams.x_shape))
        y_dim = int(np.prod(self.hparams.y_shape))
        input_dim = x_dim + y_dim
        latent_dim = input_dim // 2

        if self.hparams.architecture == 'joint':
            return ConcatCritic(latent_dim, 256, 2, 'relu', self.hparams.divergence)
        elif self.hparams.architecture == 'separable':
            return SeparableCritic(x_dim, y_dim, 256, 32, 2, 'relu', self.hparams.divergence)
        elif self.hparams.architecture == 'deranged':
            single_model = Discriminator(2 * latent_dim, 1)
            return CombinedArchitecture(single_model, self.hparams.divergence)
        elif self.hparams.architecture == 'conv_critic':
            return ConvolutionalCritic(self.hparams.divergence, None)

    @classmethod
    def load_model(cls, checkpoint_path, **kwargs):
        model = cls.load_from_checkpoint(checkpoint_path, **kwargs)
        return model
    
    def data_generation_mi(self,data_x, data_y, device="cpu"):
        """
        Generates samples of the product of marginal distributions, given the samples from the joint distribution.
        """
        def derangement(l, device):
            import random
            """Random derangement"""
            o = l[:]
            while any(x == y for x, y in zip(o, l)):
                random.shuffle(l)
            return torch.Tensor(l).int().to(device)
        dismutations = True
        data_xy = torch.hstack((data_x, data_y))
        if dismutations:  # Derangement
            data_y_shuffle = torch.index_select(data_y, 0, derangement(list(range(data_y.shape[0])), device))
        else:  # Permutation
            data_y_shuffle = torch.index_select(data_y, 0, torch.Tensor(np.random.permutation(data_y.shape[0])).int().to(device))

        data_x_y = torch.hstack((data_x, data_y_shuffle))
        return data_xy, data_x_y

    def configure_optimizers(self):
        return optim.Adam(self.critic.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        
        if self.hparams.architecture == 'deranged':
            x_batch, y_batch = self.data_generation_mi(x_batch, y_batch, self.device)
            D_value_1, D_value_2 = self.critic(x_batch, y_batch)
            loss, R = self._compute_loss_ratio(D_value_1=D_value_1, D_value_2=D_value_2)
        else:
            scores = self.critic(x_batch, y_batch)
            loss, R = self._compute_loss_ratio(scores=scores)
            
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        mi_estimate = torch.log(R).mean().item()
        self.smoothed_mi_history.append(mi_estimate)
        return loss

    def on_train_end(self):
        warnings = TrainerFactory.detect_warnings(self.smoothed_mi_history)
        for warning_type, warning_message in warnings.items():
            print(warning_message)

    def fit(self, X: np.ndarray, Y: np.ndarray, X_val=None, Y_val=None):
        # Infer shapes from data if not provided at construction time.
        if self.hparams.x_shape is None or self.hparams.y_shape is None:
            self.hparams.x_shape = X.shape[1:]
            self.hparams.y_shape = Y.shape[1:]

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        if self.critic is None:
            self.critic = self._create_critic()

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)
        
        trainer = TrainerFactory.configure_trainer(
            trainer_config=self.trainer_config,
            model_name=self.model_name,
            task_name=self.task_name,
            logger_name=self.logger_name,
            train_sample_num=len(X),
            test_num=self.hparams.test_num
        )
        trainer.fit(self, dataloader)

    def estimate(self, X: np.ndarray, Y: np.ndarray) -> float:
        self.critic.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            Y = torch.tensor(Y, dtype=torch.float32).to(self.device)
            
            if self.hparams.architecture == 'deranged':
                X, Y = self.data_generation_mi(X, Y, self.device)
                D_value_1, D_value_2 = self.critic(X, Y)
                _, R = self._compute_loss_ratio(D_value_1, D_value_2)
            else:
                scores = self.critic(X, Y)
                _, R = self._compute_loss_ratio(scores=scores)
            mi_estimate = torch.log(R).mean().item()
        
        print(f"Final estimate - MI: {mi_estimate:.4f}")
        return mi_estimate

    def _compute_loss_ratio(self, D_value_1=None, D_value_2=None, scores=None):
        if self.hparams.divergence == 'KL':
            if self.hparams.architecture == 'deranged':
                loss, R,_ = self._kl_dime_deranged(D_value_1, D_value_2)
            else:
                loss, R = self._kl_dime_e(scores)
        elif self.hparams.divergence == 'GAN':
            if self.hparams.architecture == 'deranged':
                loss, R = self._gan_dime_deranged(D_value_1, D_value_2)
            else:
                loss, R = self._gan_dime_e(scores)
        elif self.hparams.divergence == 'HD':
            if self.hparams.architecture == 'deranged':
                loss, R = self._hd_dime_deranged(D_value_1, D_value_2)
            else:
                loss, R = self._hd_dime_e(scores)
        return loss, R

    def _kl_dime_deranged(self, D_value_1, D_value_2):
        eps = 1e-5
        batch_size_1 = D_value_1.size(0)
        batch_size_2 = D_value_2.size(0)
        valid_1 = torch.ones((batch_size_1, 1), device=self.device)
        valid_2 = torch.ones((batch_size_2, 1), device=self.device)
        loss_1 = self._my_binary_crossentropy(valid_1, D_value_1) * self.alpha
        loss_2 = self._wasserstein_loss(valid_2, D_value_2)
        loss = loss_1 + loss_2
        J_e = self.alpha * torch.mean(torch.log(D_value_1 + eps)) - torch.mean(D_value_2)
        VLB_e = J_e / self.alpha + 1 - np.log(self.alpha)
        R = D_value_1 / self.alpha
        return loss, R, VLB_e

    def _kl_dime_e(self, scores):
        eps = 1e-7
        scores_diag = scores.diag()
        n = scores.size(0)
        scores_no_diag = scores - scores_diag * torch.eye(n, device=self.device)
        loss_1 = -torch.mean(torch.log(scores_diag + eps))
        loss_2 = torch.sum(scores_no_diag) / (n*(n-1))
        loss = loss_1 + loss_2
        return loss, scores_diag

    def _gan_dime_deranged(self, D_value_1, D_value_2):
        BCE = nn.BCELoss()
        batch_size_1 = D_value_1.size(0)
        batch_size_2 = D_value_2.size(0)
        valid_2 = torch.ones((batch_size_2, 1), device=self.device)
        fake_1 = torch.zeros((batch_size_1, 1), device=self.device)
        loss_1 = BCE(D_value_1, fake_1)
        loss_2 = BCE(D_value_2, valid_2)
        loss = loss_1 + loss_2
        R = (1 - D_value_1) / D_value_1
        return loss, R

    def _gan_dime_e(self, scores):
        eps = 1e-5
        batch_size = scores.size(0)
        scores_diag = scores.diag()
        scores_no_diag = scores - scores_diag*torch.eye(batch_size, device=self.device) + torch.eye(batch_size, device=self.device)
        R = (1 - scores_diag) / scores_diag
        loss_1 = torch.mean(torch.log(torch.ones(scores_diag.shape, device=self.device) - scores_diag + eps))
        loss_2 = torch.sum(torch.log(scores_no_diag + eps)) / (batch_size*(batch_size-1))
        return -(loss_1+loss_2), R

    def _hd_dime_deranged(self, D_value_1, D_value_2):
        batch_size_1 = D_value_1.size(0)
        batch_size_2 = D_value_2.size(0)
        valid_1 = torch.ones((batch_size_1, 1), device=self.device)
        valid_2 = torch.ones((batch_size_2, 1), device=self.device)
        loss_1 = self._wasserstein_loss(valid_1, D_value_1)
        loss_2 = self._reciprocal_loss(valid_2, D_value_2)
        loss = loss_1 + loss_2
        R = 1 / (D_value_1 ** 2)
        return loss, R

    def _hd_dime_e(self, scores):
        eps = 1e-5
        Eps = 1e7
        scores_diag = scores.diag()
        n = scores.size(0)
        scores_no_diag = scores + Eps * torch.eye(n, device=self.device)
        loss_1 = torch.mean(scores_diag)
        loss_2 = torch.sum(torch.pow(scores_no_diag, -1))/(n*(n-1))
        loss = -(2 - loss_1 - loss_2)
        return loss, 1 / (scores_diag**2)

    def _my_binary_crossentropy(self, y_true, y_pred):
        eps = 1e-7
        return -torch.mean(torch.log(y_true)+torch.log(y_pred + eps))

    def _wasserstein_loss(self, y_true, y_pred):
        return torch.mean(y_true * y_pred)

    def _reciprocal_loss(self, y_true, y_pred):
        return torch.mean(1 / y_pred)

if __name__ == '__main__':
    """Simple demo: DIME on a synthetic 1D Gaussian task."""

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
    X_test, Y_test = task.sample(1000, seed=0)

    dime = DIMEEstimator(
        x_shape=(1,),
        y_shape=(1,),
        learning_rate=1e-4,
        batch_size=256,
        max_n_steps=1000,
        max_epochs=1,
        hidden_layers=(256, 256),
        divergence='GAN',
        architecture='separable',
        seed=42,
        task_name="dime_demo",
        task_gt=task.mutual_information,
        test_num=1000,
        early_stopping=False,
        create_checkpoint=False,
    )

    dime.fit(X, Y)
    mi_estimate = dime.estimate(X_test, Y_test)
    print(
        f"DIME estimated MI: {mi_estimate:.4f}, "
        f"gt={task.mutual_information:.4f}, "
        f"abs_err={abs(mi_estimate - task.mutual_information):.4f}"
    )
