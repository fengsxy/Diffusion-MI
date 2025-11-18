from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset

from .TrainerFactory import TrainerFactory
from ._critic import MLP, ConvCritic

class MINEEstimator(L.LightningModule):
    def __init__(self, 
                 x_shape=None,
                 y_shape=None,
                 learning_rate=1e-4,
                 batch_size=256,
                 max_n_steps=None,
                 max_epochs=None,
                 hidden_layers=(100, 100),
                 seed=42, 
                 task_name="default", 
                 task_gt=-1, 
                 test_num=10000, 
                 early_stopping=True, 
                 create_checkpoint=False,
                 # Backward compatibility: accepted but unused.
                 log_to_tensorboard: bool = False,
                ):
        super().__init__()
        self.save_hyperparameters()
        self.critic = None
        self.task_name = task_name
        self.model_name = 'mine_estimator'
        
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
                'max_steps': max_n_steps,
                'max_epochs': max_epochs,
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

    def _create_critic(self, input_dim):
        return MLP(input_dim, self.hparams.hidden_layers).to(self.device)

    def _create_image_critic(self, x_channels, y_channels):
        return ConvCritic(x_channels, y_channels).to(self.device)

    @classmethod
    def load_model(cls, checkpoint_path, **kwargs):
        '''
        To load the model, use
        model = MINDEstimator.load_model(checkpoint_path)
        '''
        model = cls.load_from_checkpoint(checkpoint_path, **kwargs)
        return model
    
    @staticmethod
    def _mine_lower_bound(t, t_shuffle):
        max_t_shuffle = torch.max(t_shuffle)
        log_sum_exp_t_shuffle = torch.log(torch.mean(torch.exp(t_shuffle - max_t_shuffle))) + max_t_shuffle
        
        mi_lb = torch.mean(t) - log_sum_exp_t_shuffle
        R = torch.exp(-mi_lb)
        return mi_lb, R

    def configure_optimizers(self):
        return optim.Adam(self.critic.parameters(), lr=self.hparams.learning_rate)
    
    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_shuffle = y_batch[torch.randperm(y_batch.size(0))]
        t = self.critic(x_batch, y_batch)
        t_shuffle = self.critic(x_batch, y_shuffle)
        mi_estimate, R = self._mine_lower_bound(t, t_shuffle)
        
        loss = -mi_estimate
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.smoothed_mi_history.append(mi_estimate)
        return loss
    
    def on_train_end(self):
        warnings = TrainerFactory.detect_warnings(self.smoothed_mi_history)
        for warning_type, warning_message in warnings.items():
            # self.log(warning_type, True) # Can't use self.log() in on_train_end().
            print(warning_message)

    def fit(self, X: np.ndarray, Y: np.ndarray, X_val=None, Y_val=None):
        
        # Infer shapes from data if not provided at construction time.
        if self.hparams.x_shape is None or self.hparams.y_shape is None:
            self.hparams.x_shape = X.shape[1:]
            self.hparams.y_shape = Y.shape[1:]
        elif X.shape[1:] != self.hparams.x_shape or Y.shape[1:] != self.hparams.y_shape:
            raise ValueError(
                f"Input shapes do not match. Expected X shape: {self.hparams.x_shape}, "
                f"Y shape: {self.hparams.y_shape}. Got X shape: {X.shape[1:]}, Y shape: {Y.shape[1:]}"
            )

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        if self.critic is None:
            input_dim = np.prod(self.hparams.x_shape) + np.prod(self.hparams.y_shape)
            self.critic = self._create_critic(input_dim)

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)
        
        # trainer = TrainerFactory.configure_trainer(
        #     task_name=self.task_name, 
        #     logger_name=self.logger_name, 
        #     train_sample_num=len(X), 
        #     test_num=self.hparams.test_num,
        #     max_n_steps=self.hparams.max_n_steps,
        #     max_epochs=self.hparams.max_epochs,
        #     early_stopping=self.hparams.early_stopping,
        #     early_stopping_params=self.early_stopping_params,
        #     create_checkpoint=self.hparams.create_checkpoint,
        #     checkpoint_params=self.checkpoint_params
        # )
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
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_shuffle = Y[torch.randperm(Y.size(0))]
            t = self.critic(X, Y)
            t_shuffle = self.critic(X, y_shuffle)
            mi_estimate, R = self._mine_lower_bound(t, t_shuffle)

        print(f"Final estimate - MI: {mi_estimate.item()}, R: {R.item()}")
        return mi_estimate.item()

if __name__ == '__main__':
    """Simple demo: MINE on a synthetic 1D Gaussian task."""

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
    X_test, Y_test = task.sample(1000, seed=41)

    mine = MINEEstimator(
    x_shape=(1,),
    y_shape=(1,),
    learning_rate=1e-4,
    batch_size=256,
    max_n_steps=3000,
    hidden_layers=(100, 100),
    seed=42,
    task_name="mine_estimation",
    )
    mine.fit(X, Y)
    mi_estimate = mine.estimate(X_test, Y_test)
    print(
        f"MINE estimated MI: {mi_estimate:.4f}, "
        f"gt={task.mutual_information:.4f}, "
        f"abs_err={abs(mi_estimate - task.mutual_information):.4f}"
    )
