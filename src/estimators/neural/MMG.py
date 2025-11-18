import math
import numpy as np
import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from pytorch_lightning.callbacks import ModelCheckpoint
import os

from ._critic import UnetMLP
from sklearn import preprocessing
from .libs.util import EMA, SNRMMSEPlotter

class Denoiser(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim=128, n_layers=3, emb_size=64):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim   
        input_dim = x_dim + y_dim
        hidden_dim = 64 if input_dim <= 10 else 128 if input_dim <= 50 else 256
        self.unet = UnetMLP(dim=input_dim, 
                            init_dim=hidden_dim, 
                            dim_mults=[], 
                            time_dim=hidden_dim, 
                            nb_mod=1,
                            out_dim=x_dim)
            
    def forward(self, x, logsnr, y=None):
        if y is None:
            y = t.zeros(x.shape[0], self.y_dim, device=x.device)
        input_tensor = t.cat([x.flatten(1), y.flatten(1)], dim=1)
        return self.unet(input_tensor, logsnr)



class MMGEstimator(pl.LightningModule):
    def __init__(self, 
             x_shape=(1,), 
             y_shape=(1,),
             learning_rate=1e-4, 
             batch_size=512,
             logsnr_loc=2., 
             logsnr_scale=3., 
             max_n_steps=None, 
             max_epochs=None,
             seed=42, 
             task_name="default", 
             task_gt=-1, 
             test_num=10000, 
             mi_estimation_interval=500,
             update_logsnr_loc_flag=False,
             use_ema=True,
             ema_decay=0.999,
             test_batch_size=1000,
             create_checkpoint=False,
             # Kept for backward compatibility with old checkpoints;
             # no-op in current implementation.
             log_to_tensorboard: bool = False,
             enable_plot: bool = False):

        super().__init__()
        self.save_hyperparameters()
        self.d_x = np.prod(x_shape)
        self.d_y = np.prod(y_shape)
        self.h_g = 0.5 * self.d_x * math.log(2 * math.pi * math.e)
        self.left = (-1,) + (1,) * len(x_shape)
        self.mi_estimation_interval = mi_estimation_interval
        self.model = Denoiser(self.d_x, self.d_y)
        self.task_name = task_name
        self.logger_name = f"mind_estimator_{task_name}_seed_{seed}_lr_{learning_rate}_update_{update_logsnr_loc_flag}"
        self.task_gt = task_gt
        self.test_X = None
        self.test_Y = None
        self.test_num = test_num
        self.logsnr_scale = logsnr_scale
        self.use_ema = use_ema
        if max_n_steps:
            self.logger_name += f"_max_steps_{max_n_steps}"
        if max_epochs:
            self.logger_name += f"_max_epochs_{max_epochs}"
        self.logsnr_loc = t.tensor(logsnr_loc, device=self.device)
        self.update_logsnr_loc_flag = update_logsnr_loc_flag
        self.model_ema = EMA(self.model, decay=ema_decay) if use_ema else None
        self.plotter = None

    def on_before_backward(self, loss: t.Tensor) -> None:
        if self.model_ema:
            self.model_ema.update(self.model)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        if t.rand(1) < 0.5:
            loss = self.nll(x)            
        else:
            loss = self.nll(x, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if self.global_step % self.mi_estimation_interval == 0:
            with t.no_grad():
                loss_xy_train = self.nll(x, y)
                loss_x_train = self.nll(x)
            
                
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        x, y = batch
        loss_x = self.nll(x)
        loss_xy = self.nll(x, y)
        mi_estimate, mi_orthogonal = self.estimate_mi_during_training(x, y)
        return loss_x, loss_xy

    def estimate_mi_during_training(self, x, y):
        self.eval()
        with t.no_grad():
            nll_x = self.nll(x)
            nll_xy = self.nll(x, y)
            mi_estimate = nll_x - nll_xy
            mi_estimate_orthogonal = self.estimate_x_y(x, y)
        self.train()
        return mi_estimate.item(), mi_estimate_orthogonal.item()

    def configure_optimizers(self):
        optimizer = t.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-5,
        )
        # Simple ReduceLROnPlateau scheduler; omit the ``verbose`` kwarg
        # for broader compatibility across torch versions.
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=200,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }

    def noisy_channel(self, x, logsnr):
        logsnr = logsnr.view(self.left)
        eps = t.randn_like(x)
        return t.sqrt(t.sigmoid(logsnr)) * x + t.sqrt(t.sigmoid(-logsnr)) * eps, eps

    def mse(self, x, logsnr, y=None):
        z, eps = self.noisy_channel(x, logsnr)
        eps_hat = self.model(z, logsnr, y)
        error = (eps - eps_hat).flatten(start_dim=1)
        return t.einsum('ij,ij->i', error, error)

    def nll(self, x, y=None):
        logsnr, weights = self.logistic_integrate(len(x))
        mses = self.mse(x, logsnr, y)
        mmse_gap = mses - self.d_x * t.sigmoid(logsnr)
        return self.h_g + 0.5 * (weights * mmse_gap).mean()

    def logistic_integrate(self, npoints, clip=4.):
        loc, scale = self.logsnr_loc, self.logsnr_scale
        loc, scale, clip = map(lambda x: t.tensor(x, device=self.device), [loc, scale, clip])
        ps = t.rand(npoints, device=self.device)
        ps = t.sigmoid(-clip) + (t.sigmoid(clip) - t.sigmoid(-clip)) * ps
        logsnr = loc + scale * (t.log(ps) - t.log(1-ps))
        weights = scale * t.tanh(clip / 2) / (t.sigmoid((logsnr - loc)/scale) * t.sigmoid(-(logsnr - loc)/scale))
        return logsnr, weights

    def mse_x_y(self, x, logsnr, y):
        z, eps = self.noisy_channel(x, logsnr)
        eps_hat_x = self.model(z, logsnr)
        eps_hat_y = self.model(z, logsnr, y)
        error = (eps_hat_x - eps_hat_y).flatten(start_dim=1)
        return t.einsum('ij,ij->i', error, error)

    def estimate_x_y(self, x, y):
        logsnr, weights = self.logistic_integrate(len(x))
        mses = self.mse_x_y(x, logsnr, y)
        return 0.5 * (weights * mses).mean()

    def configure_trainer(self, train_sample_num, **trainer_kwargs):
        # Configure Trainer without external loggers or checkpoints.
        callbacks = []

        # We keep the checkpoint option, but default is False.
        if getattr(self.hparams, "create_checkpoint", False):
            checkpoint_callback = ModelCheckpoint(
                dirpath=f'checkpoints/{self.task_name}',
                filename=f'mind_estimator-{self.logger_name}-{train_sample_num}',
                save_last=True,
                save_top_k=1,
                monitor='train_loss',
                mode='min',
            )
            callbacks.append(checkpoint_callback)

        # Plotter requires a logger; since we no longer use loggers by
        # default, we disable plotting unless explicitly enabled and a
        # custom logger is provided externally.
        self.plotter = None

        trainer_kwargs.update({
            'callbacks': callbacks,
            'logger': False,
            'log_every_n_steps': 10,
            # Disable Lightning's default checkpointing callback unless
            # explicit checkpointing is requested via the
            # ``create_checkpoint`` hyperparameter.
            'enable_checkpointing': getattr(self.hparams, 'create_checkpoint', False),
        })
        # Default to single-device (no DDP) training unless overridden.
        if 'accelerator' not in trainer_kwargs:
            trainer_kwargs['accelerator'] = 'gpu' if t.cuda.is_available() else 'cpu'
        trainer_kwargs.setdefault('devices', 1)
       
        if self.hparams.max_n_steps:
            trainer_kwargs['max_steps'] = self.hparams.max_n_steps
        if self.hparams.max_epochs:
            trainer_kwargs['max_epochs'] = self.hparams.max_epochs

        trainer = pl.Trainer(**trainer_kwargs)
        return trainer
    
    def fit(self, X: np.ndarray, Y: np.ndarray, X_val=None, Y_val=None):
        if X_val is None or Y_val is None:
            raise ValueError("MMGEstimator.fit requires validation data X_val and Y_val.")
        train_sample_num = len(X)
        
        dataset = TensorDataset(t.tensor(X, dtype=t.float32), t.tensor(Y, dtype=t.float32))
        train_data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True) 
        validation_dataset = TensorDataset(t.tensor(X_val, dtype=t.float32), t.tensor(Y_val, dtype=t.float32))
        validation_dataloader = DataLoader(validation_dataset, batch_size=self.hparams.test_batch_size, shuffle=False)

        trainer = self.configure_trainer(train_sample_num)
        trainer.fit(model=self, train_dataloaders=train_data_loader,
                val_dataloaders=validation_dataloader)
        return self

    def estimate(self, X, Y, n_samples=1000) -> float:
        self.eval()
        tmp_model = self.model
        self.model = self.model_ema.module if self.use_ema else self.model

        X = t.tensor(X, dtype=t.float32).to(self.device)
        Y = t.tensor(Y, dtype=t.float32).to(self.device)
        
        with t.no_grad():
            mean_estimate = []
            mean_orthogonal = []
            for _ in range(10):
                nll_x = self.nll(X)
                nll_xy = self.nll(X, Y)
                mi_estimate = nll_x - nll_xy
                mi_estimate_orthogonal = self.estimate_x_y(X, Y)
                mean_estimate.append(mi_estimate)
                mean_orthogonal.append(mi_estimate_orthogonal)
            mi_estimate = t.stack(mean_estimate).mean()
            mi_estimate_orthogonal = t.stack(mean_orthogonal).mean()
        
        self.model = tmp_model
        # For a unified API we return the primary MI estimate as a
        # float. The orthogonal estimate is computed for internal
        # diagnostics but not returned.
        return mi_estimate.item()
    
    @classmethod
    def load_model(cls, checkpoint_path, **kwargs):
        '''
        To load the model, use
        model = MMGEstimator.load_model(checkpoint_path)
        '''
        model = cls.load_from_checkpoint(checkpoint_path, **kwargs)
        return model
    
    def on_save_checkpoint(self, checkpoint):
        if self.use_ema:
            checkpoint['ema_state_dict'] = self.model_ema.state_dict()
        if self.plotter is not None:
            checkpoint['plotter_task_name'] = self.plotter.task_name
            checkpoint['plotter_logger_name'] = self.plotter.logger_name
            checkpoint['plotter_num_bins'] = self.plotter.num_bins
    
    def on_load_checkpoint(self, checkpoint):
        if self.use_ema and 'ema_state_dict' in checkpoint:
            self.model_ema.load_state_dict(checkpoint['ema_state_dict'])
    
        # Plotter relies on an external logger; we no longer restore it
        # by default in this minimal setup.
    @staticmethod
    def logsnr_to_weight(logsnr, loc, scale, clip=4.):
        """
        Convert a given logsnr to its corresponding weight.
        
        Args:
        logsnr (torch.Tensor): The input logsnr value(s)
        loc (float): The location parameter of the logistic distribution
        scale (float): The scale parameter of the logistic distribution
        clip (float): The clipping value, default is 4.0
        
        Returns:
        torch.Tensor: The corresponding weight(s) for the input logsnr
        """
        # Ensure all inputs are tensors on the same device as logsnr
        loc = t.tensor(loc, device=logsnr.device)
        scale = t.tensor(scale, device=logsnr.device)
        clip = t.tensor(clip, device=logsnr.device)
        
        # Calculate the weight using the formula from the original function
        weights = scale * t.tanh(clip / 2) / (t.sigmoid((logsnr - loc)/scale) * t.sigmoid(-(logsnr - loc)/scale))
        
        return weights

    def estimate_alpha(self, x, y, num_logsnr_steps=200):
        self.eval()
        x= t.tensor(x, dtype=t.float32).to(self.device)
        y = t.tensor(y, dtype=t.float32).to(self.device)
        original_shape = x.shape
        x_flat = x.flatten()
        y_flat = y.flatten() if y.dim() > 1 else y   
        loc, scale = 2, 3
        logsnr_min = loc - 3 * scale
        logsnr_max = loc + 3 * scale
        logsnrs = t.linspace(logsnr_min, logsnr_max, num_logsnr_steps, device=self.device)
        weights = self.logsnr_to_weight(logsnrs, self.logsnr_loc, self.logsnr_scale)
        x_repeated = x_flat.repeat(num_logsnr_steps)
        y_repeated = y_flat.repeat(num_logsnr_steps)
        logsnrs_repeated = logsnrs.repeat_interleave(x_flat.shape[0])
        x_reshaped = x_repeated.view(num_logsnr_steps*original_shape[0], *original_shape[1:])
        y_reshaped = y_repeated.view(num_logsnr_steps*original_shape[0], *original_shape[1:])
        logsnrs_reshaped = logsnrs_repeated.view(num_logsnr_steps*original_shape[0], 1)       
        mses = self.mse_x_y(x_reshaped, logsnrs_reshaped, y_reshaped)
        mses_reshaped = mses.view(num_logsnr_steps, -1)
        mses = mses_reshaped.mean(dim=1)
        weighted_mses = weights.unsqueeze(-1) * mses
        mi_estimate = 0.5 * weighted_mses.mean()
        self.plotter.plot_mmse_vs_logsnr_no_bins({'estimate': mses.detach().cpu()}, logsnrs.cpu(), 'alpha', mi_dict={'estimate': mi_estimate.item()}, gt_mi=self.task_gt)
        return mi_estimate.item(),mi_estimate.item()


# Backwards compatibility alias: old name MINDEstimator
MINDEstimator = MMGEstimator


if __name__ == '__main__':
    """Run a simple 1D Gaussian MI estimation demo with MMGEstimator.

    This avoids any dependency on bmi/JAX and mirrors the behaviour
    of our unit tests: we train briefly on a synthetic correlated
    Gaussian task and print the estimated MI and its error.
    """

    import numpy as np
    from sklearn.preprocessing import StandardScaler

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
    x_train, y_train = task.sample(100000, seed=0)
    x_test, y_test = task.sample(10000, seed=1)

    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    x_train_s = x_scaler.transform(x_train)
    y_train_s = y_scaler.transform(y_train)
    x_test_s = x_scaler.transform(x_test)
    y_test_s = y_scaler.transform(y_test)

    model = MMGEstimator(
        x_shape=(task.dim_x,),
        y_shape=(task.dim_y,),
        learning_rate=1e-4,
        batch_size=512,
        max_n_steps=3000,
        seed=0,
        task_name="mmg_demo",
        task_gt=task.mutual_information,
        mi_estimation_interval=200,
        use_ema=True,
        create_checkpoint=False,
    )

    model.fit(x_train_s, y_train_s, x_test_s, y_test_s)
    mi_estimate, _ = model.estimate(x_test_s, y_test_s)
    print(f"MMG estimated MI: {mi_estimate:.4f}, gt={task.mutual_information:.4f}, "
          f"abs_err={abs(mi_estimate - task.mutual_information):.4f}")
