from __future__ import annotations

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing

from .libs.SDE import VP_SDE
from .libs.util import EMA, concat_vect, deconcat, marginalize_data, cond_x_data, SynthetitcDataset
from .libs.info_measures import mi_cond, mi_cond_sigma, mi_joint, mi_joint_sigma
from ._critic import UnetMLP_simple
import json
import argparse

class MINDEEstimator(pl.LightningModule):
    def __init__(self,
                 x_shape=None,
                 y_shape=None,
                 learning_rate=5e-5,
                 batch_size=64,
                 max_n_steps=None,
                 max_epochs=None,
                 hidden_dim=None,
                 smoothing_alpha=0.01,
                 device=  "cuda" if torch.cuda.is_available() else "cpu",
                 seed=42,
                 task_name="default",
                 task_gt=-1,
                 test_num=10000,
                 mi_estimation_interval=500,
                 use_ema=True,
                 ema_decay=0.999,
                 arch="mlp",
                 type="c",
                 sigma=0.1,
                 mc_iter=10,
                 preprocessing="rescale",
                 importance_sampling=True,
                 early_stopping=False,
                 early_stopping_params=None,
                 create_checkpoint=False,
                 checkpoint_params=None,
                 # Backward compatibility: accepted but unused.
                 log_to_tensorboard: bool = False):
        super(MINDEEstimator, self).__init__()
        self.save_hyperparameters()

        # Lazily initialized components that depend on x_shape/y_shape.
        self.var_list = ["x0", "x1"]
        self.sizes = None
        self.score = None
        self.model_ema = None
        self.use_ema = use_ema
        self.sde = None

        self.smoothed_mi_history = []
        
        if early_stopping and early_stopping_params is None:
            self.early_stopping_params = {
                'monitor': 'train_loss',
                'mode': 'min',
                'patience': 2
            }
        if create_checkpoint and checkpoint_params is None:
            self.checkpoint_params = { 
                'dirpath': f'checkpoints/{task_name}',
                'save_last': True,
                'save_top_k': 1,
                'monitor': 'train_loss',
                'mode': 'min',
            }
        
        self.logger_name = f"minde_{task_name}_seed_{seed}_lr_{learning_rate}"
        if max_n_steps:
            self.logger_name += f"_max_steps_{max_n_steps}"
        if max_epochs:
            self.logger_name += f"_max_epochs_{max_epochs}"
        self.arch = arch
        #if seed is not None:
            #pl.seed_everything(seed, workers=True)

        

    def calculate_hidden_dim(self):
        dim = np.sum(self.sizes)
        if dim <= 10:
            return 64
        elif dim <= 50:
            return 128
        else:
            return 256

    @classmethod
    def load_model(cls, checkpoint_path, **kwargs):
        model = cls.load_from_checkpoint(checkpoint_path, **kwargs)
        return model

    def configure_optimizers(self):
        return torch.optim.Adam(self.score.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        loss = self.sde.train_step(batch, self.score_forward).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.sde.train_step(batch, self.score_forward).mean()
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        if self.current_epoch % self.hparams.mi_estimation_interval == 0:
            mi, mi_sigma = self.compute_mi()
            self.log("mi_estimate", mi)
            self.log("mi_sigma_estimate", mi_sigma)
            self.smoothed_mi_history.append(mi)

    def fit(self, X: np.ndarray, Y: np.ndarray, X_val=None, Y_val=None):
        # Infer shapes and lazily initialize networks / SDE at first fit.
        if self.hparams.x_shape is None or self.hparams.y_shape is None:
            self.hparams.x_shape = X.shape[1:]
            self.hparams.y_shape = Y.shape[1:]

        if self.sizes is None or self.score is None or self.sde is None:
            self.sizes = [np.prod(self.hparams.x_shape), np.prod(self.hparams.y_shape)]
            hidden_dim = self.calculate_hidden_dim()
            if self.arch == "mlp":
                self.score = UnetMLP_simple(
                    dim=np.sum(self.sizes),
                    init_dim=hidden_dim,
                    dim_mults=[],
                    time_dim=hidden_dim,
                    nb_var=len(self.var_list),
                )
            else:
                raise NotImplementedError

            self.model_ema = EMA(self.score, decay=self.hparams.ema_decay) if self.use_ema else None
            self.sde = VP_SDE(importance_sampling=True, var_sizes=self.sizes, type=self.hparams.type)

        if self.hparams.preprocessing == "rescale":
            X = preprocessing.StandardScaler(copy=True).fit_transform(X)
            Y = preprocessing.StandardScaler(copy=True).fit_transform(Y)
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        dataset = SynthetitcDataset([X, Y])

        if X_val is not None and Y_val is not None:
            # Preprocess and convert test samples to torch tensors so that
            # ``compute_mi`` can safely call ``.to(self.device)`` on them.
            if self.hparams.preprocessing == "rescale":
                X_test_p = preprocessing.StandardScaler(copy=True).fit_transform(X_val)
                Y_test_p = preprocessing.StandardScaler(copy=True).fit_transform(Y_val)
            else:
                X_test_p, Y_test_p = X_val, Y_val

            self.test_samples = {
                self.var_list[0]: torch.tensor(X_test_p, dtype=torch.float32),
                self.var_list[1]: torch.tensor(Y_test_p, dtype=torch.float32),
            }
        else:
            # Use a subset of the (already torch) training data for MI
            # estimation if no explicit test set is provided.
            self.test_samples = {
                self.var_list[0]: X[:self.hparams.test_num],
                self.var_list[1]: Y[:self.hparams.test_num],
            }
        train_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)

        early_stop_callback = EarlyStopping(**self.early_stopping_params) if self.hparams.early_stopping else None
        checkpoint_callback = ModelCheckpoint(**self.checkpoint_params) if self.hparams.create_checkpoint else None

        callbacks = [cb for cb in [early_stop_callback, checkpoint_callback] if cb is not None]

        trainer = pl.Trainer(
            logger=False,
            callbacks=callbacks,
            max_epochs=self.hparams.max_epochs,
            # Disable Lightning's default checkpointing; explicit
            # ModelCheckpoint callbacks above are still honored when
            # ``create_checkpoint`` is True.
            enable_checkpointing=self.hparams.create_checkpoint,
            # Default to single-device (no DDP) training unless
            # overridden externally.
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
        )

        trainer.fit(self, train_loader)

    def estimate(self, X: np.ndarray, Y: np.ndarray) -> float:
        if self.hparams.preprocessing == "rescale":
            X = preprocessing.StandardScaler(copy=True).fit_transform(X)
            Y = preprocessing.StandardScaler(copy=True).fit_transform(Y)
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        
        mi, mi_sigma = self.compute_mi(data={self.var_list[0]: X, self.var_list[1]: Y})
        print(f"Final estimate - MI: {mi:.4f}, MI (sigma): {mi_sigma:.4f}")
        return mi

    def score_forward(self, x, t=None, mask=None, std=None):
        if self.hparams.arch == "mlp":
            t = t.expand(t.shape[0], mask.size(-1)) 
            marg = (-mask).clip(0, 1)
            cond = 1 - (mask.clip(0, 1)) - marg
            t = t * (1 - cond) + 0.0 * cond
            t = t * (1 - marg) + 1 * marg
            return self.score(x, t=t, std=std)

    def score_inference(self, x, t=None, mask=None, std=None):
        """
        Perform score inference on the input data.

        Args:
            x (torch.Tensor): Concatenated variables.
            t (torch.Tensor, optional): The time t. 
            mask (torch.Tensor, optional): The mask data.
            std (torch.Tensor, optional): The standard deviation to rescale the network output.

        Returns:
            torch.Tensor: The output score function (noise/std) if std !=None , else return noise .
        """
        # Get the model to use for inference, use the ema model if use_ema is set to True

        score = self.model_ema.module if self.use_ema else self.score
        with torch.no_grad():
            score.eval()
            
            if self.args.arch == "mlp":
                t = t.expand(t.shape[0],mask.size(-1)) 
          
                marg = (- mask).clip(0, 1) ## max <0 
                cond = 1 - (mask.clip(0, 1)) - marg  ##mask ==0
             
                t = t * (1- cond)  + 0.0 * cond
                t = t* (1-marg) + 1 * marg

                
                return score(x, t=t, std=std)

    def compute_mi(self, data=None, eps=1e-5):
        self.eval()
        self.to( "cuda" if torch.cuda.is_available() else "cpu")
        if data is None:
            data = self.test_samples
        data_0 = {x_i: data[x_i].to(self.device) for x_i in self.var_list}
        z_0 = concat_vect(data_0)

        M = z_0.shape[0]
        mi = []
        mi_sigma = []
        
        marg_masks, cond_mask = self.get_masks()

        for _ in range(self.hparams.mc_iter):
            if self.hparams.importance_sampling:
                t = self.sde.sample_importance_sampling_t((M, 1)).to(self.device)
            else:
                t = ((self.sde.T - eps) * torch.rand((M, 1)) + eps).to(self.device)
            
            _, g = self.sde.sde(t)
            z_t, _, mean, std = self.sde.sample(z_0, t=t)
            
            std_w = None if self.hparams.importance_sampling else std 
            z_t = deconcat(z_t, self.var_list, self.sizes)
            
            if self.hparams.type == "c":
                s_marg, s_cond = self.infer_scores(z_t, t, data_0, std_w, marg_masks, cond_mask)
                mi.append(mi_cond(s_marg=s_marg, s_cond=s_cond, g=g, importance_sampling=self.hparams.importance_sampling))
                mi_sigma.append(mi_cond_sigma(s_marg=s_marg, s_cond=s_cond, g=g, mean=mean, std=std, x_t=z_t[self.var_list[0]], sigma=self.hparams.sigma, importance_sampling=self.hparams.importance_sampling))
            elif self.hparams.type == "j":
                s_joint, s_cond_x, s_cond_y = self.infer_scores(z_t, t, data_0, std_w, marg_masks, cond_mask)
                mi.append(mi_joint(s_joint=s_joint, s_cond_x=s_cond_x, s_cond_y=s_cond_y, g=g, importance_sampling=self.hparams.importance_sampling))
                mi_sigma.append(mi_joint_sigma(s_joint=s_joint, s_cond_x=s_cond_x, s_cond_y=s_cond_y, x_t=z_t[self.var_list[0]], y_t=z_t[self.var_list[1]], g=g, mean=mean, std=std, sigma=self.hparams.sigma, importance_sampling=self.hparams.importance_sampling))

        return np.mean(mi), np.mean(mi_sigma)

    def get_masks(self):
        return {self.var_list[0]: torch.tensor([1,-1]).to(self.device),
                self.var_list[1]: torch.tensor([-1,1]).to(self.device)
               }, {self.var_list[0]: torch.tensor([1,0]).to(self.device),
                   self.var_list[1]: torch.tensor([0,1]).to(self.device)
                  }

    def infer_scores(self, z_t, t, data_0, std_w, marg_masks, cond_mask):
        with torch.no_grad():
            if self.hparams.type == "c":
                marg_x = concat_vect(marginalize_data(z_t, self.var_list[0], fill_zeros=True))
                cond_x = concat_vect(cond_x_data(z_t, data_0, self.var_list[0]))
                s_marg = -self.score_inference(marg_x, t=t, mask=marg_masks[self.var_list[0]], std=std_w)
                s_cond = -self.score_inference(cond_x, t=t, mask=cond_mask[self.var_list[0]], std=std_w)
                return deconcat(s_marg, self.var_list, self.sizes)[self.var_list[0]], deconcat(s_cond, self.var_list, self.sizes)[self.var_list[0]]
            elif self.hparams.type == "j":
                s_joint = -self.score_inference(concat_vect(z_t), t=t, std=std_w, mask=torch.ones_like(marg_masks[self.var_list[0]]))
                cond_x = concat_vect(cond_x_data(z_t, data_0, self.var_list[0]))
                cond_y = concat_vect(cond_x_data(z_t, data_0, self.var_list[1]))
                s_cond_x = -self.score_inference(cond_x, t=t, mask=cond_mask[self.var_list[0]], std=std_w)
                s_cond_y = -self.score_inference(cond_y, t=t, mask=cond_mask[self.var_list[1]], std=std_w)
                return s_joint, deconcat(s_cond_x, self.var_list, self.sizes)[self.var_list[0]], deconcat(s_cond_y, self.var_list, self.sizes)[self.var_list[1]]
    
    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.model_ema:
            self.model_ema.update(self.score)
    
    def score_inference(self, x, t=None, mask=None, std=None):
        """
        Perform score inference on the input data.

        Args:
            x (torch.Tensor): Concatenated variables.
            t (torch.Tensor, optional): The time t. 
            mask (torch.Tensor, optional): The mask data.
            std (torch.Tensor, optional): The standard deviation to rescale the network output.

        Returns:
            torch.Tensor: The output score function (noise/std) if std !=None , else return noise .
        """
        # Get the model to use for inference, use the ema model if use_ema is set to True

        score = self.model_ema.module if self.use_ema else self.score
        with torch.no_grad():
            score.eval()
            
            if self.arch == "mlp":
                t = t.expand(t.shape[0],mask.size(-1)) 
          
                marg = (- mask).clip(0, 1) ## max <0 
                cond = 1 - (mask.clip(0, 1)) - marg  ##mask ==0
             
                t = t * (1- cond)  + 0.0 * cond
                t = t* (1-marg) + 1 * marg

                
                return score(x, t=t, std=std)
            
def append_results(result_dict, filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            results = json.load(f)
    else:
        results = []
    results.append(result_dict)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    """Simple demo: MINDE (SDE) on a synthetic 1D Gaussian task."""

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
    X, Y = task.sample(100000, seed=42)
    X_test, Y_test = task.sample(10000, seed=0)

    minde = MINDEEstimator(
        x_shape=(1,),
        y_shape=(1,),
        learning_rate=1e-4,
        batch_size=256,
        max_n_steps=3000,
        early_stopping=True
    )

    minde.fit(X, Y)
    mi_estimate = minde.estimate(X_test, Y_test)

    print(
        f"MINDE (SDE) estimated MI: {mi_estimate:.4f}, "
        f"gt={task.mutual_information:.4f}, "
        f"abs_err={abs(mi_estimate - task.mutual_information):.4f}"
    )
