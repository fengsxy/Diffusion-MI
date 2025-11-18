import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from copy import deepcopy
import bmi
from libs.importance import sample_vp_truncated_q, get_normalizing_constant
from _critic import UnetMLP
from libs.SDE import VP_SDE
from libs.util import EMA,concat_vect,deconcat


class MINDEEstimator:
    def __init__(
        self,
        batch_size=256,
        max_n_steps=2000,
        learning_rate=1e-4,
        debias=False,
        use_ema=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.batch_size = batch_size
        self.max_n_steps = max_n_steps
        self.learning_rate = learning_rate
        self.debias = debias
        self.use_ema = use_ema
        self.device = device
        self.score = None
        self.sde = VP_SDE(importance_sampling=self.debias, liklihood_weighting=False)
        self.T = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=False)

    def _create_score(self, dim):
        hidden_dim = 64 if dim <= 10 else 128 if dim <= 50 else 256
        return UnetMLP(dim=dim, init_dim=hidden_dim, dim_mults=(1, 2, 4), time_dim=hidden_dim, nb_mod=2).to(self.device)

    def score_inference(self, x, t, std):
        with torch.no_grad():
            if self.use_ema:
                self.model_ema.module.eval()
                return self.model_ema.module(x, t, std)
            else:
                return self.score(x, t, std)

    def mi_compute_minde_j_sigma(self, data, debias=False, sigma=1.0, eps=1e-5):
        self.sde.device = self.device
        self.score.eval()

        x, y = data["x"], data["y"]
        mods_list = list(data.keys())
        mods_sizes = [data[key].size(1) for key in mods_list]
        nb_mods = len(mods_list)

        t_ = self.sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(self.device) if debias else torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(self.device) * (self.T - eps) + eps
        t_n = t_.expand((x.shape[0], nb_mods))

        Y, _, std, g, mean = self.sde.sample(t_n, data, mods_list)

        y_xy = concat_vect(Y)
        std_xy = concat_vect(std)
        mean_xy = concat_vect(mean)

        mask_time_x = torch.tensor([1, 0]).to(self.device).expand(t_n.size())
        mask_time_y = torch.tensor([0, 1]).to(self.device).expand(t_n.size())

        t_n_x = t_n * mask_time_x
        t_n_y = t_n * mask_time_y

        y_x = concat_vect({"x": Y["x"], "y": y})
        y_y = concat_vect({"x": x, "y": Y["y"]})

        with torch.no_grad():
            a_xy = - self.score_inference(y_xy, t_n, None if debias else std_xy).detach()
            a_x = - self.score_inference(y_x, t_n_x, None if debias else std_xy).detach()
            a_y = - self.score_inference(y_y, t_n_y, None if debias else std_xy).detach()

        M = x.size(0)

        a_x = deconcat(a_x, mods_list, mods_sizes)["x"]
        a_y = deconcat(a_y, mods_list, mods_sizes)["y"]

        chi_t_x = mean["x"] ** 2 * sigma ** 2 + std["x"]**2
        ref_score_x = Y["x"] / chi_t_x

        chi_t_y = mean["y"] ** 2 * sigma ** 2 + std["y"]**2
        ref_score_y = Y["y"] / chi_t_y

        chi_t_xy = mean_xy ** 2 * sigma ** 2 + std_xy**2
        ref_score_xy = y_xy / chi_t_xy

        if debias:
            const = get_normalizing_constant((1,), T=1).to(x)
            e_x = -const * 0.5 * ((a_x + std["x"] * ref_score_x)**2).sum() / M
            e_y = -const * 0.5 * ((a_y + std["y"] * ref_score_y)**2).sum() / M
            e_xy = -const * 0.5 * ((a_xy + std_xy * ref_score_xy)**2).sum() / M
        else:
            g = g["x"].reshape(t_.shape)
            e_x = -0.5 * (g**2 * (a_x + ref_score_x)**2).sum() / M
            e_y = -0.5 * (g**2 * (a_y + ref_score_y)**2).sum() / M          
            e_xy = -0.5 * (g**2 * (a_xy + ref_score_xy)**2).sum() / M

        return e_xy - e_x - e_y

    def mi_compute_minde_c(self, data, debias=False, eps=1e-3):
        self.sde.device = self.device
        self.score.eval()
        x, y = data["x"], data["y"]
        
        mods_list = list(data.keys())
        nb_mods = len(mods_list)
        
        t_ = self.sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(self.device) if debias else torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(self.device) * (self.T - eps) + eps
        
        t_n = t_.expand((x.shape[0], nb_mods))
        
        Y, _, std, g, _ = self.sde.sample(t_n, data, mods_list)
        
        std_x = std["x"]
        
        y_x = concat_vect({"x": Y["x"], "y": torch.zeros_like(Y["y"])})
        y_xc = concat_vect({"x": Y["x"], "y": data["y"]})
        
        mask_time_x = torch.tensor([1, 0]).to(self.device).expand(t_n.size())
        t_n_x = t_n * mask_time_x + 0.0 * (1 - mask_time_x)
        t_n_c = t_n * mask_time_x + 1.0 * (1 - mask_time_x)
        
        with torch.no_grad():
            a_x = -self.score_inference(y_x, t_n_x, None if debias else std_x).detach()
            a_xy = -self.score_inference(y_xc, t_n_c, None if debias else std_x).detach()
        
        M = x.size(0)
        
        if debias:
            const = get_normalizing_constant((1,), T=1).to(x)
            est_score = const * 0.5 * ((a_x - a_xy)**2).sum() / M
        else:
            g = g["x"].reshape(g["x"].size(0), 1)
            est_score = 0.5 * (g**2 * (a_x - a_xy)**2).sum() / M
        
        return est_score.detach()

    def mi_compute_minde_j(self, data, debias=False, eps=1e-3):
        self.sde.device = self.device
        self.score.eval()
        x, y = data["x"], data["y"]
        
        mods_list = list(data.keys())
        mods_sizes = [data[key].size(1) for key in mods_list]
        nb_mods = len(mods_list)
        
        t_ = self.sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(self.device) if debias else torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(self.device) * (self.T - eps) + eps
        
        t_n = t_.expand((x.shape[0], nb_mods))
        
        Y, _, std, g, _ = self.sde.sample(t_n, data, mods_list)
        
        y_xy = concat_vect(Y)
        std_xy = concat_vect(std)
        
        mask_time_x = torch.tensor([1, 0]).to(self.device).expand(t_n.size())
        mask_time_y = torch.tensor([0, 1]).to(self.device).expand(t_n.size())
        
        t_n_x = t_n * mask_time_x
        t_n_y = t_n * mask_time_y
        
        y_x = concat_vect({"x": Y["x"], "y": y})
        y_y = concat_vect({"x": x, "y": Y["y"]})
        
        with torch.no_grad():
            a_xy = -self.score_inference(y_xy, t_n, None if debias else std_xy).detach()
            a_x = -self.score_inference(y_x, t_n_x, None if debias else std_xy).detach()
            a_y = -self.score_inference(y_y, t_n_y, None if debias else std_xy).detach()
        
        M = x.size(0)
        
        a_x = deconcat(a_x, mods_list, mods_sizes)
        a_y = deconcat(a_y, mods_list, mods_sizes)
        
        a_cond = concat_vect({"x": a_x["x"], "y": a_y["y"]})
        
        if debias:
            const = get_normalizing_constant((1,)).to(x)
            est_score = const * 0.5 * ((a_xy - a_cond)**2).sum() / M
        else:
            g = g["x"].reshape(t_.shape)
            est_score = 0.5 * (g**2 * (a_xy - a_cond)**2).sum() / M * self.T
        
        return est_score.detach()

    def mi_compute_minde_c_sigma(self, data, debias=False, sigma=1.0, eps=1e-3):
        self.sde.device = self.device
        self.score.eval()
        x, y = data["x"], data["y"]
        
        mods_list = list(data.keys())
        nb_mods = len(mods_list)
        
        t_ = self.sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(self.device) if debias else torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(self.device) * (self.T - eps) + eps
        
        t_n = t_.expand((x.shape[0], nb_mods))
        
        Y, _, std, g, mean = self.sde.sample(t_n, data, mods_list)
        
        std_x = std["x"]
        mean_x = mean["x"]
        
        y_x = concat_vect({"x": Y["x"], "y": torch.zeros_like(Y["y"])})
        y_xc = concat_vect({"x": Y["x"], "y": data["y"]})
        
        mask_time_x = torch.tensor([1, 0]).to(self.device).expand(t_n.size())
        t_n_x = t_n * mask_time_x + 0.0 * (1 - mask_time_x)
        t_n_c = t_n * mask_time_x + 1.0 * (1 - mask_time_x)
        
        with torch.no_grad():
            a_x = -self.score_inference(y_x, t_n_x, None if debias else std_x).detach()
            a_xy = -self.score_inference(y_xc, t_n_c, None if debias else std_x).detach()
        
        M = x.size(0)
        
        chi_t_x = mean_x ** 2 * sigma ** 2 + std_x**2
        ref_score_x = Y["x"] / chi_t_x
        
        if debias:
            const = get_normalizing_constant((1,), T=1-eps).to(x)
            e_x = -const * 0.5 * ((a_x + std_x * ref_score_x)**2).sum() / M
            e_xc = -const * 0.5 * ((a_xy + std_x * ref_score_x)**2).sum() / M
        else:
            g = g["x"].reshape(g["x"].size(0), 1)
            e_x = -0.5 * (g**2 * (a_x + ref_score_x)**2).sum() / M
            e_xc = -0.5 * (g**2 * (a_xy + ref_score_x)**2).sum() / M
        
        return e_x - e_xc

    def fit(self, X: np.ndarray, Y: np.ndarray,method='minde_c_sigma'):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        if self.score is None:
            self.score = self._create_score(X.shape[1] + Y.shape[1])
            if self.use_ema:
                self.model_ema = EMA(self.score, decay=0.999)

        optimizer = optim.Adam(self.score.parameters(), lr=self.learning_rate)
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        pbar = tqdm(total=self.max_n_steps, desc="Training MINDE")
        steps = 0
        while steps < self.max_n_steps:
            for x_batch, y_batch in dataloader:
                if steps >= self.max_n_steps:
                    break
                
                batch = {"x": x_batch, "y": y_batch}
                if method in ['minde_c', 'minde_c_sigma']:
                    loss = self.sde.train_step_cond(batch, self.score, d=0.5).mean()
                else:  # minde_j, minde_j_sigma
                    loss = self.sde.train_step(batch, self.score, d=0.5).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.use_ema:
                    self.model_ema.update(self.score)

                steps += 1
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})

                if steps >= self.max_n_steps:
                    break

        pbar.close()

    def estimate(self, X: np.ndarray, Y: np.ndarray, method='minde_c_sigma') -> float:
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            Y = torch.tensor(Y, dtype=torch.float32).to(self.device)
            data = {"x": X, "y": Y}
            
            if method == 'minde_c':
                mi_estimate = self.mi_compute_minde_c(data, debias=self.debias)
            elif method == 'minde_j':
                mi_estimate = self.mi_compute_minde_j(data, debias=self.debias)
            elif method == 'minde_c_sigma':
                mi_estimate = self.mi_compute_minde_c_sigma(data, debias=self.debias)
            elif method == 'minde_j_sigma':
                mi_estimate = self.mi_compute_minde_j_sigma(data, debias=self.debias)
            else:
                raise ValueError(f"Unknown method: {method}")
            
        return mi_estimate.item()

if __name__ == '__main__':
    import bmi

    task = bmi.benchmark.BENCHMARK_TASKS['1v1-normal-0.75']
    print(f"Task: {task.name}")
    print(f"Task {task.name} with dimensions {task.dim_x} and {task.dim_y}")
    print(f"Ground truth mutual information: {task.mutual_information:.2f}")

    X, Y = task.sample(100000, seed=42)
    X = X.__array__()
    Y = Y.__array__()

    methods = ['minde_j', 'minde_j_sigma']

    results = {}
    max_n_steps=10000
    minde = MINDEEstimator(debias=True, max_n_steps=max_n_steps, batch_size=128, learning_rate=1e-4, use_ema=True)
    minde.fit(X, Y,method='minde_j')

    for method in methods:
        error_list = []
        X, Y = task.sample(10000, seed=42)
        mi_estimate = minde.estimate(X, Y, method=method)
        error = abs(mi_estimate - task.mutual_information)
        error_list.append(error)
        print(f"Method: {method}, Steps: {max_n_steps}, MINDE estimate: {mi_estimate:.4f}, Absolute error: {error:.4f}")
        results[method] = error_list
    print(results)