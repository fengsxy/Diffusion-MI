import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import Dataset ,DataLoader


class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from timm"""
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
        
def pop_elem_i(encodings , i =[]  ):
    encodings = encodings.copy()
    return{
        key : encodings[key] for key in encodings.keys() if ( key in i ) == False
    } 
    
def deconcat(z,var_list,sizes):
    data = torch.split(z, sizes, dim=1)
    return {var: data[i] for i, var in enumerate(var_list)}



def concat_vect(encodings):
    return torch.cat(list(encodings.values()),dim = -1)
    


def unsequeeze_dict(data):
        for key in data.keys():
            if data[key].ndim == 1 :
                data[key]= data[key].view(data[key].size(0),1)
        return data


def cond_x_data(x_t,data,mod):

    x = x_t.copy()
    for k in x.keys():
        if k !=mod:
            x[k]=data[k] 
    return x



def marginalize_data(x_t, mod,fill_zeros =False):
    x = x_t.copy()
    for k in x.keys():
        if k !=mod:
            if fill_zeros:
                x[k]=torch.zeros_like(x_t[k] ) 
            else:
                x[k]=torch.randn_like(x_t[k] )
    return x


def marginalize_one_var(x_t, mod,fill_zeros =False):
    x = x_t.copy()
    for k in x.keys():
        if k ==mod:
            if fill_zeros:
                x[k]=torch.zeros_like(x_t[k] ) 
            else:
                x[k]=torch.randn_like(x_t[k] )
    return x


def minus_x_data(x_t, mod,fill_zeros=True):
        x = x_t.copy()
        for k in x.keys():
                if k ==mod:
                    if fill_zeros:
                        x[k]=torch.zeros_like(x_t[k] ) 
                    else:
                        x[k]=torch.rand_like(x_t[k] )
        return x


def expand_mask(mask, var_sizes):
        return torch.cat([
            mask[:, i].view(mask.shape[0], 1).expand(mask.shape[0], size) for i, size in enumerate(var_sizes)
        ], dim=1)
        
        
def get_samples(test_loader,device,N=10000):
    
    var_list = list(test_loader.dataset[0].keys())
    
    data = {var: torch.Tensor().to(device) for var in var_list}
    for batch in test_loader:
            for var in var_list:
                data[var] = torch.cat([data[var], batch[var].to(device)])
    return {var: data[var][:N,:] for var in var_list}


class SynthetitcDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data):
        self.x = data [0]
        self.y = data [1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"x":torch.FloatTensor(self.x[idx]) ,"y":torch.FloatTensor(self.y[idx]) }
    

import torch as t
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

class SNRMMSEPlotter:
    def __init__(self, model=None, task_name=None, logger=None, logger_name=None, num_bins=1000):
        self.model = model
        self.task_name = task_name
        self.logger = logger
        self.logger_name = logger_name
        self.num_bins = num_bins

    def plot_improved_snr_mse(self, x, y, gt_mi=None, tag=None):
        self.model.eval()
        #to tensor
        x = x.to(self.model.device)
        y = y.to(self.model.device)

        logsnr, weights = self.model.logistic_integrate(len(x))
        
        with t.no_grad():
            mse_x = self.model.mse(x, logsnr)
            mse_x_y = self.model.mse(x, logsnr, y)
            mses_o = self.model.mse_x_y(x, logsnr, y)
            mses = mse_x - mse_x_y

        logsnr, mses_o, mses, mse_x, mse_x_y = map(lambda t: t.cpu().numpy(), (logsnr, mses_o, mses, mse_x, mse_x_y))
        
        bin_edges, bin_centers = self._create_bins(logsnr, 100)
        
        mses_o_binned = self._bin_data(logsnr, mses_o, bin_edges)
        mses_binned = self._bin_data(logsnr, mses, bin_edges)
        mse_x_binned = self._bin_data(logsnr, mse_x, bin_edges)
        mse_x_y_binned = self._bin_data(logsnr, mse_x_y, bin_edges)

        mi_estimate, mi_orthogonal = self.model.estimate(x.cpu().numpy(), y.cpu().numpy())

        self._plot_mse_vs_logsnr(bin_centers, mses_o_binned, mses_binned, mi_estimate, mi_orthogonal, gt_mi, tag)
        self._plot_mse_x_and_mse_x_y(bin_centers, mse_x_binned, mse_x_y_binned, mi_estimate, mi_orthogonal, gt_mi, tag)

    def plot_mmse_vs_logsnr(self, mmse_dict, logsnr, tag, mi_dict=None, gt_mi=None, weights=None):
        plt.figure(figsize=(12, 8))
        
        bin_edges, bin_centers = self._create_bins(logsnr)
        colors = self._generate_color_cycle(len(mmse_dict) + 1)
        
        mmse_bin_list = self._plot_mmse_lines(mmse_dict, logsnr, bin_edges, bin_centers, colors, weights)
        self._plot_max_mmse(mmse_bin_list, bin_centers, colors[-1])
        
        self._set_plot_attributes(f'MSE vs log SNR {self.task_name}_{tag}')
        self._add_mi_annotations(mi_dict, gt_mi)
        self._save_and_log_figure(f'MSE_vs_logsnr_{self.task_name}_{tag}')

    def plot_mmse_vs_logsnr_no_bins(self, mmse_dict, logsnr, tag, mi_dict=None, gt_mi=None):
        plt.figure(figsize=(12, 8))
        
        colors = self._generate_color_cycle(len(mmse_dict) + 1)
        
        self._plot_mmse_lines_no_bins(mmse_dict, logsnr, colors)
        self._plot_max_mmse_no_bins(mmse_dict, logsnr, colors[-1])
        
        self._set_plot_attributes(f'MSE vs log SNR (No Binning) {self.task_name}_{tag}')
        self._add_mi_annotations(mi_dict, gt_mi)
        self._save_and_log_figure(f'MSE_vs_logsnr_no_bins_{self.task_name}_{tag}')

    def _plot_mmse_lines_no_bins(self, mmse_dict, logsnr, colors):
        for (label, mmse), color in zip(mmse_dict.items(), colors):
            plt.plot(logsnr, mmse, label=label, color=color, alpha=0.6)

    def _plot_max_mmse_no_bins(self, mmse_dict, logsnr, color):
        max_mmse = np.max(list(mmse_dict.values()), axis=0)
        plt.plot(logsnr, max_mmse, label='max_mmse', color=color, linestyle='--')

    def _create_bins(self, data, num_bins=None):
        num_bins = num_bins or self.num_bins
        bin_edges = np.linspace(data.min(), data.max(), num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_edges, bin_centers

    def _bin_data(self, x, y, bin_edges):
        return binned_statistic(x, y, statistic='mean', bins=bin_edges)[0]

    def _generate_color_cycle(self, num_colors):
        return plt.cm.rainbow(np.linspace(0, 1, num_colors))

    def _plot_mmse_lines(self, mmse_dict, logsnr, bin_edges, bin_centers, colors, weights):
        mmse_bin_list = []
        for (label, mmse), color in zip(mmse_dict.items(), colors):
            mmse_binned = self._bin_data(logsnr, mmse, bin_edges)
            if weights is not None:
                weights_binned = self._bin_data(logsnr, weights, bin_edges)
                mmse_binned *= weights_binned  # Apply weights if provided
            mmse_bin_list.append(mmse_binned)
            plt.plot(bin_centers, mmse_binned, label=label, color=color)
        return mmse_bin_list

    def _plot_max_mmse(self, mmse_bin_list, bin_centers, color):
        max_mmse_bin = np.max(mmse_bin_list, axis=0)
        plt.plot(bin_centers, max_mmse_bin, label='max_mmse', color=color, linestyle='--')

    def _plot_mse_vs_logsnr(self, bin_centers, mses_o_binned, mses_binned, mi_estimate, mi_orthogonal, gt_mi, tag):
        plt.figure(figsize=(12, 8))
        plt.plot(bin_centers, mses_o_binned, label=r'$E[(\hat{\varepsilon}_\alpha(x))^2 - \hat{\varepsilon}_\alpha(x|y^*))^2]$', color='skyblue')
        plt.plot(bin_centers, mses_binned, label=r'$E[(\varepsilon - \hat{\varepsilon}_\alpha(x))^2] - E[(\varepsilon - \hat{\varepsilon}_\alpha(x|y^*))^2]$', color='lightcoral')
        self._set_plot_attributes(f'MSE vs log SNR {self.task_name}_{tag}')
        self._add_text_annotations({'MI Estimate': mi_estimate, 'MI Orthogonal': mi_orthogonal}, gt_mi)
        self._save_and_log_figure(f'MSE_vs_logsnr_{self.task_name}_{tag}')

    def _plot_mse_x_and_mse_x_y(self, bin_centers, mse_x_binned, mse_x_y_binned, mi_estimate, mi_orthogonal, gt_mi, tag):
        plt.figure(figsize=(12, 8))
        plt.plot(bin_centers, mse_x_binned, label=r'$E[(\varepsilon - \hat{\varepsilon}_\alpha(x))^2]$', color='skyblue')
        plt.plot(bin_centers, mse_x_y_binned, label=r'$E[(\varepsilon - \hat{\varepsilon}_\alpha(x|y^*))^2]$', color='lightcoral')
        self._set_plot_attributes(f'MSE x and MSE x|y vs log SNR {self.task_name}_{tag}')
        self._add_text_annotations({'MI Estimate': mi_estimate, 'MI Orthogonal': mi_orthogonal}, gt_mi)
        self._save_and_log_figure(f'MSE_x_MSE_x_y_vs_logsnr_{self.task_name}_{tag}')

    def _set_plot_attributes(self, title):
        plt.xlabel('log SNR')
        plt.ylabel('bits')
        plt.title(title)
        plt.legend()
        plt.grid(True)

    def _add_text_annotations(self, mi_dict, gt_mi):
        for i, (mi_name, mi_value) in enumerate(mi_dict.items()):
            plt.text(0.05, 0.95 - i*0.05, f'{mi_name}: {mi_value:.4f}', transform=plt.gca().transAxes, verticalalignment='top')
        if gt_mi is not None:
            plt.text(0.05, 0.95 - len(mi_dict)*0.05, f'GT MI: {gt_mi:.4f}', transform=plt.gca().transAxes, verticalalignment='top')

    def _save_and_log_figure(self, filename):
        plt.savefig(f'./{filename}_{self.logger_name}.png')
        if self.logger:
            self.logger.experiment.add_figure(filename, plt.gcf(), self.model.global_step)
        plt.close()
    
    def _add_mi_annotations(self, mi_dict, gt_mi):
        if mi_dict:
            for i, (mi_name, mi_value) in enumerate(mi_dict.items()):
                plt.text(0.05, 0.95 - i*0.05, f'{mi_name}: {mi_value:.4f}', transform=plt.gca().transAxes, verticalalignment='top')
        if gt_mi is not None:
            plt.text(0.05, 0.95 - (len(mi_dict) if mi_dict else 0)*0.05, f'GT MI: {gt_mi:.4f}', transform=plt.gca().transAxes, verticalalignment='top')
