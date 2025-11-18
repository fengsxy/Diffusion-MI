from __future__ import print_function, division
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
from torch import nn
import torch.nn.functional as F
from functools import partial

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))
    
def mlp(dim, hidden_dim, output_dim, layers, activation):
    """Create a mlp"""
    activation = {
        'relu': nn.ReLU
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)



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

class ConvCritic(nn.Module):
    def __init__(self, x_channels, y_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(x_channels + y_channels, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.fc = nn.Linear(128 * 16 * 16, 1024)  # 假设输入图像大小为64x64
        self.output = nn.Linear(1024, 1)

    def forward(self, x, y):
        combined = torch.cat([x, y], dim=1)
        h = torch.relu(self.conv1(combined))
        h = torch.relu(self.conv2(h))
        h = h.view(h.size(0), -1)
        h = torch.relu(self.fc(h))
        return self.output(h)

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


class ConcatCritic(nn.Module):
    """Concat critic, where the inputs are concatenated and reshaped in a squared matrix."""
    def __init__(self, dim, hidden_dim, layers, activation, divergence):
        super(ConcatCritic, self).__init__()
        # output is scalar score
        self._f = mlp(dim * 2, hidden_dim, 1, layers, activation)
        if divergence == "GAN":
            self.last_activation = nn.Sigmoid()
        elif divergence == "KL" or divergence == "HD":
            self.last_activation = nn.Softplus()
        else:
            self.last_activation = nn.Identity()

    def forward(self, x, y):
        batch_size = x.shape[0]
        # Create all the possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [batch_size * batch_size, -1])
        scores = self._f(xy_pairs)
        out = torch.reshape(scores, [batch_size, batch_size]).t()
        out = self.last_activation(out)
        return out


class SeparableCritic(nn.Module):
    """Separable critic, where the output value is the inner product between the outputs of g(x) and h(y)."""
    def __init__(self, x_dim,y_dim, hidden_dim, embed_dim, layers, activation, divergence):
        super(SeparableCritic, self).__init__()
        self._g = mlp(x_dim, hidden_dim, embed_dim, layers, activation)
        self._h = mlp(y_dim, hidden_dim, embed_dim, layers, activation)
        if divergence == "GAN":
            self.last_activation = nn.Sigmoid()
        elif divergence == "KL" or divergence == "HD":
            self.last_activation = nn.Softplus()
        else:
            self.last_activation = nn.Identity()

    def forward(self, x, y):
        g_x = self._g(x)  # Shape: (batch_size, embed_dim)
        h_y = self._h(y)  # Shape: (batch_size, embed_dim)

        # Compute pairwise inner products
        scores = torch.mm(g_x, h_y.t())  # Shape: (batch_size, batch_size)
        
        return self.last_activation(scores)


class ConvolutionalCritic(nn.Module):
    """Convolutional critic, used for the consistency tests"""
    def __init__(self, divergence, test_type):
        super(ConvolutionalCritic, self).__init__()
        if "bs" in test_type:
            n_ch_input = 2
        else:
            n_ch_input = 4
        self.conv = nn.Sequential(
            nn.Conv2d(n_ch_input, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.lin = nn.Sequential(
            nn.Linear(6272, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        if divergence == "GAN":
            self.last_activation = nn.Sigmoid()
        elif divergence == "KL" or divergence == "HD":
            self.last_activation = nn.Softplus()
        else:
            self.last_activation = nn.Identity()

    def forward(self, x, y):
        batch_size = x.size(0)
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        cat_pairs = torch.cat((x_tiled, y_tiled), dim=2)
        xy_pairs = torch.reshape(cat_pairs, [batch_size * batch_size, -1, 28, 28])
        scores_tmp = self.conv(xy_pairs)
        flattened_scores_tmp = torch.flatten(scores_tmp, start_dim=1)
        out = self.lin(flattened_scores_tmp)
        out = torch.reshape(out, [batch_size, batch_size]).t()
        out = self.last_activation(out)
        return out

class UnetMLP(nn.Module):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 1), resnet_block_groups=8, time_dim=128, nb_mod=1):
        super().__init__()
        self.nb_mod = nb_mod
        init_dim = init_dim or dim
        self.init_lin = nn.Linear(dim, init_dim)

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        self.time_mlp = nn.Sequential(
            nn.Linear(nb_mod, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                nn.Linear(dim_in, dim_out) if is_last else nn.Linear(dim_in, dim_out)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                nn.Linear(dim_out, dim_in) if not is_last else nn.Linear(dim_out, dim_in)
            ]))

        self.out_dim = out_dim or dim
        self.final_res_block = block_klass(init_dim * 2, init_dim, time_emb_dim=time_dim)
        self.final_lin = nn.Sequential(
            nn.GroupNorm(resnet_block_groups, init_dim),
            nn.SiLU(),
            nn.Linear(init_dim, self.out_dim)
        )

    def forward(self, x, t, std=None):
        t = t.reshape((t.size(0), self.nb_mod))
        x = self.init_lin(x)
        r = x.clone()
        t = self.time_mlp(t).squeeze()
        h = []

        for block1, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for block1, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        
        return self.final_lin(x) / std if std is not None else self.final_lin(x)


class UnetMLP(nn.Module):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 1), resnet_block_groups=8, time_dim=128, nb_mod=1):
        super().__init__()
        self.nb_mod = nb_mod
        init_dim = init_dim or dim
        self.init_lin = nn.Linear(dim, init_dim)

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        self.time_mlp = nn.Sequential(
            nn.Linear(nb_mod, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                nn.Linear(dim_in, dim_out) if is_last else nn.Linear(dim_in, dim_out)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                nn.Linear(dim_out, dim_in) if not is_last else nn.Linear(dim_out, dim_in)
            ]))

        self.out_dim = out_dim or dim
        self.final_res_block = block_klass(init_dim * 2, init_dim, time_emb_dim=time_dim)
        self.final_lin = nn.Sequential(
            nn.GroupNorm(resnet_block_groups, init_dim),
            nn.SiLU(),
            nn.Linear(init_dim, self.out_dim)
        )

    def forward(self, x, t, std=None):
        t = t.reshape((t.size(0), self.nb_mod))
        x = self.init_lin(x)
        r = x.clone()
        t = self.time_mlp(t).squeeze()
        h = []

        for block1, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for block1, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_lin(x) / std if std is not None else self.final_lin(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=32, shift_scale=False):
        super().__init__()
        self.shift_scale = shift_scale
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out*2 if shift_scale else dim_out)
        ) if time_emb_dim is not None else None

        self.block1 = Block(dim, dim_out, groups=groups, shift_scale=shift_scale)
        self.block2 = Block(dim_out, dim_out, groups=groups, shift_scale=shift_scale)
        self.lin_layer = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            scale_shift = time_emb

        h = self.block1(x, t=scale_shift)
        h = self.block2(h)
        return h + self.lin_layer(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, shift_scale=True):
        super().__init__()
        self.proj = nn.Linear(dim, dim_out)
        self.act = nn.SiLU()
        self.norm = nn.GroupNorm(groups, dim)
        self.shift_scale = shift_scale

    def forward(self, x, t=None):
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)

        if t is not None:
            if self.shift_scale:
                scale, shift = t
                x = x * (scale.squeeze() + 1) + shift.squeeze()
            else:
                x = x + t
        return x
    
class ConvCritic(nn.Module):
    def __init__(self, x_channels, y_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(x_channels + y_channels, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.fc = nn.Linear(6272, 1024)  
        self.output = nn.Linear(1024, 1)

    def forward(self, x, y):
        combined = torch.cat([x, y], dim=1)
        h = torch.relu(self.conv1(combined))
        h = torch.relu(self.conv2(h))
        h = h.view(h.size(0), -1)
        h = torch.relu(self.fc(h))
        return self.output(h)



class SinusoidalEmbedding(torch.nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0]).to(x.device)) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size, device=x.device))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size
    


from functools import partial
import torch
from torch import nn


# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        # nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Linear(dim, default(dim_out, dim))
    )


def Downsample(dim, dim_out=None):
    return nn.Linear(dim, default(dim_out, dim))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, shift_scale=True):
        super().__init__()
        # self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.proj = nn.Linear(dim, dim_out)
        self.act = nn.SiLU()
        # self.act = nn.Relu()
        self.norm = nn.GroupNorm(groups, dim)
        # self.norm = nn.BatchNorm1d( dim)
        self.shift_scale = shift_scale

    def forward(self, x, t=None):
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)

        if exists(t):
            if self.shift_scale:
                scale, shift = t
                x = x * (scale.squeeze() + 1) + shift.squeeze()
            else:
                x = x + t

        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=32, shift_scale=False):
        super().__init__()
        self.shift_scale = shift_scale
        self.mlp = nn.Sequential(
            nn.SiLU(),
            # nn.Linear(time_emb_dim, dim_out * 2)
            nn.Linear(time_emb_dim, dim_out*2 if shift_scale else dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups,
                            shift_scale=shift_scale)
        self.block2 = Block(dim_out, dim_out, groups=groups,
                            shift_scale=shift_scale)
        # self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.lin_layer = nn.Linear(
            dim, dim_out) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):

            time_emb = self.mlp(time_emb)
            scale_shift = time_emb

        h = self.block1(x, t=scale_shift)

        h = self.block2(h)

        return h + self.lin_layer(x)


class UnetMLP_simple(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=128,
        dim_mults=(1, 1),
        resnet_block_groups=8,
        time_dim=128,
        nb_var=1,
    ):
        super().__init__()

        # determine dimensions
        self.nb_var = nb_var
        init_dim = default(init_dim, dim)
        if init_dim == None:
            init_dim = dim * dim_mults[0]

        dim_in = dim
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        self.init_lin = nn.Linear(dim, init_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(nb_var, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            module = nn.ModuleList([block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                                    #        block_klass(dim_in, dim_in, time_emb_dim = time_dim)
                                    ])

            # module.append( Downsample(dim_in, dim_out) if not is_last else nn.Linear(dim_in, dim_out))
            self.downs.append(module)

        mid_dim = dims[-1]
        joint_dim = mid_dim
       # joint_dim = 24
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # self.mid_block2 = block_klass(joint_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            module = nn.ModuleList([block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                                    #       block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim)
                                    ])
            # module.append( Upsample(dim_out, dim_in) if not is_last else  nn.Linear(dim_out, dim_in))
            self.ups.append(module)

        # default_out_dim = channels * (1 if not learned_variance else 2)

        self.out_dim = dim_in

        self.final_res_block = block_klass(
            init_dim * 2, init_dim, time_emb_dim=time_dim)

        self.proj = nn.Linear(init_dim, dim)

        self.proj.weight.data.fill_(0.0)
        self.proj.bias.data.fill_(0.0)

        self.final_lin = nn.Sequential(
            nn.GroupNorm(resnet_block_groups, init_dim),
            nn.SiLU(),
            self.proj
        )

    def forward(self, x, t=None, std=None):
        t = t.reshape(t.size(0), self.nb_var)

        x = self.init_lin(x.float())

        r = x.clone()

        t = self.time_mlp(t).squeeze()

        h = []

        for blocks in self.downs:

            block1 = blocks[0]

            x = block1(x, t)

            h.append(x)
       #     x = downsample(x)

        # x = self.mid_block1(x, t)

        # x = self.mid_block2(x, t)

        for blocks in self.ups:

            block1 = blocks[0]
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            # x = torch.cat((x, h.pop()), dim = 1)
            # x = block2(x, t)

           # x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)

        if std != None:
            return self.final_lin(x) / std
        else:
            return self.final_lin(x)
