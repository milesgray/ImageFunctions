import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Identity
from torch.nn.init import _calculate_correct_fan, calculate_gain

from .stiefel import dof_stiefel, dof_stiefel_canonical
from ImageFunctions.utility.torch import get_statedict_num_params, deep_transform, is_conv_transposed
from ImageFunctions.utility.tensor import get_tt_contraction_fn_and_flops
from ImageFunctions.utility.math import get_ranks_tt, dim_factorize_simple

from .registry import register

class SpectralLayerBase(torch.nn.Module):
    def __init__(self, module, name, spectral_tensors_factory):
        super().__init__()
        assert isinstance(module, (torch.nn.modules.conv._ConvNd, torch.nn.Linear, torch.nn.Embedding))
        self.name = name
        self.weight_shape = module.weight.shape
        self.spectral_tensors_factory = [spectral_tensors_factory]  # guard parameters iterator from traversing down
        self.init_checks(module)
        if not self.is_embedding:
            if module.bias is None:
                self.register_parameter('bias', None)
            else:
                self.bias = module.bias
        self.num_param_weight = None
        self.initialize()

    def initialize(self, fan_mode='fan_in', nonlinearity='relu', nonlinearity_negative_slope=0):
        with torch.no_grad():
            std_target = None
            permute_to_factory, permute_from_factory = None, None
            if not self.is_embedding:
                fan = _calculate_correct_fan(torch.empty(*self.weight_shape), fan_mode)
                gain = calculate_gain(nonlinearity, nonlinearity_negative_slope)
                std_target = gain / math.sqrt(fan)
                if self.is_conv and self.is_transposed:
                    # unlike every other op having leading dim C_out, transposed_conv.weight is C_in x C_out x K x K
                    permute_to_factory = [1, 0, 2, 3]
                    permute_from_factory = [1, 0, 2, 3]
            self.num_param_weight = self.spectral_tensors_factory[0].add_tensor(
                self.name, self.weight_shape, std_target, permute_to_factory, permute_from_factory
            )

    def num_parameters_weight(self):
        return self.num_param_weight


class SpectralConvNd(SpectralLayerBase):
    map_type_to_name = {
        torch.nn.Conv1d: 'Conv1d',
        torch.nn.Conv2d: 'Conv2d',
        torch.nn.Conv3d: 'Conv3d',
        torch.nn.ConvTranspose1d: 'ConvTranspose1d',
        torch.nn.ConvTranspose2d: 'ConvTranspose2d',
        torch.nn.ConvTranspose3d: 'ConvTranspose3d',
    }

    map_type_to_F = {
        torch.nn.Conv1d: F.conv1d,
        torch.nn.Conv2d: F.conv2d,
        torch.nn.Conv3d: F.conv3d,
        torch.nn.ConvTranspose1d: F.conv_transpose1d,
        torch.nn.ConvTranspose2d: F.conv_transpose2d,
        torch.nn.ConvTranspose3d: F.conv_transpose3d,
    }

    def __init__(self, module, *args, **kwargs):
        self.layer_type_name = self.map_type_to_name[type(module)]
        self.is_conv = True
        self.is_transposed = is_conv_transposed(module)
        self.is_embedding = False
        super().__init__(module, *args, **kwargs)
        self.conv_fn = self.map_type_to_F[type(module)]
        self.conv_cls = type(module)
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.kernel_size = module.kernel_size
        self.kernel_numel = torch.tensor(self.kernel_size).prod().item()
        self.stride = module.stride
        self.padding = module.padding
        self.output_padding = module.output_padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.padding_mode = module.padding_mode

    def init_checks(self, module):
        assert isinstance(module, torch.nn.modules.conv._ConvNd)
        assert not self.is_transposed and module.padding_mode != 'circular' or module.padding_mode == 'zeros', \
            'Not implemented'

    def forward(self, x):
        weight = self.spectral_tensors_factory[0].get_tensor_by_name(self.name)
        if self.is_transposed:
            out = self.conv_fn(
                x, weight, self.bias, self.stride, self.padding,
                self.output_padding, self.groups, self.dilation
            )
        else:
            out = self.conv_fn(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    def extra_repr(self):
        return f'{self.layer_type_name} [{" x ".join([str(a) for a in self.weight_shape])}] stride={self.stride} ' \
            f'padding={self.padding} dilation={self.dilation} groups={self.groups} has_bias={self.bias is not None}'


class SpectralLinear(SpectralLayerBase):
    def __init__(self, module, *args, **kwargs):
        self.layer_type_name = 'Linear'
        self.is_conv = False
        self.is_embedding = False
        super().__init__(module, *args, **kwargs)
        self.in_features = module.in_features
        self.out_features = module.out_features

    def init_checks(self, module):
        assert isinstance(module, torch.nn.modules.Linear)

    def forward(self, x):
        weight = self.spectral_tensors_factory[0].get_tensor_by_name(self.name)
        return F.linear(x, weight, self.bias)

    def extra_repr(self):
        return f'[{" x ".join([str(a) for a in self.weight_shape])}] has_bias={self.bias is not None}'


class SpectralEmbedding(SpectralLayerBase):
    def __init__(self, module, *args, **kwargs):
        self.layer_type_name = 'Embedding'
        self.is_conv = False
        self.is_embedding = True
        super().__init__(module, *args, **kwargs)
        self.embedding_dim = module.embedding_dim
        self.num_embeddings = module.num_embeddings
        self.padding_idx = module.padding_idx
        self.max_norm = module.max_norm
        self.norm_type = module.norm_type
        self.scale_grad_by_freq = module.scale_grad_by_freq
        self.sparse = module.sparse

    def init_checks(self, module):
        assert isinstance(module, torch.nn.modules.Embedding)

    def forward(self, x):
        weight = self.spectral_tensors_factory[0].get_tensor_by_name(self.name)
        return F.embedding(
            x, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )

    def extra_repr(self):
        return f'[{" x ".join([str(a) for a in self.weight_shape])}]'
    
    
class SpectralTensorsFactoryBase(torch.nn.Module):
    def __init__(
            self, cls_stiefel_full, cls_stiefel_canonical, max_rank, flatten_filter_dimensions=True,
            spectrum_eye=False, init_mode=None, init_std=None, init_seed=None,
    ):
        super().__init__()
        assert init_seed is None or type(init_seed) is int
        self.cls_stiefel_full = cls_stiefel_full
        self.cls_stiefel_canonical = cls_stiefel_canonical
        self.max_rank = max_rank
        self.flatten_filter_dimensions = flatten_filter_dimensions
        self.spectrum_eye = spectrum_eye
        self.init_mode = init_mode
        self.init_std = init_std
        self.init_seed = init_seed
        self.stiefels_full = OrderedDict()
        self.stiefels_canonical = OrderedDict()
        self.singular_values = OrderedDict()
        self.map_tensorname_to_descriptor = OrderedDict()
        self.map_corename_to_descriptor = OrderedDict()
        self.list_names = []
        self.instantiated = False
        self.num_params = 0
        self.list_names = []
        self.map_name_to_id = {}
        self.last_tensors = None

    def add_stiefel(self, name, rank, n, is_thin, is_canonical, target_shape):
        assert rank <= n
        shape_desc = f'{rank}_{n}'
        stiefels = self.stiefels_full if (not is_canonical or self.cls_stiefel_full == self.cls_stiefel_canonical) \
            else self.stiefels_canonical
        stiefel = stiefels.get(shape_desc, None)
        if stiefel is None:
            stiefel = {
                'batch': 0,
                'rank': rank,
                'n': n,
            }
        self.map_corename_to_descriptor[name] = {
            'stiefel_desc': shape_desc,
            'id': stiefel['batch'],
            'shape': target_shape,
            'is_thin': is_thin,
            'is_canonical': is_canonical,
        }
        stiefel['batch'] += 1
        stiefels[shape_desc] = stiefel
        return dof_stiefel_canonical(rank, n) if is_canonical else dof_stiefel(rank, n)

    def add_tensor(self, name, shape, target_std=None, permute_fwd=None, permute_bwd=None):
        assert not self.instantiated
        assert name not in self.map_name_to_id

        self.map_name_to_id[name] = len(self.list_names)
        self.list_names.append(name)

        return self.on_add_tensor(name, shape, target_std, permute_fwd, permute_bwd)

    def instantiate(self):
        assert not self.instantiated

        stats = []
        for stiefels, cls_stiefel, is_canonical in (
                (self.stiefels_full, self.cls_stiefel_full, False),
                (self.stiefels_canonical, self.cls_stiefel_canonical, True),
        ):
            for i, (k, v) in enumerate(stiefels.items()):
                m = cls_stiefel(
                    batch=v['batch'],
                    rank=v['rank'],
                    n=v['n'],
                    is_thin=True,
                    init_mode=self.init_mode,
                    init_std=self.init_std,
                    init_seed=None if self.init_seed is None else self.init_seed + i,
                )
                stiefels[k] = m
                self.num_params += m.dof()
                v = v.copy()
                v['is_canonical'] = is_canonical
                stats.append(v)
        self.stiefels_full = torch.nn.ModuleDict(self.stiefels_full)
        self.stiefels_canonical = torch.nn.ModuleDict(self.stiefels_canonical)

        if not self.spectrum_eye:
            for k, v in self.map_tensorname_to_descriptor.items():
                sv_init = torch.ones((v['num_singular_values'],), dtype=torch.float32)
                self.singular_values[k.replace('.', '-')] = torch.nn.Parameter(sv_init)
                self.num_params += v['num_singular_values']
            self.singular_values = torch.nn.ParameterDict(self.singular_values)

        self.instantiated = True
        return stats

    def get_names(self):
        assert self.instantiated
        return self.list_names

    @property
    def have_singular_values(self):
        return not self.spectrum_eye

    def forward_singular_value_one(self, name):
        assert self.instantiated
        sv = self.singular_values[name]
        sv = sv / sv.abs().max().clamp(min=1e-8)
        return sv

    def forward_singular_values(self):
        assert self.instantiated
        return {k.replace('-', '.'): self.forward_singular_value_one(k) for k in self.singular_values.keys()}

    def forward(self):
        assert self.instantiated
        singular_values = None
        if not self.spectrum_eye:
            singular_values = self.forward_singular_values()

        stiefels_full, stiefels_canonical = {}, {}
        for src, dst in (
                (self.stiefels_full, stiefels_full),
                (self.stiefels_canonical, stiefels_canonical),
        ):
            for k, v in src.items():
                v = v.forward()
                v = v.chunk(v.shape[0], dim=0)  # produces TracerWarning -- safe to ignore
                v = [a.squeeze() for a in v]
                dst[k] = v

        tensors = OrderedDict()
        for name, desc in self.map_tensorname_to_descriptor.items():
            num_cores = desc['num_cores']
            cores = []
            for i in range(num_cores):
                core_name = self.core_name(name, i)
                core_desc = self.map_corename_to_descriptor[core_name]
                stiefel_desc = core_desc['stiefel_desc']
                id_in_stiefel = core_desc['id']
                is_canonical = core_desc['is_canonical']
                stiefels = stiefels_full if (not is_canonical or self.cls_stiefel_full == self.cls_stiefel_canonical) \
                    else stiefels_canonical
                core = stiefels[stiefel_desc][id_in_stiefel]
                if not core_desc['is_thin']:
                    core = core.T
                core = core.reshape(core_desc['shape'])
                cores.append(core)

            if not self.spectrum_eye:
                sv = singular_values[name]
                sv_id = desc['singular_insertion_rank_id'] - 1
                core_id_shape = cores[sv_id].shape
                core_id = cores[sv_id].reshape(-1, core_id_shape[-1]) * sv.reshape(1, -1)
                cores[sv_id] = core_id.view(core_id_shape)

            tensor = desc['contraction_fn'](*cores)
            tensor = tensor.reshape(desc['shape_src'])
            if desc['permute_from_factory'] is not None:
                tensor = tensor.permute(desc['permute_from_factory'])
            tensors[name] = tensor

        return tuple(tensors.values())

    def forward_2(self):
        t = self.forward()
        self.set_tensors(t)

    @property
    def has_stiefel_penalty(self):
        return self.cls_stiefel_full.has_penalty

    def stiefel_penalty(self, normalize=False):
        assert self.cls_stiefel_full == self.cls_stiefel_canonical
        penalty = sum(v.penalty()
                      for stiefels in (self.stiefels_full, self.stiefels_canonical)
                      for v in stiefels.values())
        if normalize:
            denom = sum(v.batch * (v.rank ** 2)
                        for stiefels in (self.stiefels_full, self.stiefels_canonical)
                        for v in stiefels.values())
            penalty = penalty / denom
        return penalty

    def num_parameters(self):
        assert self.instantiated
        return self.num_params

    def state_dict(self, *args, **kwargs):
        assert self.instantiated
        sd = super().state_dict(*args, **kwargs)
        sd.pop('last_tensors', None)
        return sd

    def set_tensors(self, tensors):
        assert self.instantiated
        self.last_tensors = tensors

    def get_tensor_by_name(self, name):
        assert self.instantiated
        return self.last_tensors[self.map_name_to_id[name]]


class SpectralTensorsFactorySVDP(SpectralTensorsFactoryBase):
    @staticmethod
    def core_name(name, id):
        assert id in (0, 1)
        return f'{name}#{"U" if id == 0 else "Vt"}'

    def on_add_tensor(self, name, shape, target_std=None, permute_to_factory=None, permute_from_factory=None):
        assert not self.instantiated
        assert name not in self.map_tensorname_to_descriptor
        assert len(shape) in (2, 4)
        assert permute_to_factory is None or len(permute_to_factory) == len(shape)
        assert permute_from_factory is None or len(permute_to_factory) == len(shape)

        if permute_to_factory is not None:
            shape = [shape[permute_to_factory[i]] for i in range(len(shape))]
        shape_dst = [shape[0], torch.tensor(shape[1:]).prod().item()]
        rank = num_singular_values = min(shape_dst[0], shape_dst[1], self.max_rank)

        n_param_uncompressed = shape_dst[0] * shape_dst[1]
        n_param_compressed = 0 if self.spectrum_eye else num_singular_values

        n_param_compressed += self.add_stiefel(
            name=self.core_name(name, 0),
            rank=rank,
            n=shape_dst[0],
            is_thin=True,
            is_canonical=self.spectrum_eye,
            target_shape=(shape_dst[0], rank),
        )

        n_param_compressed += self.add_stiefel(
            name=self.core_name(name, 1),
            rank=rank,
            n=shape_dst[1],
            is_thin=False,
            is_canonical=False,
            target_shape=(rank, shape_dst[1])
        )

        contraction_fn = lambda U, Vt: U.mm(Vt)

        self.map_tensorname_to_descriptor[name] = {
            'shape_src': shape,
            'permute_from_factory': permute_from_factory,
            'singular_insertion_rank_id': 1,
            'target_std': target_std,
            'num_singular_values': num_singular_values,
            'num_cores': 2,
            'contraction_fn': contraction_fn,
            'n_param_uncompressed': n_param_uncompressed,
            'n_param_compressed': n_param_compressed,
        }

        return n_param_compressed


class SpectralTensorsFactorySTTP(SpectralTensorsFactoryBase):
    def _add_core(self, name, r_l, m, r_r, is_left, is_canonical):
        if is_left:
            rank = r_r
            n = r_l * m
            is_thin = True
        else:
            rank = r_l
            n = m * r_r
            is_thin = False
        return self.add_stiefel(name, rank, n, is_thin, is_canonical, (r_l, m, r_r))

    @staticmethod
    def core_name(name, id):
        return f'{name}#core{id}'

    def on_add_tensor(self, name, shape, target_std=None, permute_to_factory=None, permute_from_factory=None):
        assert not self.instantiated
        assert name not in self.map_tensorname_to_descriptor
        assert len(shape) in (2, 4)
        assert permute_to_factory is None or len(permute_to_factory) == len(shape)
        assert permute_from_factory is None or len(permute_to_factory) == len(shape)

        if permute_to_factory is not None:
            shape = [shape[permute_to_factory[i]] for i in range(len(shape))]
        shape_dst = dim_factorize_simple(shape[0], True)
        singular_insertion_rank_id = len(shape_dst)
        shape_dst.extend(dim_factorize_simple(shape[1], False))
        if len(shape) > 2:
            if self.flatten_filter_dimensions:
                mode_filter_dims = torch.tensor(shape[2:]).prod().item()
                shape_dst.append(mode_filter_dims)
            else:
                for d in shape[2:]:
                    shape_dst.extend(dim_factorize_simple(d, False))
        ranks_tt = get_ranks_tt(shape_dst, self.max_rank)
        assert len(ranks_tt) == len(shape_dst) + 1 and ranks_tt[0] == ranks_tt[-1] == 1
        num_singular_values = ranks_tt[singular_insertion_rank_id]

        n_param_uncompressed = torch.tensor(shape_dst).prod().item()
        n_param_compressed = 0 if self.spectrum_eye else num_singular_values
        list_core_shapes = []
        for i, m in enumerate(shape_dst):
            core_name = self.core_name(name, i)
            r_l = ranks_tt[i]
            r_r = ranks_tt[i+1]
            is_left = i < singular_insertion_rank_id
            if i < singular_insertion_rank_id - 1:
                is_canonical = True
            elif i == singular_insertion_rank_id - 1:
                is_canonical = self.spectrum_eye
            elif i == singular_insertion_rank_id:
                is_canonical = False
            else:
                is_canonical = True
            n_param_compressed += self._add_core(core_name, r_l, m, r_r, is_left, is_canonical)
            list_core_shapes.append((r_l, m, r_r))
        contraction_fn, contraction_flops = get_tt_contraction_fn_and_flops(list_core_shapes)

        self.map_tensorname_to_descriptor[name] = {
            'shape_src': shape,
            'permute_from_factory': permute_from_factory,
            'singular_insertion_rank_id': singular_insertion_rank_id,
            'target_std': target_std,
            'num_singular_values': num_singular_values,
            'num_cores': len(shape_dst),
            'contraction_fn': contraction_fn,
            'contraction_flops': contraction_flops,
            'n_param_uncompressed': n_param_uncompressed,
            'n_param_compressed': n_param_compressed,
        }

        return n_param_compressed
    
@register("spectral_conv2d")
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, shift=False, freq=False):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.shift = shift # apply fftshift to put high frequency components in front
        self.freq = freq # apply fftfreq to fft results

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, norm="ortho")
        if self.shift:
            x_ft = torch.fft.fftshift(x_ft)
        if self.freq:
            freq = torch.fft.rfftfreq(x_ft.shape[-2])
            if torch.cuda.is_available():
                freq = freq.cuda()
            x_ft = x_ft * freq

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x