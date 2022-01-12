import torch
import torch.nn as nn

from models import register
from .layers import LinearResidual
from .layers import create as create_layer
from .layers.activations import create as create_act

@register('mlp')
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, 
                act='gelu', 
                has_bn=False,
                norm="nn.LayerNorm", 
                has_bias=False, 
                use_residual=True,
                w0=1.0):
        super().__init__()
        if norm is not None:
            norm = eval(norm) if isinstance(norm, str) else norm
        self.norm = norm
        if act is None:
            self.act = None
        elif isinstance(act, str):
            self.act = create_act(act.lower())
        elif isinstance(act, nn.Module):
            self.act = act
            act = str(type(act)).split("'")[-2]
            act = act.split(".")[-1] if "." in act else act
        else:
            assert False, f'activation {act} is not supported'
            
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            if use_residual and lastv == hidden:
                block = []
                block.append(nn.Linear(lastv, hidden, bias=has_bias))
                if has_bn:
                    block.append(self.norm(hidden))
                if self.act:
                    block.append(self.act)
                transform = nn.Sequential(*block)
                layers.append(LinearResidual(1.0, 
                                             True, 
                                             'residual', 
                                             transform))
            else:
                layers.append(nn.Linear(lastv, hidden, bias=has_bias))
                if has_bn:
                    layers.append(self.norm(hidden))
                if self.act:
                    layers.append(self.act)
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)
        if act is not None and act.lower() == 'sine':
            self.layers.apply(self.act.init)
            self.layers[0].apply(self.act.first_layer_init)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.contiguous().view(-1, x.shape[-1]))
        return x.view(*shape, -1)

@register('siren')
class SIREN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, has_bn=False, has_bias=False, w0=1.0):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden, bias=has_bias))
            if has_bn:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(create_act('sine', {"w0": w0}))
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

def init_weights(m):
    # if hasattr(modules, 'weight'):
    if isinstance(m, nn.Linear):
        num_input = m.weight.size(-1)
        # See supplement Sec. 1.5 for discussion of factor 30
        m.weight.data.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)