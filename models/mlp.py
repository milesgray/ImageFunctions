import torch
import torch.nn as nn

from models import register
from .layers import LinearResidual

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
        elif act.lower() == 'relu':
            self.act = nn.ReLU() 
        elif act.lower() == 'gelu':
            self.act = nn.GELU()
        elif act.lower() == 'sine':
            self.act = Sine()
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
                layers.append(LinearResidual(1.0, True, 'residual', transform))
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
            self.layers.apply(sine_init)
            self.layers[0].apply(first_layer_sine_init)

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
            layers.append(Sine(w0=w0))            
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

@register('sine')
class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        """Sine activation function with w0 scaling support.
        Example:
            >>> w = torch.tensor([3.14, 1.57])
            >>> Sine(w0=1)(w)
            torch.Tensor([0, 1])
        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return torch.sin(self.w0 * x)

    @staticmethod
    def _check_input(x):
        if not isinstance(x, torch.Tensor):
            raise TypeError('input to forward() must be torch.Tensor')

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            print('sine_init for Siren...')
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            print('first_layer_sine_init for Siren...')
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
            
def init_weights(m):
    # if hasattr(modules, 'weight'):
    if isinstance(m, nn.Linear):
        num_input = m.weight.size(-1)
        # See supplement Sec. 1.5 for discussion of factor 30
        m.weight.data.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)