import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.layers import Scale
from .methods import *

from .registry import register

@register("sine_learnable")
class SineLearnable(nn.Module):
    def __init__(self, w0: float=1.0, 
                 scale: float=1.0, 
                 learnable: bool=False):
        """Sine activation function with w0 scaling support along with
            a second learnable scaling factor applied to the output.
        Example:
            >>> w = torch.tensor([3.14, 1.57])
            >>> Sine(w0=1)(w)
            torch.Tensor([0, 1])
        :param w0: w0 in the activation step `act(x; w0) = s * sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        :param scale: s in the activation step `act(x; w0; s) = s * sin(w0 * x)`.
            defaults to 1.0
        :type scale: float, optional
        :param learnable: enables updates to `s` value in activation step  
            `act(x; w0; s) = s * sin(w0 * x)`.
        :type learnable: float, optional
        """
        super().__init__()
        self.w0 = w0
        self.scale = Scale(scale) if learnable else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.scale(torch.sin(self.w0 * x))
    
    @staticmethod
    def init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                print('sine_init for Siren...')
                num_input = m.weight.size(-1)
                # See supplement Sec. 1.5 for discussion of factor 30
                m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
                
    @staticmethod
    def first_layer_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                print('first_layer_sine_init for Siren...')
                num_input = m.weight.size(-1)
                # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
                m.weight.uniform_(-1 / num_input, 1 / num_input)    

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
        
    @staticmethod
    def init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                print('sine_init for Siren...')
                num_input = m.weight.size(-1)
                # See supplement Sec. 1.5 for discussion of factor 30
                m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
                
    @staticmethod
    def first_layer_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                print('first_layer_sine_init for Siren...')
                num_input = m.weight.size(-1)
                # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
                m.weight.uniform_(-1 / num_input, 1 / num_input)
                    

@register("sines_cosines")
class SinesCosines(nn.Module):
    """
    Sines-cosines activation function
    It applies both sines and cosines and concatenates the results
    """
    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([x.sin(), x.cos()], dim=1)