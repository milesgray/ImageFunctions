import math
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from argparse import Namespace

from .registry import register

from .activations import create as create_activation
from .linear_residual import LinearResidual

@register("fourier_inr")
class FourierINR(nn.Module):
    """
    INR with Fourier features as specified in https://people.eecs.berkeley.edu/~bmild/fourfeat/
    """
    def __init__(self, in_features, args, 
                 num_fourier_feats=64, 
                 layer_sizes=[64,64,64], 
                 out_features=64, 
                 has_bias=True, 
                 activation="leaky_relu", 
                 learnable_basis=True,):
        super().__init__()

        layers = [
            nn.Linear(num_fourier_feats * 2, layer_sizes[0], bias=has_bias),
            create_activation(activation)
        ]

        for index in range(len(layer_sizes) - 1):
            transform = nn.Sequential(
                nn.Linear(layer_sizes[index], layer_sizes[index + 1], bias=has_bias),
                create_activation(activation)
            )

            if args.residual.enabled:
                layers.append(LinearResidual(transform, **args.residual))
            else:
                layers.append(transform)

        layers.append(nn.Linear(layer_sizes[-1], out_features, bias=has_bias))

        self.model = nn.Sequential(*layers)

        # Initializing the basis
        basis_matrix = args.scale * torch.randn(num_fourier_feats, in_features)
        self.basis_matrix = nn.Parameter(basis_matrix, requires_grad=learnable_basis)

    def compute_fourier_feats(self, coords: Tensor) -> Tensor:
        sines = (2 * np.pi * coords @ self.basis_matrix.t()).sin() # [batch_size, num_fourier_feats]
        cosines = (2 * np.pi * coords @ self.basis_matrix.t()).cos() # [batch_size, num_fourier_feats]

        return torch.cat([sines, cosines], dim=1) # [batch_size, num_fourier_feats * 2]

    def forward(self, coords: Tensor) -> Tensor:
        return self.model(self.compute_fourier_feats(coords))
    
@register("fourier_conv2d")
class FourierConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

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

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x

@register("fourier_mlp")
class FourierFeatureMLP(nn.Module):
    """MLP which uses Fourier features as a preprocessing step."""

    def __init__(self, num_inputs: int, num_outputs: int,
                 a_values: torch.Tensor, b_values: torch.Tensor,
                 layer_channels: List[int], linear: nn.Module = nn.Linear):
        """Constructor.
        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            a_values (torch.Tensor): a values for encoding
            b_values (torch.Tensor): b values for encoding
            num_layers (int): Number of layers in the MLP
            layer_channels (List[int]): Number of channels per layer.
        """
        nn.Module.__init__(self)
        self.params = {
            "num_inputs": num_inputs,
            "num_outputs": num_outputs,
            "a_values": None if a_values is None else a_values.tolist(),
            "b_values": None if b_values is None else b_values.tolist(),
            "layer_channels": layer_channels
        }
        self.num_inputs = num_inputs
        if b_values is None:
            self.a_values = None
            self.b_values = None
            num_inputs = num_inputs
        else:
            assert b_values.shape[0] == num_inputs
            assert a_values.shape[0] == b_values.shape[1]
            self.a_values = nn.Parameter(a_values, requires_grad=False)
            self.b_values = nn.Parameter(b_values, requires_grad=False)
            num_inputs = b_values.shape[1] * 2

        self.layers = nn.ModuleList()
        for num_channels in layer_channels:
            self.layers.append(linear(num_inputs, num_channels))
            num_inputs = num_channels

        self.layers.append(nn.Linear(num_inputs, num_outputs))

        self.use_view = False
        self.keep_activations = False
        self.activations = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predicts outputs from the provided uv input."""
        if self.b_values is None:
            output = inputs
        else:
            # NB: the below should be 2*math.pi, but the values
            # coming in are already in the range of -1 to 1 or
            # 0 to 2, so we want to keep the range so that it does
            # not exceed 2pi
            encoded = (math.pi * inputs) @ self.b_values
            output = torch.cat([self.a_values * encoded.cos(),
                                self.a_values * encoded.sin()], dim=-1)

        self.activations.clear()
        for layer in self.layers[:-1]:
            output = torch.relu(layer(output))

        if self.keep_activations:
            self.activations.append(output.detach().cpu().numpy())

        output = self.layers[-1](output)
        return output

    def save(self, path: str):
        """Saves the model to the specified path.
        Args:
            path (str): Path to the model file on disk
        """
        state_dict = self.state_dict()
        state_dict["type"] = "fourier"
        state_dict["params"] = self.params
        torch.save(state_dict, path)

@register("fourier_unit_circle_mlp")
class BasicFourierMLP(FourierFeatureMLP):
    """Basic version of FFN in which inputs are projected onto the unit circle."""

    def __init__(self, num_inputs: int, num_outputs: int, num_layers=3,
                 num_channels=256):
        """Constructor.
        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
        """
        a_values = torch.ones(num_inputs)
        b_values = torch.eye(num_inputs)
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs,
                                   a_values, b_values,
                                   [num_channels] * num_layers)

@register("fourier_positional_mlp")
class PositionalFourierMLP(FourierFeatureMLP):
    """Version of FFN with positional encoding."""
    def __init__(self, num_inputs: int, num_outputs: int, max_log_scale: float,
                 num_layers=3, num_channels=256, embedding_size=256):
        """Constructor.
        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            max_log_scale (float): Maximum log scale for embedding
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            embedding_size (int, optional): The size of the feature embedding.
                                            Defaults to 256.
        """
        b_values = self._encoding(max_log_scale, embedding_size, num_inputs)
        a_values = torch.ones(b_values.shape[1])
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs,
                                   a_values, b_values,
                                   [num_channels] * num_layers)

    @staticmethod
    def _encoding(max_log_scale: float, embedding_size: int, num_inputs: int):
        """Produces the encoding b_values matrix."""
        embedding_size = embedding_size // num_inputs
        frequencies_matrix = 2. ** torch.linspace(0, max_log_scale, embedding_size)
        frequencies_matrix = frequencies_matrix.reshape(-1, 1, 1)
        frequencies_matrix = torch.eye(num_inputs) * frequencies_matrix
        frequencies_matrix = frequencies_matrix.reshape(-1, num_inputs)
        frequencies_matrix = frequencies_matrix.transpose(0, 1)
        return frequencies_matrix

@register("fourier_gaussian_mlp")
class GaussianFourierMLP(FourierFeatureMLP):
    """Version of a FFN using a full Gaussian matrix for encoding."""

    def __init__(self, num_inputs: int, num_outputs: int, sigma: float,
                 num_layers=3, num_channels=256, embedding_size=256):
        """Constructor.
        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            sigma (float): Standard deviation of the Gaussian distribution
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            embedding_size (int, optional): Number of frequencies to use for
                                             the encoding. Defaults to 256.
        """
        b_values = torch.normal(0, sigma, size=(num_inputs, embedding_size))
        a_values = torch.ones(b_values.shape[1])
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs,
                                   a_values, b_values,
                                   [num_channels] * num_layers)