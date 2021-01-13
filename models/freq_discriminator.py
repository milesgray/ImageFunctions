from argparse import Namespace

import torch
import torch.nn as nn

from models import register

class FourierINR(nn.Module):
    """
    INR with Fourier features as specified in https://people.eecs.berkeley.edu/~bmild/fourfeat/
    """
    def __init__(self, config):
        super(FourierINR, self).__init__()

        layers = [
            nn.Linear(config.num_fourier_feats * 2, config.layer_sizes[0], bias=config.has_bias),
            create_activation(config.activation)
        ]

        for index in range(len(config.layer_sizes) - 1):
            transform = nn.Sequential(
                nn.Linear(config.layer_sizes[index], config.layer_sizes[index + 1], bias=config.has_bias),
                create_activation(config.activation)
            )

            if config.residual.enabled:
                layers.append(LinearResidual(config.residual, transform))
            else:
                layers.append(transform)

        layers.append(nn.Linear(config.layer_sizes[-1], config.out_features, bias=config.has_bias))

        self.model = nn.Sequential(*layers)

        # Initializing the basis
        basis_matrix = config.scale * torch.randn(config.num_fourier_feats, config.in_features)
        self.basis_matrix = nn.Parameter(basis_matrix, requires_grad=config.learnable_basis)

    def compute_fourier_feats(self, coords):
        sines = (2 * np.pi * coords @ self.basis_matrix.t()).sin() # [batch_size, num_fourier_feats]
        cosines = (2 * np.pi * coords @ self.basis_matrix.t()).cos() # [batch_size, num_fourier_feats]

        return torch.cat([sines, cosines], dim=1) # [batch_size, num_fourier_feats * 2]

    def forward(self, coords):
        return self.model(self.compute_fourier_feats(coords))

class BasicDiscriminatorBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.block = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=2,
                padding=1,
            )),
            nn.LeakyReLU(0.2, True),

            nn.utils.weight_norm(nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )),
            nn.LeakyReLU(0.2, True),

            nn.utils.weight_norm(nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )),
            nn.LeakyReLU(0.2, True),

            nn.utils.weight_norm(nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )),
        )

    def forward(self, x):
        return self.block(x)

class ResDiscriminatorBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=2,
                padding=1,
            )),
            nn.LeakyReLU(0.2, True),

            nn.utils.weight_norm(nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )),
        )

        self.shortcut1 = nn.utils.weight_norm(nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=1,
                stride=2,
            ))

        self.block2 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )),
            nn.LeakyReLU(0.2, True),

            nn.utils.weight_norm(nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )),
        )

        self.shortcut2 = nn.utils.weight_norm(nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=1,
                stride=1,
            ))

    def forward(self, x):
        x1 = self.block1(x)
        x1 = x1 + self.shortcut1(x)
        return self.block2(x1) + self.shortcut2(x1)


class ResNet18Discriminator(nn.Module):
    def __init__(self, stft_channel, in_channel=64):
        super().__init__()
        self.input = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(stft_channel, in_channel, kernel_size=7, stride=2, padding=1,)),
            nn.LeakyReLU(0.2, True),
            )
        self.df1 = BasicDiscriminatorBlock(in_channel, in_channel)
        self.df2 = ResDiscriminatorBlock(in_channel, in_channel*2)
        self.df3 = ResDiscriminatorBlock(in_channel*2, in_channel*4)
        self.df4 = ResDiscriminatorBlock(in_channel * 4, in_channel * 8)

    def forward(self, x):
        x = self.input(x)
        x = self.df1(x)
        x = self.df2(x)
        x = self.df3(x)
        return self.df4(x)



class FrequencyDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fft_size = args.fft_size
        self.hop_length = args.hop_length
        self.win_length = args.win_length
        self.window = getattr(torch, window)(args.win_length)
        self.stft_channel = args.fft_size // 2 + 1
        self.resnet_disc = ResNet18Discriminator(self.stft_channel, args.in_channel)

    def forward(self, x):
        x_stft = torch.stft(x, self.fft_size, self.hop_length, self.win_length, self.window.cuda())
        real = x_stft[..., 0]
        imag = x_stft[..., 1]

        x_real = self.resnet_disc(real)
        x_imag = self.resnet_disc(imag)

        # return magnitude
        return torch.sqrt(torch.clamp(x_real ** 2 + x_imag ** 2, min=1e-7)).transpose(2, 1)


@register('freq_disc')
def make_freq_disc(in_channel=64, fft_size=1024, hop_length=256, win_length=1024, window="hann_window"):
    args = Namespace()
    args.in_channel = in_channel
    args.fft_size = fft_size
    args.hop_length = hop_length
    args.win_length = win_length
    args.window = window

    return FrequencyDiscriminator(args)
