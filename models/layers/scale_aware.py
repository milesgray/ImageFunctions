import torch
import torch.nn as nn

class SA_adapt(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2,
                        mode='bilinear',
                        align_corners=False),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.adapt = SA_conv(channels, channels, 3, 1, 1)

    def forward(self, x, scale, scale2):
        mask = self.mask(x)
        adapted = self.adapt(x, scale, scale2)

        return x + adapted * mask


class SA_conv(nn.Module):
    def __init__(self, channels_in, channels_out,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False,
                 num_experts=4):
        super().__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts
        self.bias = bias

        # FC layers to generate routing weights
        self.routing = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, num_experts),
            nn.Softmax(1)
        )

        # initialize experts
        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(
                nn.Parameter(
                    torch.Tensor(channels_out,
                                 channels_in,
                                 kernel_size,
                                 kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

        if bias:
            self.bias_pool = nn.Parameter(
                torch.Tensor(num_experts,
                             channels_out)
            )
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_pool)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, x, scale, scale2):
        # generate routing weights
        scale = torch.ones(1, 1).to(x.device) / scale
        scale2 = torch.ones(1, 1).to(x.device) / scale2
        routing_weights = self.routing(torch.cat((scale, scale2), 1))
        routing_weights = routing_weights.view(self.num_experts, 1, 1)

        # fuse experts
        fused_weight = self.weight_pool.view(self.num_experts, -1, 1)
        fused_weight = fused_weight * routing_weights
        fused_weight = fused_weight.sum(0)
        fused_weight = fused_weight.view(-1,
                                         self.channels_in,
                                         self.kernel_size,
                                         self.kernel_size)

        if self.bias:
            fused_bias = torch.mm(routing_weights, self.bias_pool)
            fused_bias = fused_bias.view(-1)
        else:
            fused_bias = None

        # convolution
        out = F.conv2d(x, fused_weight, fused_bias,
                       stride=self.stride,
                       padding=self.padding)

        return out
