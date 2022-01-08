import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor

from .registry import register

@register("zpool")
class ZPool(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(
            (
                torch.max(x, self.dim)[0].unsqueeze(1),
                torch.mean(x, self.dim).unsqueeze(1)
            ),
            dim=1
        )

@register("spatialmeanpool")
class SpatialMeanPool(nn.Module):
    def __init__(self, dim=1, keepdims=True):
        super().__init__()
        self.dim = dim
        self.keepdims = keepdims
    def forward(self, x):
        y = torch.mean(x, self.dim)
        if self.keepdims:
            y = y.unsqueeze(self.dim)
        return y

@register("spatialmaxpool")
class SpatialMaxPool(nn.Module):
    def __init__(self, dim=1, keepdims=True):
        super().__init__()
        self.dim = dim
        self.keepdims = keepdims
    def forward(self, x):
        y = torch.max(x, self.dim)[0]
        if self.keepdims:
            y = y.unsqueeze(self.dim)
        return y
    

class Covpool(Function):
    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h*w
        x = x.reshape(batchSize,dim,M)

        # change made here to save memory during validation time. Taken from: https://github.com/daitao/SAN/pull/29
        I_hat = torch.empty(M, M, device=x.device).fill_(-1./M/M)
        I_hat_diag = I_hat.diagonal()
        I_hat_diag += (1./M)

        # I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
        I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
        y = x.bmm(I_hat).bmm(x.transpose(1,2))

        ctx.save_for_backward(input, I_hat)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        input,I_hat = ctx.saved_tensors
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h*w
        x = x.reshape(batchSize,dim,M)
        grad_input = grad_output + grad_output.transpose(1,2)
        grad_input = grad_input.bmm(x).bmm(I_hat)
        grad_input = grad_input.reshape(batchSize,dim,h,w)
        return grad_input

@register("covarpool")
class CovarPool(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return Covpool.apply(x)

@register("global_std_pool2d")
class GlobalSTDPool2D(nn.Module):
    """2D global standard variation pooling
    https://github.com/buyizhiyou/NRVQA/blob/master/VSFA/CNNfeatures.py#L87
    """
    def __init__(self, dim=2, keepdim=True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        
    def  forward(self, x):
        return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                        dim=self.dim, keepdim=self.keepdim)

@register("temporalpool")
class TemporalPool(nn.Module):
    """  subjectively-inspired temporal pooling
    https://github.com/buyizhiyou/NRVQA/blob/master/VSFA/VSFA.py
    """
    def __init__(self, tau=12, beta=0.5):
        super().__init__()
        self.tau = tau
        self.beta = beta
                
    def forward(self, x):
        """subjectively-inspired temporal pooling"""
        q = torch.unsqueeze(torch.t(x), 0)
        qm = -torch.ones((1, 1, self.tau-1)) \
                .mul(float('inf')) \
                    .to(q.device)
        qp = torch.ones((1, 1, self.tau - 1)) \
                .mul(10000.0) \
                    .to(q.device)  #
        l = -F.max_pool1d(
                torch.cat((qm, -q), 2), 
                self.tau, 
                stride=1)
        m = F.avg_pool1d(
                torch.cat((
                    q * torch.exp(-q), 
                    qp * torch.exp(-qp)
                ), 2), 
                self.tau, 
                stride=1)
        n = F.avg_pool1d(
                torch.cat((torch.exp(-q), torch.exp(-qp)), 2), 
                self.tau, 
                stride=1)
        m = m / n
        return self.beta * m + (1 - self.beta) * l
    
@register("channelpool")
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

@register("maskedaveragepool")
class MaskedAveragePool(nn.Module):
    def __init__(self, interpolation="bilinear"):
        super().__init__()
        self.interpolation = interpolation
        
    def forward(self, x, mask):
        """Resize feature `x` to match `mask` tensor via 
        `F.interpolate`. This allows for element-wise 
        multiplication with the mask and features,
        then channel-wise summation (along the spatial (2,3) dimensions).
        The final output is the element-wise division of each
        feature channel sum by the original mask channel sum.

        Args:
            x (Tensor): [description]
            mask (Tensor): [description]

        Returns:
            Tensor: [description]
        """
        feature = F.interpolate(x, 
                                size=mask.shape[-2:], 
                                mode=self.interpolation, 
                                align_corners=True)
        masked_feature = torch.sum(feature * mask[:, None, ...], dim=(2, 3)) \
                         / (mask[:, None, ...].sum(dim=(2, 3)) + 1e-5)
        return masked_feature
    


class ConvMeanPool(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding=0, use_sn: bool=False):
        super().__init__(
            sn_wrapper(nn.Conv2d(in_channels, out_channels, kernel_size, bias, padding=padding), use_sn),
            nn.AvgPool2d((2,2)),
        )
            
class MeanPoolConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding=0, use_sn: bool=False):
        super().__init__(
            nn.AvgPool2d((2,2)),
            sn_wrapper(nn.Conv2d(in_channels, out_channels, kernel_size, bias, padding=padding), use_sn),
        )