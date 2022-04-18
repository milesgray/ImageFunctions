import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register

def kl_divergence(p, q):
    p = F.softmax(p)
    q = F.softmax(q)
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    
    return s1 + s2

def kl_div_2d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # D_KL(P || Q)
    batch, chans, height, width = p.shape
    unsummed_kl = F.kl_div(
        q.reshape(batch * chans, height * width).log(), p.reshape(batch * chans, height * width), reduction='none'
    )
    kl_values = unsummed_kl.sum(-1).view(batch, chans)
    return kl_values

def js_div_2d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # JSD(P || Q)
    m = 0.5 * (p + q)
    return 0.5 * kl_div_2d(p, m) + 0.5 * kl_div_2d(q, m)

def _reduce_loss(losses: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == 'none':
        return losses
    return torch.mean(losses) if reduction == 'mean' else torch.sum(losses)

@register("kl")
class KLloss(nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, pred, target):
        loss = kl_divergence(pred, target)
        return loss 
@register("kl_sparse")
class SparseKLloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('zero', torch.tensor(0.01, dtype=torch.float))

    def __call__(self,input):
        input = torch.sum(input, 0, keepdim=True)
        target_zero = self.zero.expand_as(input)
        loss = kl_divergence(target_zero, input)
        return loss
    
@register("prob_kl")
class ProbKLLoss(nn.Module):   
    def __init__(self):
        super().__init__()

    def forward(self, f_s, f_t):
        return self.cosine_similarity_loss(f_s, f_t)

    @staticmethod
    def cosine_similarity_loss(points_x, points_y, eps=0.0000001):
        # Normalize each vector by its norm        
        points_x = torch.nn.functional.normalize(points_x)        
        points_y = torch.nn.functional.normalize(points_y)        

        # Calculate the cosine similarity
        x_similarity = torch.mm(points_x, torch.transpose(points_x, 0, 1))
        y_similarity = torch.mm(points_y, torch.transpose(points_y, 0, 1))

        # Scale cosine similarity to 0..1
        x_similarity = (x_similarity + 1.0) / 2.0
        y_similarity = (y_similarity + 1.0) / 2.0

        # Transform them into probabilities
        x_similarity = x_similarity / torch.sum(x_similarity, dim=1, keepdim=True)
        y_similarity = y_similarity / torch.sum(y_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        loss = torch.mean(y_similarity * torch.log((y_similarity + eps) / (x_similarity + eps)))

        return loss
    
@register("antiuniform_kl")
class AntiUniformKLloss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self,input):
        input = torch.sum(input, 0, keepdim=True)
        target_uniform = torch.rand_like(input)
        loss = kl_divergence(target_uniform, input)
        return loss     



@register("jsd")
def JSLoss(input: torch.Tensor, target: torch.Tensor, reduction: str = 'mean'):
    r"""Calculate the Jensen-Shannon divergence loss between heatmaps.

    Args:
        input: the input tensor with shape :math:`(B, N, H, W)`.
        target: the target tensor with shape :math:`(B, N, H, W)`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.

    Examples:
        >>> input = torch.full((1, 1, 2, 4), 0.125)
        >>> loss = js_div_loss_2d(input, input)
        >>> loss.item()
        0.0
    """
    return _reduce_loss(js_div_2d(target, input), reduction)


@register("kld_2d")
def KL2dLoss(input: torch.Tensor, target: torch.Tensor, reduction: str = 'mean'):
    r"""Calculate the Kullback-Leibler divergence loss between heatmaps.

    Args:
        input: the input tensor with shape :math:`(B, N, H, W)`.
        target: the target tensor with shape :math:`(B, N, H, W)`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.

    Examples:
        >>> input = torch.full((1, 1, 2, 4), 0.125)
        >>> loss = kl_div_loss_2d(input, input)
        >>> loss.item()
        0.0
    """
    return _reduce_loss(kl_div_2d(target, input), reduction)