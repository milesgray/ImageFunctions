import torch
import torch.nn as nn

from .registry import register

def kl_divergence(p, q):
    p = F.softmax(p)
    q = F.softmax(q)
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    
    return s1 + s2

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

    