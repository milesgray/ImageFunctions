# https://arxiv.org/abs/2111.07224
# https://github.com/Bodhis4ttva/LHC_Net

import torch
from torch import nn
import numpy as np

class LocalMultiHeadChannelAttention(nn.Module):
    def __init__(self, out_channels, 
                 pool_size=3, 
                 head_dim=128, 
                 head_num=16, 
                 kernel_size=1, 
                 norm_c=0.1):
        super().__init__()
        self.pool_size = pool_size
        self.head_dim = head_dim
        self.head_num = head_num
        self.out_channels = out_channels
        self.res = int(np.sqrt(head_num * head_dim))
        self.kernel_size = kernel_size
        self.norm_c = norm_c
        
        self.pool_q = nn.AdaptiveAvgPool2d((self.res, self.res))
        self.pool_k = nn.AdaptiveMaxPool2d((self.res, self.res))
        self.w_q_k = [nn.Linear(self.head_dim, self.head_dim) for _ in range(self.head_num)]
        self.w_p = nn.Sequential(*[
            nn.Linear(self.head_dim, self.out_channels),
            nn.Sigmoid()
        ])
        self.w_v = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size)
        self.pool_v = nn.AdaptiveAvgPool2d((self.res, self.res))
        
        self.weight = nn.Parameter(torch.FloatTensor((0.0,)), requires_grad=True)
    
    def vector_scaled_dot_product_attention(self, q, k, v):
        scores = torch.matmul(q, k.T)       # [Batch, Heads, Channels, Channels]
        scores_p = scores.mean(3)           # [Batch, Heads, Channels]
        scores_p = self.w_p(scores_p)       # [Batch, Heads, Channels]
        scores_p = scores_p.unsqueeze(-1)   # [Batch, Heads, Channels, 1]
        norm_scores = torch.div(scores, 
                                torch.pow(k.shape[3].float(),
                                          self.norm_c + scores_p))
                                            # [Batch, Heads, Channels, Channels]
        weights = nn.Softmax(dim=3)(norm_scores)
        attentions = torch.matmul(weights, v)
        return attentions
    
    def forward(self, x):
        B = x.shape[0]
        C = self.out_channels
        R = self.res
        head_res_dim = (R*R)//self.head_num
        
        query = self.pool_q(x) \
                    .view(B, self.head_num, C, head_res_dim) # [Batch, Heads, Channels, Head Res Dim]
        q = [None] * self.head_num
        for i in range(self.head_num):
            q[i] = self.w_q_k[i](query[:, i, :, :]) \
                .unsqueeze(1)       # [Batch, 1, Channels, Head Res Dim]
        query = torch.cat(q, dim=1) # [Batch, Heads, Channels, Head Res Dim]
        
        key = self.pool_k(x) \
            .view(B, self.head_num, C, head_res_dim) # [Batch, Heads, Channels, Head Res Dim]
        k = [None] * self.head_num
        for i in range(self.head_num):
            k[i] = self.w_q_k[i](key[:, i, :, :]) \
                .unsqueeze(1)       # [Batch, 1, Channels, Head Res Dim]
        key = torch.cat(k, dim=1)   # [Batch, Heads, Channels, Head Res Dim]
        
        value = self.w_v(x)
        value = self.pool_v(value) \
            .view(B, self.head_num, C, head_res_dim) # [Batch, Heads, Channels, Head Res Dim]
        
        attention = self.vector_scaled_dot_product_attention(query, key, value)
        attention = attention.view(B, C, R, R)   # [Batch, Channels, Resolution, Resolution]
        
        return x + (attention * (1 + self.weight)) # [Batch, Channels, Resolution, Resolution]