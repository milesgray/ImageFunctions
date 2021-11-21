# https://arxiv.org/abs/2111.07224
# https://github.com/Bodhis4ttva/LHC_Net

import torch
from torch import nn

class LocalMultiHeadChannelAttention(nn.Module):
    def __init__(self, out_channels, pool_size, head_dim, 
                 head_num, res, kernel_size, norm_c):
        super().__init__()
        self.pool_size = pool_size
        self.head_dim = head_dim
        self.head_num = head_num
        self.out_channels = out_channels
        self.res = res
        self.kernel_size = kernel_size
        self.norm_c = norm_c
        
        self.pool_q = nn.AvgPool2d(kernel_size=self.pool_size)
        self.pool_k = nn.MaxPool2d(kernel_size=self.pool_size)
        self.w_q_k = [nn.Linear(self.head_dim, self.head_dim) for _ in range(self.head_num)]
        self.w_p = nn.Sequential([
            nn.Linear(self.out_channels, self.out_channels),
            nn.Sigmoid()
        ])
        self.w_v = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size)
        self.pool_v = nn.AvgPool2d(3)
    
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
    
    def forward(self, x, weight=0):
        B = x.shape[0]
        C = self.out_channels
        R = self.res
        head_res_dim = (R*R)//self.head_num
        
        query = self.pool_q(x) \
            .view(B, R*R, C)\
                .permute((0, 2, 1)) \
                    .view(B, C, self.num_heads, head_res_dim) \
                        .permute((0, 2, 1, 3)) # [Batch, Heads, Channels, Head Res Dim]
        q = [None] * self.head_num
        for i in range(self.head_num):
            q[i] = self.w_q_k(query[:, i, :, :]) \
                .unsqueeze(1)       # [Batch, 1, Channels, Head Res Dim]
        query = torch.cat(q, dim=1) # [Batch, Heads, Channels, Head Res Dim]
        
        key = self.pool_k(x) \
            .view(B, R*R, C) \
                .permute((0,2,1)) \
                    .view(B, C, self.head_num, head_res_dim) \
                        .permute(0,2,1,3) # [Batch, Heads, Channels, Head Res Dim]
        k = [None] * self.head_num
        for i in range(self.head_num):
            k[i] = self.w_q_k(key[:, i, :, :]) \
                .unsqueeze(1)       # [Batch, 1, Channels, Head Res Dim]
        key = torch.cat(k, dim=1)   # [Batch, Heads, Channels, Head Res Dim]
        
        value = self.w_v(x)
        value = self.pool_v(value) \
            .view(B, R*R, C) \
                .permute((0,2,1)) \
                    .view(B, C, self.head_num, head_res_dim) \
                        .permute(0,2,1,3) # [Batch, Heads, Channels, Head Res Dim]
        
        attention = self.vector_scaled_dot_product_attention(query, key, value)
        attention = attention.permute((0,2,1,3)) \
            .view(B, C, R*R) \
                .permute((0,2,1)) \
                    .view(B, R, R, C)   # [Batch, Resolution, Resolution, Channels]
        
        return x + (attention * (1 + weight)) # [Batch, Resolution, Resolution, Channels]