from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange

from .registry import register

@register("hrvit_attn")
class HRViTAttention(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        dim: int = 64,
        heads: int = 2,
        ws: int = 1,  # window size
        qk_scale: Optional[float] = None,
        proj_drop: float = 0.0,
        with_cp: bool = False,
    ):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."
        self.in_dim = in_dim
        self.dim = dim
        self.heads = heads
        self.dim_head = dim // heads
        self.ws = ws
        self.with_cp = with_cp

        self.to_qkv = nn.Linear(in_dim, 2 * dim)

        self.scale = qk_scale or self.dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.attn_act = nn.Hardswish(inplace=True)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop),
        )

        self.attn_bn = nn.BatchNorm1d(
            dim, momentum=1 - 0.9 ** 0.5 if self.with_cp else 0.1
        )
        nn.init.constant_(self.attn_bn.bias, 0)
        nn.init.constant_(self.attn_bn.weight, 0)

        self.parallel_conv = nn.Sequential(
            nn.Hardswish(inplace=False),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,
            ),
        )

    @lru_cache(maxsize=4)
    def _generate_attn_mask(self, h: int, hp: int, device):
        x = torch.empty(hp, hp, device=device).fill_(-100.0)
        x[:h, :h] = 0
        return x

    def _cross_shaped_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        H: int,
        W: int,
        HP: int,
        WP: int,
        ws: int,
        horizontal: bool = True,
    ):
        B, N, C = q.shape
        if C < self.dim_head:  # half channels are smaller than the defined dim_head
            dim_head = C
            scale = dim_head ** -0.5
        else:
            scale = self.scale
            dim_head = self.dim_head

        if horizontal:
            q, k, v = map(
                lambda y: y.reshape(B, HP // ws, ws, W, C // dim_head, -1)
                .permute(0, 1, 4, 2, 3, 5)
                .flatten(3, 4),
                (q, k, v),
            )
        else:
            q, k, v = map(
                lambda y: y.reshape(B, H, WP // ws, ws, C // dim_head, -1)
                .permute(0, 2, 4, 3, 1, 5)
                .flatten(3, 4),
                (q, k, v),
            )

        attn = q.matmul(k.transpose(-2, -1)).mul(
            scale
        )  # [B,H_2//ws,W_2//ws,h,(b1*b2+1)*(ws*ws),(b1*b2+1)*(ws*ws)]

        ## need to mask zero padding before softmax
        if horizontal and HP != H:
            attn_pad = attn[:, -1:]  # [B, 1, num_heads, ws*W, ws*W]
            mask = self._generate_attn_mask(
                h=(ws - HP + H) * W, hp=attn.size(-2), device=attn.device
            )  # [ws*W, ws*W]
            attn_pad = attn_pad + mask
            attn = torch.cat([attn[:, :-1], attn_pad], dim=1)

        if not horizontal and WP != W:
            attn_pad = attn[:, -1:]  # [B, 1, num_head, ws*H, ws*H]
            mask = self._generate_attn_mask(
                h=(ws - WP + W) * H, hp=attn.size(-2), device=attn.device
            )  # [ws*H, ws*H]
            attn_pad = attn_pad + mask
            attn = torch.cat([attn[:, :-1], attn_pad], dim=1)

        attn = self.attend(attn)

        attn = attn.matmul(v)  # [B,H_2//ws,W_2//ws,h,(b1*b2+1)*(ws*ws),D//h]

        attn = rearrange(
            attn,
            "B H h (b W) d -> B (H b) W (h d)"
            if horizontal
            else "B W h (b H) d -> B H (W b) (h d)",
            b=ws,
        )  # [B,H_1, W_1,D]
        if horizontal and HP != H:
            attn = attn[:, :H, ...]
        if not horizontal and WP != W:
            attn = attn[:, :, :W, ...]
        attn = attn.flatten(1, 2)
        return attn

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        def _inner_forward(x: Tensor) -> Tensor:
            B = x.shape[0]
            ws = self.ws
            qv = self.to_qkv(x)
            q, v = qv.chunk(2, dim=-1)

            v_conv = (
                self.parallel_conv(v.reshape(B, H, W, -1).permute(0, 3, 1, 2))
                .flatten(2)
                .transpose(-1, -2)
            )

            qh, qv = q.chunk(2, dim=-1)
            vh, vv = v.chunk(2, dim=-1)
            kh, kv = vh, vv  # share key and value

            # padding to a multple of window size
            if H % ws != 0:
                HP = int((H + ws - 1) / ws) * ws
                qh = (
                    F.pad(
                        qh.transpose(-1, -2).reshape(B, -1, H, W),
                        pad=[0, 0, 0, HP - H],
                    )
                    .flatten(2, 3)
                    .transpose(-1, -2)
                )
                vh = (
                    F.pad(
                        vh.transpose(-1, -2).reshape(B, -1, H, W),
                        pad=[0, 0, 0, HP - H],
                    )
                    .flatten(2, 3)
                    .transpose(-1, -2)
                )
                kh = vh
            else:
                HP = H

            if W % ws != 0:
                WP = int((W + ws - 1) / ws) * ws
                qv = (
                    F.pad(
                        qv.transpose(-1, -2).reshape(B, -1, H, W),
                        pad=[0, WP - W, 0, 0],
                    )
                    .flatten(2, 3)
                    .transpose(-1, -2)
                )
                vv = (
                    F.pad(
                        vv.transpose(-1, -2).reshape(B, -1, H, W),
                        pad=[0, WP - W, 0, 0],
                    )
                    .flatten(2, 3)
                    .transpose(-1, -2)
                )
                kv = vv
            else:
                WP = W

            attn_h = self._cross_shaped_attention(
                qh,
                kh,
                vh,
                H,
                W,
                HP,
                W,
                ws,
                horizontal=True,
            )
            attn_v = self._cross_shaped_attention(
                qv,
                kv,
                vv,
                H,
                W,
                H,
                WP,
                ws,
                horizontal=False,
            )

            attn = torch.cat([attn_h, attn_v], dim=-1)
            attn = attn.add(v_conv)
            attn = self.attn_act(attn)

            attn = self.to_out(attn)
            attn = self.attn_bn(attn.flatten(0, 1)).view_as(attn)
            return attn

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

    def extra_repr(self) -> str:
        s = f"window_size={self.ws}"
        return s