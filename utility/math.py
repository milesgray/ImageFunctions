import numpy as np
import torch
from torch import Tensor


def compute_covariance(feats: Tensor) -> Tensor:
    """
    Computes empirical covariance matrix for a batch of feature vectors
    """
    assert feats.ndim == 2

    feats -= feats.mean(dim=0)
    cov_unscaled = feats.t() @ feats # [feat_dim, feat_dim]
    cov = cov_unscaled / (feats.size(0) - 1)

    return cov

def prime_factors(num):
    factors = []
    for i in range(2, int(np.sqrt(num)) + 1):
        while num % i == 0:
            factors.append(i)
            num //= i
    if num > 1:
        factors.append(num)
    return factors


def dim_factorize_simple(d, is_dim_dst):
    return prime_factors(d)


def get_ranks_tt(shape, max_rank):
    ranks_left = [1] + np.cumprod(list(shape)).tolist()
    ranks_right = list(reversed([1] + np.cumprod(list(reversed(shape))).tolist()))
    ranks_tt = [min(a, b, max_rank) for a, b in zip(ranks_left, ranks_right)]
    return ranks_tt