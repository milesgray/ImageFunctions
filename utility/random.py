import random
from argparse import Namespace

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # sets the seed for cpu
    torch.cuda.manual_seed(seed) # Sets the seed for the current GPU.
    torch.cuda.manual_seed_all(seed) #  Sets the seed for the all GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False
    torch.set_deterministic(True)

def numpy_random_init(worker_id):
    process_seed = torch.initial_seed()
    base_seed    = process_seed - worker_id
    ss  = np.random.SeedSequence([worker_id, base_seed])
    np.random.seed(ss.generate_state(4))

def numpy_fix_init(worker_id):
    np.random.seed(2<<16 + worker_id)
    
numpy_init_dict = {
    "train": numpy_random_init,
    "val"  : numpy_fix_init,
    "test" : numpy_fix_init
}

def init_random_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_noise(dist: str, z_dim: int, batch_size: int, args: Namespace=None) -> Tensor:
    assert dist in {'normal', 'uniform'}, f'Unknown latent distribution: {dist}'

    if dist == 'normal':
        if not correction is None and correction.enabled and correction.type == 'truncated':
            r = correction.kwargs.truncation_factor
            z = truncnorm.rvs(a=-r, b=r, size=(batch_size, z_dim))
            z = torch.from_numpy(z).float()
        else:
            z = torch.randn(batch_size, z_dim)

            if not correction is None and correction.enabled and correction.type == 'projected':
                # https://math.stackexchange.com/questions/827826/average-norm-of-a-n-dimensional-vector-given-by-a-normal-distribution
                norm = (1 / np.sqrt(2)) * z_dim * gamma((z_dim + 1)/2) / gamma((z_dim + 2)/2)
                # norm = np.sqrt(z_dim) # A fast approximation
                z /= z.norm(dim=1, keepdim=True)
                z *= norm
    elif dist == 'uniform':
        assert correction is None or correction.enabled == False, f"Unimplemented correction for uniform dist: {correction}"
        z = torch.rand(batch_size, z_dim) * 2 - 1

    return z
