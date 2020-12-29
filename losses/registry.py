import copy

from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, CTCLoss, NLLLoss, PoissonNLLLoss, KLDivLoss, BCELoss, BCEWithLogitsLoss
from torch.nn import MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss
from torch.nn import CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss, TripletMarginWithDistanceLoss

lookup = {
    'l1': L1Loss, 
    'mse': MSELoss, 
    'cross_entropy': CrossEntropyLoss, 
    'ctc': CTCLoss, 
    'nll': NLLLoss, 
    'poisson_nll': PoissonNLLLoss, 
    'kld': KLDivLoss, 
    'bce': BCELoss, 
    'bce_with_logits': BCEWithLogitsLoss,
    'margin_ranking': MarginRankingLoss, 
    'hinge_embedding': HingeEmbeddingLoss, 
    'multi_label_margin': MultiLabelMarginLoss, 
    'smooth_l1': SmoothL1Loss, 
    'soft_margin': SoftMarginLoss, 
    'multi_label_soft_margin': MultiLabelSoftMarginLoss,
    'cosine_embedding': CosineEmbeddingLoss, 
    'multi_margin': MultiMarginLoss, 
    'triplet_margin': TripletMarginLoss, 
    'triplet_margin_with_distance': TripletMarginWithDistanceLoss
}

def register(name):
    def decorator(cls):
        lookup[name] = cls
        return cls
    return decorator

def make(spec, args=None):
    if args is not None:
        target_args = copy.deepcopy(spec['args'])
        target_args.update(args)
    else:
        target_args = spec['args']
    target = lookup[spec['name']](param_list, **target_args)
    return target
