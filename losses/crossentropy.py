"""
CrossentropyND and TopKLoss are from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/ND_Crossentropy.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np

from argparse import Namespace
from .registry import register
from . import reduce_loss, weight_reduce_loss

@register("ce_nd")
class CrossentropyND(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)

        return super().forward(inp, target)

@register("topk_ce")
class TopKLoss(CrossentropyND):
    """self.metric_fn
    Network has to have NO LINEARITY!
    """
    def __init__(self, weight=None, ignore_index=-100, k=10):
        self.k = k
        super().__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super().forward(inp, target)
        num_voxels = np.prod(res.shape)
        res, _ = torch.topk(
            res.view((-1, )), 
            int(num_voxels * self.k / 100), 
            sorted=False)
        return res.mean()

@register("weighted_ce")
class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)
        wce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)

        return wce_loss(inp, target)

@register("weighted_ce_2")
class WeightedCrossEntropyLossV2(torch.nn.Module):
    """
    WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    Network has to have NO LINEARITY!
    copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L121
    """
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, net_output, gt):
        if self.weight is None:
            # compute weight
            shp_x = net_output.shape
            shp_y = gt.shape
            #print(shp_x, shp_y)
            with torch.no_grad():
                if len(shp_x) != len(shp_y):
                    gt = gt.view((shp_y[0], 1, *shp_y[1:]))

                if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                    # if this is the case then gt is probably already a one hot encoding
                    y_onehot = gt
                else:
                    gt = gt.long()
                    y_onehot = torch.zeros(shp_x)
                    if net_output.device.type == "cuda":
                        y_onehot = y_onehot.cuda(net_output.device.index)
                    y_onehot.scatter_(1, gt, 1)
            y_onehot = y_onehot.transpose(0,1).contiguous()
            class_weights = (torch.einsum("cbxyz->c", y_onehot).type(torch.float32) + 1e-10)/torch.numel(y_onehot)
            #print('class_weights', class_weights)
            class_weights = class_weights.view(-1)
        else:
            class_weights = self.weight
        gt = gt.long()
        num_classes = net_output.size()[1]
        # class_weights = self._class_weights(inp)

        i0 = 1
        i1 = 2

        while i1 < len(net_output.shape): # this is ugly but torch only allows to transpose two axes at once
            net_output = net_output.transpose(i0, i1)
            i0 += 1
            i1 += 1

        net_output = net_output.contiguous()
        net_output = net_output.view(-1, num_classes) #shape=(vox_num, class_num)

        gt = gt.view(-1,)
        return F.cross_entropy(net_output, gt) # , weight=class_weights

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, _stacklevel=5)
        flattened = self._flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights

    def _flatten(self, tensor):
        """Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
        (N, C, D, H, W) -> (C, N * D * H * W)
        """
        C = tensor.size(1)
        # new axis order
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
        transposed = transposed.contiguous()
        return transposed.view(C, -1)

def compute_edts_forPenalizedLoss(GT):
    """
    GT.shape = (batch_size, x,y,z)
    only for binary segmentation
    """
    GT = np.squeeze(GT)
    res = np.zeros(GT.shape)
    for i in range(GT.shape[0]):
        posmask = GT[i]
        negmask = ~posmask
        pos_edt = distance_transform_edt(posmask)
        pos_edt = (np.max(pos_edt)-pos_edt)*posmask 
        neg_edt =  distance_transform_edt(negmask)
        neg_edt = (np.max(neg_edt)-neg_edt)*negmask        
        res[i] = pos_edt/np.max(pos_edt) + neg_edt/np.max(neg_edt)
    return res

def nll_loss(x, y):
    """
    customized nll loss
    source: https://medium.com/@zhang_yang/understanding-cross-entropy-
    implementation-in-pytorch-softmax-log-softmax-nll-cross-entropy-416a2b200e34
    """
    loss = -x[range(y.shape[0]), target]
    return loss.mean()


# implementations reference - https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py
# paper - https://www.ijcai.org/Proceedings/2020/0305.pdf
@register("taylor_softmax")
class TaylorSoftmax(nn.Module):

    def __init__(self, dim=1, n=2):
        super().__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n+1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out
    
@register("label_smoothing")
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, use_softmax=False): 
        super().__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls = classes 
        self.dim = dim 
        self.use_softmax = use_softmax
        
    def forward(self, pred, target): 
        """Taylor Softmax and log are already applied on the logits"""
        if self.use_softmax:
            pred = pred.log_softmax(dim=self.dim) 
        with torch.no_grad(): 
            true_dist = torch.zeros_like(pred) 
            true_dist.fill_(self.smoothing / (self.cls - 1)) 
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        return torch.mean(
            torch.sum(
                -true_dist * pred, 
                dim=self.dim))
    
@register("taylor_ce")
class TaylorCrossEntropyLoss(nn.Module):

    def __init__(self, n=2, ignore_index=-1, reduction='mean', smoothing=0.2):
        super().__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lab_smooth = LabelSmoothingLoss(1, smoothing=smoothing)

    def forward(self, logits, labels):

        log_probs = self.taylor_softmax(logits).log()
        #loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
        #        ignore_index=self.ignore_index)
        loss = self.lab_smooth(log_probs, labels)
        return loss


# Code taken from https://github.com/fhopfmueller/bi-tempered-loss-pytorch/blob/master/bi_tempered_loss_pytorch.py

class ComputeNormalization(torch.autograd.Function):
    """
    Class implementing custom backward pass for compute_normalization. See compute_normalization.
    """
    @staticmethod
    def forward(ctx, activations, t, num_iters):
        if t < 1.0:
            normalization_constants = self.compute_normalization_binary_search(activations, t, num_iters)
        else:
            normalization_constants = self.compute_normalization_fixed_point(activations, t, num_iters)

        ctx.save_for_backward(activations, normalization_constants)
        ctx.t = t
        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t
        normalized_activations = activations - normalization_constants 
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output
        
        return grad_input, None, None

    @staticmethod
    def log_t(u, t):
        """Compute log_t for `u'."""
        if t==1.0:
            return u.log()
        else:
            return (u.pow(1.0 - t) - 1.0) / (1.0 - t)
    @staticmethod
    def exp_t(u, t):
        """Compute exp_t for `u'."""
        if t==1:
            return u.exp()
        else:
            return (1.0 + (1.0-t)*u).relu().pow(1.0 / (1.0 - t))
            
    @staticmethod
    def compute_normalization_fixed_point(activations, t, num_iters):

        """Returns the normalization value for each example (t > 1.0).
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (> 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same shape as activation with the last dimension being 1.
        """
        mu, _ = torch.max(activations, -1, keepdim=True)
        normalized_activations_step_0 = activations - mu

        normalized_activations = normalized_activations_step_0

        for _ in range(num_iters):
            logt_partition = torch.sum(
                    self.exp_t(normalized_activations, t), -1, keepdim=True)
            normalized_activations = normalized_activations_step_0 * \
                    logt_partition.pow(1.0-t)

        logt_partition = torch.sum(
                exp_t(normalized_activations, t), -1, keepdim=True)
        normalization_constants = - log_t(1.0 / logt_partition, t) + mu

        return normalization_constants

    @staticmethod
    def compute_normalization_binary_search(activations, t, num_iters):

        """Returns the normalization value for each example (t < 1.0).
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (< 1.0 for finite support).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """

        mu, _ = torch.max(activations, -1, keepdim=True)
        normalized_activations = activations - mu

        effective_dim = \
            torch.sum(
                    (normalized_activations > -1.0 / (1.0-t)).to(torch.int32),
                dim=-1, keepdim=True).to(activations.dtype)

        shape_partition = activations.shape[:-1] + (1,)
        lower = torch.zeros(shape_partition, dtype=activations.dtype, device=activations.device)
        upper = -log_t(1.0/effective_dim, t) * torch.ones_like(lower)

        for _ in range(num_iters):
            logt_partition = (upper + lower)/2.0
            sum_probs = torch.sum(
                    exp_t(normalized_activations - logt_partition, t),
                    dim=-1, keepdim=True)
            update = (sum_probs < 1.0).to(activations.dtype)
            lower = torch.reshape(
                    lower * update + (1.0-update) * logt_partition,
                    shape_partition)
            upper = torch.reshape(
                    upper * (1.0 - update) + update * logt_partition,
                    shape_partition)

        logt_partition = (upper + lower)/2.0
        return logt_partition + mu        

def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example. 
    Backward pass is implemented.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    return ComputeNormalization.apply(activations, t, num_iters)

def tempered_sigmoid(activations, t, num_iters = 5):
    """Tempered sigmoid function.
    Args:
      activations: Activations for the positive class for binary classification.
      t: Temperature tensor > 0.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    internal_activations = torch.stack([activations,
        torch.zeros_like(activations)],
        dim=-1)
    internal_probabilities = tempered_softmax(internal_activations, t, num_iters)
    return internal_probabilities[..., 0]

def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature > 1.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    if t == 1.0:
        return activations.softmax(dim=-1)

    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)

def bi_tempered_binary_logistic_loss(activations,
        labels,
        t1,
        t2,
        label_smoothing = 0.0,
        num_iters=5,
        reduction='mean'):

    """Bi-Tempered binary logistic loss.
    Args:
      activations: A tensor containing activations for class 1.
      labels: A tensor with shape as activations, containing probabilities for class 1
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing
      num_iters: Number of iterations to run the method.
    Returns:
      A loss tensor.
    """
    internal_activations = torch.stack([activations,
        torch.zeros_like(activations)],
        dim=-1)
    internal_labels = torch.stack([labels.to(activations.dtype),
        1.0 - labels.to(activations.dtype)],
        dim=-1)
    return bi_tempered_logistic_loss(internal_activations, 
            internal_labels,
            t1,
            t2,
            label_smoothing = label_smoothing,
            num_iters = num_iters,
            reduction = reduction)

def bi_tempered_logistic_loss(activations,
        labels,
        t1,
        t2,
        label_smoothing=0.0,
        num_iters=5,
        reduction = 'mean'):

    """Bi-Tempered Logistic Loss.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      labels: A tensor with shape and dtype as activations (onehot), 
        or a long tensor of one dimension less than activations (pytorch standard)
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing parameter between [0, 1). Default 0.0.
      num_iters: Number of iterations to run the method. Default 5.
      reduction: ``'none'`` | ``'mean'`` | ``'sum'``. Default ``'mean'``.
        ``'none'``: No reduction is applied, return shape is shape of
        activations without the last dimension.
        ``'mean'``: Loss is averaged over minibatch. Return shape (1,)
        ``'sum'``: Loss is summed over minibatch. Return shape (1,)
    Returns:
      A loss tensor.
    """

    if len(labels.shape)<len(activations.shape): #not one-hot
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels

    if label_smoothing > 0:
        num_classes = labels_onehot.shape[-1]
        labels_onehot = ( 1 - label_smoothing * num_classes / (num_classes - 1) ) \
                * labels_onehot + \
                label_smoothing / (num_classes - 1)

    probabilities = tempered_softmax(activations, t2, num_iters)

    loss_values = labels_onehot * log_t(labels_onehot + 1e-10, t1) \
            - labels_onehot * log_t(probabilities, t1) \
            - labels_onehot.pow(2.0 - t1) / (2.0 - t1) \
            + probabilities.pow(2.0 - t1) / (2.0 - t1)
    loss_values = loss_values.sum(dim = -1) #sum over classes

    if reduction == 'none':
        return loss_values
    if reduction == 'sum':
        return loss_values.sum()
    if reduction == 'mean':
        return loss_values.mean()


@register("cev3")
class CrossEntropyV3Loss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_kpos=False,
                 partial=False,
                 reduction='mean',
                 loss_weight=1.0,
                 thrds=None):
        super().__init__()
        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.use_kpos = use_kpos
        self.partial = partial
        self.reduction = reduction
        self.loss_weight = loss_weight
        if self.use_sigmoid and thrds is not None:
            self.thrds=inverse_sigmoid(thrds)
        else:
            self.thrds = thrds

        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = self.partial_cross_entropy
            else:
                self.cls_criterion = self.binary_cross_entropy
        elif self.use_kpos:
            self.cls_criterion = self.kpos_cross_entropy
        else:
            self.cls_criterion = self.cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.thrds is not None:
            cut_high_mask = (label == 1) * (cls_score > self.thrds[1])
            cut_low_mask = (label == 0) * (cls_score < self.thrds[0])
            if weight is not None:
                weight *= (1 - cut_high_mask).float() * (1 - cut_low_mask).float()
            else:
                weight = (1 - cut_high_mask).float() * (1 - cut_low_mask).float()

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls

    @classmethod
    def _squeeze_binary_labels(label):
        if label.size(1) == 1:
            squeeze_label = label.view(len(label), -1)
        else:
            inds = torch.nonzero(label >= 1).squeeze()
            squeeze_label = inds[:,-1]
        return squeeze_label

    @classmethod
    def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
        # element-wise losses
        if label.size(-1) != pred.size(0):
            label = self._squeeze_binary_labels(label)

        loss = F.cross_entropy(pred, label, reduction='none')

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss

    def _expand_binary_labels(labels, label_weights, label_channels):
        bin_labels = labels.new_full((labels.size(0), label_channels), 0)
        inds = torch.nonzero(labels >= 1).squeeze()
        if inds.numel() > 0:
            bin_labels[inds, labels[inds] - 1] = 1
        if label_weights is None:
            bin_label_weights = None
        else:
            bin_label_weights = label_weights.view(-1, 1).expand(
                label_weights.size(0), label_channels)
        return bin_labels, bin_label_weights
    
    @classmethod
    def binary_cross_entropy(pred, label,
                             weight=None,
                             reduction='mean',
                             avg_factor=None):
        if pred.dim() != label.dim():
            label, weight = self._expand_binary_labels(label, weight, pred.size(-1))

        # weighted element-wise losses
        if weight is not None:
            weight = weight.float()

        loss = F.binary_cross_entropy_with_logits(
            pred, label.float(), weight, reduction='none')
        loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

        return loss

    @classmethod
    def partial_cross_entropy(pred, label,
                              weight=None,
                              reduction='mean',
                              avg_factor=None):
        if pred.dim() != label.dim():
            label, weight = self._expand_binary_labels(label, weight, pred.size(-1))

        # weighted element-wise losses
        if weight is not None:
            weight = weight.float()

        mask = label == -1
        loss = F.binary_cross_entropy_with_logits(
            pred, label.float(), weight, reduction='none')
        if mask.sum() > 0:
            loss *= (1-mask).float()
            avg_factor = (1-mask).float().sum()

        # do the reduction for the weighted loss
        loss = weight_reduce_loss(loss, 
                reduction=reduction, avg_factor=avg_factor)

        return loss

    @classmethod
    def kpos_cross_entropy(pred, label, 
                           weight=None, 
                           reduction='mean', 
                           avg_factor=None):
        # element-wise losses
        if pred.dim() != label.dim():
            label, weight = self._expand_binary_labels(label, weight, pred.size(-1))

        target = label.float() / torch.sum(label, dim=1, keepdim=True).float()

        loss = - target * F.log_softmax(pred, dim=1)
        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(loss, 
                weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss


def inverse_sigmoid(Y):
    X = []
    for y in Y:
        y = max(y,1e-14)
        if y == 1:
            x = 1e10
        else:
            x = -np.log(1/y-1)
        X.append(x)

    return X

@register("resample")
class ResampleLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 loss_weight=1.0,
                 partial=False,
                 focal=dict(
                     focal=True,
                     balance_param=2.0,
                     gamma=2,
                 ),
                 CB_loss=dict(
                     CB_beta=0.9,
                     CB_mode='average_w'  # 'by_class', 'average_n', 'average_w', 'min_n'
                 ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 freq_file='./class_freq.pkl'):
        super().__init__()

        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        # reweighting function
        self.reweight_func = reweight_func

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.balance_param = focal['balance_param']

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']
        self.CB_mode = CB_loss['CB_mode']

        #self.class_freq = torch.from_numpy(np.asarray(
        #    mmcv.load(freq_file)['class_freq'])).float().cuda()
        #self.neg_class_freq = torch.from_numpy(
        #    np.asarray(mmcv.load(freq_file)['neg_class_freq'])).float().cuda()
        self.num_classes = self.class_freq.shape[0]
        self.train_num = self.class_freq[0] + self.neg_class_freq[0]
        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias / self.neg_scale

        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

        # print('\033[1;35m loading from {} | {} | {} | s\033[0;0m'.format(freq_file, reweight_func, logit_reg))
        # print('\033[1;35m rebalance reweighting mapping params: {:.2f} | {:.2f} | {:.2f} \033[0;0m'.format(self.map_alpha, self.map_beta, self.map_gamma))

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        weight = self.reweight_functions(label)

        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)

        if self.focal:
            logpt = - self.cls_criterion(
                cls_score.clone(), label, weight=None, reduction='none',
                avg_factor=avg_factor)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            pt = torch.exp(logpt)
            loss = self.cls_criterion(
                cls_score, label.float(), weight=weight, reduction='none')
            loss = ((1 - pt) ** self.gamma) * loss
            loss = self.balance_param * loss
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                      reduction=reduction)

        loss = self.loss_weight * loss
        return loss

    def reweight_functions(self, label):
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(label.float())
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(label.float())
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(label.float())
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / torch.max(weight)

        return weight

    def logit_reg_functions(self, labels, logits, weight=None):
        if not self.logit_reg:
            return logits, weight
        if 'init_bias' in self.logit_reg:
            logits += self.init_bias
        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale  + logits * labels
            weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum( gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def CB_weight(self, gt_labels):
        if  'by_class' in self.CB_mode:
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
        elif 'average_n' in self.CB_mode:
            avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / \
                    torch.sum(gt_labels, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, avg_n)).cuda()
        elif 'average_w' in self.CB_mode:
            weight_ = torch.tensor((1 - self.CB_beta)).cuda() / \
                      (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
            weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / \
                     torch.sum(gt_labels, dim=1, keepdim=True)
        elif 'min_n' in self.CB_mode:
            min_n, _ = torch.min(gt_labels * self.class_freq +
                                 (1 - gt_labels) * 100000, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, min_n)).cuda()
        else:
            raise NameError
        return weight

    def RW_weight(self, gt_labels, by_class=True):
        if 'sqrt' in self.reweight_func:
            weight = torch.sqrt(self.propotion_inv)
        else:
            weight = self.propotion_inv
        if not by_class:
            sum_ = torch.sum(weight * gt_labels, dim=1, keepdim=True)
            weight = sum_ / torch.sum(gt_labels, dim=1, keepdim=True)
        return weight


##############################
## From TIMM
# # https://github.com/rwightman/pytorch-image-models/blob/master/timm/loss/cross_entropy.py

@register("label_smooth_ce")
class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

@register("soft_target_ce")
class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

@register("jsd_ce")    
class JsdCrossEntropy(nn.Module):
    """ Jensen-Shannon Divergence + Cross-Entropy Loss
    Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781
    Hacked together by / Copyright 2020 Ross Wightman
    """
    def __init__(self, num_splits=3, alpha=12, smoothing=0.1):
        super().__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        if smoothing is not None and smoothing > 0:
            self.cross_entropy_loss = LabelSmoothingCrossEntropy(smoothing)
        else:
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, output, target):
        split_size = output.shape[0] // self.num_splits
        assert split_size * self.num_splits == output.shape[0]
        logits_split = torch.split(output, split_size)

        # Cross-entropy is only computed on clean images
        loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
        probs = [F.softmax(logits, dim=1) for logits in logits_split]

        # Clamp mixture distribution to avoid exploding KL divergence
        logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-7, 1).log()
        loss += self.alpha * sum([F.kl_div(
            logp_mixture, p_split, reduction='batchmean') for p_split in probs]) / len(probs)
        return loss