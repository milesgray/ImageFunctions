#  https://github.com/Alibaba-MIIL/ASL
import torch
import torch.nn as nn
from torch import Tensor

from .registry import register

@register("cyclical")
def create_loss(gamma_pos: float, gamma_neg: float, gamma_hc: float, epochs: int, factor: int,
                eps: float=0.1, reduction: str='mean'):
    print('Loading Cyclical Focal Loss.')
    print(f"gamma_pos={gamma_pos} gamma_neg={gamma_neg} gamma_hc={gamma_hc} epochs={epochs} factor={factor}")
    if gamma_hc == 0:
        if gamma_pos == gamma_neg:
            return ASL_FocalLoss(gamma=gamma_pos, eps=eps, reduction=reduction)
        else:
            return ASLSingleLabel(gamma_pos=gamma_pos, gamma_neg=gamma_neg, 
                                  eps=eps, reduction=reduction)
    else:
        return Cyclical_FocalLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg,
                                  gamma_hc=gamma_hc, epochs=epochs, factor=factor, 
                                  eps=eps, reduction=reduction)

class ASLBase(nn.Module):
    def __init__(self,
                 gamma_pos: float=0,
                 gamma_neg: float=4,
                 eps: float=0.1,
                 reduction: str='mean'):
        super().__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.reduction = reduction
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def collapse(self, target: Tensor) -> Tensor:
        """Reduces rank of a `Tensor` to 1 if greater than 1
        by taking the `argmax` along the 2nd dimension.

        Args:
            target (Tensor): Data to collapse

        Returns:
            Tensor: Single rank tensor with argmax values or original tensor
        """
        if len(list(target.size()))>1:
            target = torch.argmax(target, 1)
        return target
    def make_target_classes(self, inputs: Tensor, target: Tensor) -> Tensor:
        return torch.zeros_like(inputs) \
            .scatter_(1,
                      target.long().unsqueeze(1),
                      1)
    def label_smoothing(self, num_classes: int):
        if self.eps > 0:
            self.targets_classes = self.targets_classes \
                .mul(1 - self.eps) \
                    .add(self.eps / num_classes)

    def loss_calc(self, log_preds: Tensor):
        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss

    def weights_calc(self, log_preds: Tensor) -> Tensor:
        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w
        return log_preds

@register("asl_single_label")
class ASLSingleLabel(ASLBase):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self,
                 gamma_pos: float=0,
                 gamma_neg: float=4,
                 eps: float=0.1,
                 reduction: str='mean'):
        super().__init__(gamma_pos=gamma_pos, gamma_neg=gamma_neg, eps=eps, reduction=reduction)

        print(f"ASLSingleLabel: gamma_pos={gamma_pos} gamma_neg={gamma_neg} eps={eps}")

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        target = self.collapse(target)
        self.targets_classes = self.make_target_classes(inputs, target)

        # ASL weights
        log_preds = self.weights_calc(log_preds)

        self.label_smoothing(num_classes)

        return self.loss_calc(log_preds)

@register("asl_focal_loss")
class ASL_FocalLoss(ASLBase):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self,
                 gamma: float=2,
                 eps: float=0.1,
                 reduction: str='mean'):
        super().__init__(gamma_pos=gamma, gamma_neg=gamma, eps=eps, reduction=reduction)

        print(f"ASL Focal Loss: gamma={gamma} eps={eps}")

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = self.make_target_classes(inputs, target)

        # ASL weights
        log_preds = self.weights_calc(log_preds)

        self.label_smoothing(num_classes)

        return self.loss_calc(log_preds)

@register("cyclical_focal_loss")
class Cyclical_FocalLoss(ASLBase):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self,
                 gamma_pos: float=0,
                 gamma_neg: float=4,
                 gamma_hc: float=0,
                 eps: float=0.1,
                 reduction: str='mean',
                 epochs: int=200,
                 factor: int=2):
        super().__init__(gamma_pos=gamma_pos, gamma_neg=gamma_neg, eps=eps, reduction=reduction)

        self.gamma_hc = gamma_hc
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.epochs = epochs
        self.factor = factor # factor=2 for cyclical, 1 for modified
        print(f"Asymetric Cyclical Focal Loss: gamma_pos={gamma_pos} gamma_neg={gamma_neg}",
              f" eps={eps} epochs={epochs} factor={factor}")

    def eta_calc(self, epoch: int):
        if self.factor * epoch < self.epochs:
            eta = 1 - self.factor * epoch / (self.epochs-1)
        else:
            eta = (self.factor * epoch / (self.epochs - 1) - 1.0)/(self.factor - 1.0)
        return eta

    def weights_calc(self, log_preds: Tensor, epoch: int) -> Tensor:
        eta = self.eta_calc(epoch)
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        positive_w = torch.pow(1 + xs_pos,
                               self.gamma_hc * targets)
        return log_preds * ((1 - eta)* asymmetric_w + eta * positive_w)

    def forward(self, inputs: Tensor, target: Tensor, epoch: int) -> Tensor:
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        target = self.collapse(target)
        self.targets_classes = self.make_target_classes(inputs, target)

        # Cyclical
        # eta = abs(1 - self.factor*epoch/(self.epochs-1))
        log_preds = self.weights_calc(log_preds, epoch)

        self.label_smoothing(num_classes)

        return self.loss_calc(log_preds)