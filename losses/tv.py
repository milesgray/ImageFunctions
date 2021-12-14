import torch.nn as nn

from .registry import register

@register("tv")
class TVLoss(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.l1 = nn.L1Loss(reduction=reduction)

    def forward(self, x, y):
        grad_x_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        grad_x_y = x[:, :, 1:, :] - x[:, :, :-1, :]

        grad_y_x = y[:, :, :, 1:] - y[:, :, :, :-1]
        grad_y_y = y[:, :, 1:, :] - y[:, :, :-1, :]

        loss_x = self.l1(grad_x_x, grad_y_x)
        loss_y = self.l1(grad_x_y, grad_y_y)

        loss = self.weight * (loss_x + loss_y)

        return loss
