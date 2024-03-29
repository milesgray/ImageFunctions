import math
import os
from argparse import Namespace

import torch
from torch.optim import Optimizer

from .registry import register

@register('adam_sgd_weighted')
def make_adam_sgd_weighted(params, lr=1e-3, weight_decay=0,
                           betas=(0.9, 0.999), eps=1e-8, amsgrad=False,
                           momentum=0, dampening=0, nesterov=False,
                           adam_w=0.5, sgd_w=0.5):
    args = Namespace()
    args.params = params
    args.lr = lr
    args.betas = betas
    args.eps = eps
    args.weight_decay = weight_decay
    args.amsgrad = amsgrad
    args.momentum = momentum
    args.dampening = dampening
    args.nesterov = nesterov
    args.adam_w = adam_w
    args.sgd_w = sgd_w

    return AdamSGDWeighted(args)

class AdamSGDWeighted(Optimizer):
    r"""Implements Adam and SGD mix algorithm.

    ```python
        params = list(model.parameters())
        optimizer = AdamSGDWeighted(params, lr)
    ```
    """

    def __init__(self, args):
        params = args.params
        lr = args.lr
        betas = args.betas
        eps = args.eps
        weight_decay = args.weight_decay
        amsgrad = args.amsgrad
        momentum = args.momentum
        dampening = args.dampening
        nesterov = args.nesterov
        adam_w = args.adam_w
        sgd_w = args.sgd_w
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr, weight_decay=weight_decay,
            betas=betas, eps=eps, amsgrad=amsgrad,
            momentum=momentum, dampening=dampening, nesterov=nesterov,
            adam_w=adam_w, sgd_w=sgd_w
        )
        super(AdamSGDWeighted, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamSGDWeighted, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamSGDWeighted does not support sparse gradients')

                d_p_adam, step_size = self.adam_step(grad, group, p)

                d_p_sgd = self.sgd_step(grad, group, p)

                megred_d_p = group['sgd_w'] * d_p_sgd + group['adam_w'] * d_p_adam
                # print(f'[{d_p_adam}, {d_p_sgd}, {megred_d_p}],')
                merged_lr = group['sgd_w'] * group['lr'] + group['adam_w'] * step_size

                p.add_(megred_d_p, alpha=-merged_lr)

        return loss

    def adam_step(self, grad, group, p):
        amsgrad = group['amsgrad']
        state = self.state[p]
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']
        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        if group['weight_decay'] != 0:
            grad = grad.add(p, alpha=group['weight_decay'])
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        step_size = group['lr'] / bias_correction1

        d_p = exp_avg / denom
        return d_p, step_size

    def sgd_step(self, grad, group, p):
        d_p = grad
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        if weight_decay != 0:
            d_p = d_p.add(p, alpha=weight_decay)
        if momentum != 0:
            param_state = self.state[p]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
            else:
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        return d_p
