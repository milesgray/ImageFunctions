import torch

from .registry import register

@register("ema")
class ModuleEMA(torch.nn.Module):
    def __init__(self, src, placement='same', momentum=0.9999):
        super().__init__()
        assert placement in ('same', 'cpu')
        self.src = [src]
        self.placement = placement
        self.momentum = momentum
        self.dst = copy.deepcopy(src)
        self.dst.zero_grad()
        self.dst.eval()
        if placement == 'cpu':
            self.dst = self.dst.cpu()

    def update(self):
        with torch.no_grad():
            for prop in ('named_parameters', 'named_buffers'):
                for (ns, ps), (nd, pd) in zip(getattr(self.src[0], prop)(), getattr(self.dst, prop)()):
                    assert ns == nd, f'{prop} mismatch: ns="{ns}" and nd="{nd}"'
                    if ns.endswith('.num_batches_tracked'):
                        continue
                    assert hasattr(ps, 'data') and hasattr(pd, 'data'), f'{nd} has no .data'
                    assert torch.is_floating_point(ps.data) and torch.is_floating_point(pd.data), f'{nd} not a float'
                    ps_data = ps.data
                    if self.placement == 'cpu':
                        ps_data = ps_data.cpu()
                    pd.data *= self.momentum
                    pd.data += ps_data * (1 - self.momentum)

    def forward(self, *args, cuda=True, **kwargs):
        dst = self.dst
        if cuda and self.placement == 'cpu':
            dst = self.dst.cuda()
        with torch.no_grad():
            return dst.forward(*args, **kwargs)