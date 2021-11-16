import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import datasets
import models
import utility

def plot_preds(model, loader, epoch, 
               save_dir="/content", 
               target_shape=(128,128), 
               batch_size=16, 
               return_data=False):
    batch = next(iter(loader))
    for k, v in batch.items():
        batch[k] = v.cuda()
    data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [0.5]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()    
    inp = (batch['inp'] / inp_div) - inp_sub
    #inp = inp.clamp_(0, 1)
    model.eval()
    
    coord, cell = make_coord_cell(target_shape=target_shape, batch_size=batch_size)
    with torch.no_grad():
        pred = model(inp, coord.cuda(), cell.cuda())

    pred = (pred * gt_div) + gt_sub
    pred = pred.clamp_(0, 1)
    pred = reshape(pred, target_shape)
    inp = batch['inp'].clamp_(0, 1)
    hr_gt = reshape(batch["hr"], batch["inp"].shape[-2:])
    plt.figure(figsize=(4, batch_size * 3))
    for i, (p, g) in enumerate(zip(pred, hr_gt)):
        plt.subplot(batch_size,3,(i * 3) + 1)
        plt.imshow(p.cpu().numpy().transpose(1,2,0))
        plt.subplot(batch_size,3,(i * 3) + 2)
        plt.imshow(g.cpu().numpy().transpose(1,2,0))
        plt.subplot(batch_size,3,(i * 3) + 3)
        plt.imshow(batch['inp'][i].cpu().numpy().transpose(1,2,0))
    plt.savefig(f"{save_dir}/testfig_{epoch}.png")
    if return_data:
        return pred, batch

def make_coord_cell(target_shape=(32, 32), batch_size=8):
    coord = make_coord(target_shape).repeat(batch_size, 1, 1)
    cell = torch.ones_like(coord)
    cell[..., 0] *= 2 / target_shape[1]
    cell[..., 1] *= 2 / target_shape[0]
    return coord, cell

def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

def reshape(pred, t_shape):
    ih, iw = t_shape
    s = math.sqrt(pred.shape[1] / (ih * iw))
    shape = [pred.shape[0], round(ih * s), round(iw * s), 3]
    pred = pred.view(*shape) \
        .permute(0, 3, 1, 2).contiguous()
    return pred

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n)
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def eval_psnr(loader, model, epoch, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, savedir="/content"):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utility.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utility.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utility.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utility.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        
        with torch.no_grad():
            pred = model(inp, batch['coord'], batch['cell'])

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))
    if eval_bsize:
        try:
            plot_preds(model, loader, epoch, save_dir=savedir,
                       batch_size=pred.shape[0])            
        except Exception as e:
            print(f"Failed to save validation preview\n{e}")
    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)
    print('result: {:.4f}'.format(res))
