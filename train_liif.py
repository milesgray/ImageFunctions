""" Train for generating LIIF, from image to implicit representation.

    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""

import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import losses
import utils
from test import eval_psnr


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        if hasattr(v, "shape"):
            log('  {}: shape={}'.format(k, tuple(v.shape)))
        else:
            log('  {}: len={}'.format(k, len(v)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    d_model = None
    d_optimizer = None
    max_val_v = 0
    if config.get('resume') is not None:
        if torch.cuda.is_available():
            sv_file = torch.load(config['resume'])
        else:
            sv_file = torch.load(config['resume'], map_location=torch.device('cpu'))
        model = models.make(sv_file['model'], load_sd=True)
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Load Discriminator Model if defined
        if 'd_model' in sv_file:
            d_model = models.make(sv_file['d_model'], load_sd=True)
            if torch.cuda.is_available():            
                d_model = d_model.cuda()
            d_optimizer = utils.make_optimizer(list(model.parameters()) + list(d_model.parameters()), sv_file['d_optimizer'], load_sd=True)
        # Set previous max value of PSNR metric tracker
        if "max_val_v" in sv_file:
            max_val_v = sv_file["max_val_v"]
        elif "max_val_v" in config:
            max_val_v = config["max_val_v"]

        # Resume at previous epoch
        epoch_start = sv_file['epoch'] + 1
        # Set LR to previous value
        optimizer = utils.make_optimizer(model.parameters(), sv_file['optimizer'], load_sd=True)
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model'])
        if torch.cuda.is_available():
            model = model.cuda()
        optimizer = optimizers.make(model.parameters(), config['optimizer'])
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))

        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        log(f"epoch start: {epoch_start}")
        if "d_model" in config:
            d_model = models.make(config['d_model'])
            if torch.cuda.is_available():            
                d_model = d_model.cuda()
            d_optimizer = optimizers.make(list(model.parameters()) + list(d_model.parameters()), config['d_optimizer'])
        if "max_val_v" in config:
            max_val_v = config["max_val_v"]

    log(f"epoch start: {epoch_start}")
    log('LIIF model: #params={}'.format(utils.compute_num_params(model, text=True)))
    if d_model is not None:
        log('Discriminator model: #params={}'.format(utils.compute_num_params(d_model, text=True)))

    return model, optimizer, epoch_start, lr_scheduler, max_val_v, d_model, d_optimizer

def manual_make_discriminator(config):
    d_model = models.make(config['d_model'])
    if torch.cuda.is_available():            
        d_model = d_model.cuda()
    d_optimizer = optimizers.make(list(model.parameters()) + list(d_model.parameters()), config['d_optimizer'])
    log('model: #params={}'.format(utils.compute_num_params(d_model, text=True)))
    return d_model, d_optimizer


def train(train_loader, model, optimizer, d_model=None, d_opt=None):
    model.train()
    loss_weight = 1.0
    loss_bce_weight = 0.10
    loss_iqa_weight = 0.10
    pix_loss_fn = losses.make('l1') # nn.L1Loss() #PixelLoss() #nn.L1Loss()
    bce_loss_fn = losses.make('vsi') # VSILoss()
    iqa_loss_fn = losses.make('vif') # VIFLoss()
    adv_loss_fn = losses.make('non_saturating') # NonSaturatingLoss()
    fid_metric = piq.FID()

    grad_penalty_wgan_fn = losses.make('wgan_grad_penalty') # GradLoss('wgan')
    grad_penalty_r1_fn = losses.make('r1_grad_penalty') # GradLoss('r1')
    train_loss = utils.Averager()
    pix_loss = utils.Averager()
    bce_loss = utils.Averager()
    adv_loss = utils.Averager()
    iqa_loss = utils.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1)
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1)
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1)
    if torch.cuda.is_available():
        inp_sub = inp_sub.cuda()
        inp_div = inp_div.cuda()
        gt_sub = gt_sub.cuda()
        gt_div = gt_div.cuda()

    batch_num = 0
    for batch in tqdm(train_loader, leave=False, desc='train'):
        if torch.cuda.is_available():
            for k, v in batch.items():
                batch[k] = v.cuda()

        #coord_ = batch['coord'].clone()

        inp = (batch['inp'] - inp_sub) / inp_div
        pred = model(inp, batch['coord'].clone(), batch['cell'])
        gt = (batch['gt'] - gt_sub) / gt_div
        loss_pix = loss_weight *  pix_loss_fn(pred, gt)
        
        pix_loss.add(loss_pix.item())
        
        loss = loss_pix #+ loss_bce
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #target_shape = batch['hr'].shape[-2:]
        #batch_size = batch['hr'].shape[0]
        #coord, cell = make_coord_cell(target_shape=target_shape, batch_size=batch_size)
        #if torch.cuda.is_available():
        #    coord = coord.cuda()
        #    cell = cell.cuda()
        #pred = model(inp, coord, cell)
        #pred = pred.clamp_(0, 1)
        #pred = reshape(pred, target_shape)

        #real_feats = model.gen_feat(batch['hr'])
        #fake_feats = model.gen_feat(fake)

        #real_feats = real_feats.view(real_feats.shape[1], -1)
        #fake_feats = fake_feats.view(fake_feats.shape[1], -1)

        #loss_iqa = loss_iqa_weight * fid_metric(real_feats, fake_feats)

        #loss_iqa = loss_iqa_weight * iqa_loss_fn(pred, batch['hr'].clamp_(0, 1))
        #iqa_loss.add(loss_iqa.item())
        #loss_bce = loss_bce_weight * bce_loss_fn(pred, batch['hr']) # bce_loss_fn(pred.squeeze()[:7000,:], gt.squeeze()[:7000,:])
        #bce_loss.add(loss_bce.item())

        #total_iqa_loss = loss_iqa #+ loss_bce
        #optimizer.zero_grad()
        #total_iqa_loss.backward()
        #optimizer.step()

        if d_model is not None:
            batch_size = gt.shape[0]            
            
            if batch_num > 0 and batch_num % 5 == 0:                
                pred = model(inp, batch['coord'].clone(), batch['cell'])
                penalty = grad_penalty_wgan_fn(d_model, gt, pred)
                d_target = torch.from_numpy(np.array([[0]] * batch_size)).float()                
                #d_target += torch.rand_like(d_target) * 0.1
                if torch.cuda.is_available():
                    d_target = d_target.cuda()
                d_out = d_model(pred) #, coord=coord_.clone().view(-1, 2))
                d_loss = -1 * adv_loss_fn(d_out) # bce_loss_fn(d_out, d_target)
                d_loss = d_loss * penalty
                adv_loss.add(d_loss.item())
                d_opt.zero_grad()
                d_loss.backward()                
                d_opt.step()
            else:
                pred = model(inp, batch['coord'].clone(), batch['cell'])
                penalty = grad_penalty_wgan_fn(d_model, gt, pred)
                #dr_target = torch.from_numpy(np.array([[1]] * batch_size)).float()
                #df_target = torch.from_numpy(np.array([[0]] * batch_size)).float()
                #dc_target += torch.rand_like(dc_target) * 0.1
                #if torch.cuda.is_available():
                    #dc_target = dc_target.cuda()
                    #dr_target = dr_target.cuda()
                    #df_target = df_target.cuda()
                #pred = model.gen_feat(inp)
                #d_out = d_model(torch.cat([gt, pred.detach()]))
                #d_loss = bce_loss_fn(d_out, dc_target)
                d_real_out = d_model(gt)
                d_fake_out = d_model(pred.detach())
                d_real_loss = adv_loss_fn(d_real_out) #bce_loss_fn(d_real_out, dr_target)
                d_fake_loss = adv_loss_fn(-1 * d_fake_out) #bce_loss_fn(d_fake_out, df_target)
                d_loss = d_real_loss + d_fake_loss * penalty
                bce_loss.add(d_loss.item())
                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()
                
        batch_num += 1
        pred = None; loss = None

    return train_loss.item(), pix_loss.item(), bce_loss.item(), iqa_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_psnr(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
