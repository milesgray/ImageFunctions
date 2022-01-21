import gc, os, math, yaml
import pathlib, importlib
from typing import Union
from functools import partial

from tqdm.notebook import tqdm

import torch
import torch.nn
from torch.utils.data import DataLoader

import ImageFunctions.optimizers as optimizers
import ImageFunctions.datasets as datasets
import ImageFunctions.losses as losses
import ImageFunctions.models as models
import ImageFunctions.utility as utils


class TrainingEngine:
    def __init__(self, dataset: str, backbone: str, head: str,
                 id: str=None,
                 resume_id: str=None,
                 teach_id: str=None,
                 comet=None, 
                 log=print):
        """Loads a config file based on the `dataset`, `backbone`, and `head`
        and builds a model to train based on the contents.

        Args:
            dataset (str): [description]
            backbone (str): [description]
            head (str): [description]
            id (str, optional): [description]. Defaults to None.
            resume_id (str, optional): [description]. Defaults to None.
            teach_id (str, optional): [description]. Defaults to None.
            comet ([type], optional): [description]. Defaults to None.
            log ([type], optional): [description]. Defaults to print.
        """
        self.comet = comet
        self.log = log        
        self.args = args
        self.dataset = args["dataset"]
        self.backbone = args["backbone"]
        self.head = args["head"]
        self.id = args["id"]
        self.resume_id = args["resume_id"] if "resume_id" in args else None
        self.teacher_id = args["teach_id"] if "teach_id" in args else None
        
        self.comet.log_parameters(args)

        os.environ['CUDA_VISIBLE_DEVICES'] = args["gpu"]

        with open(args["config"], 'r') as f:
            self.config = yaml.load(f, Loader=yaml.loader.SafeLoader)
            if self.resume_id:
                self.config["resume"] = f"/content/gdrive/My Drive/data/models/liif/{self.resume_id}/epoch-last.pth"
            self.log('config loaded.')
            
        save_name = args["name"]
        if save_name is None:
            save_name = '_' + args["config"].split('/')[-1][:-len('.yaml')]
        if args["tag"] is not None:
            save_name += '_' + args["tag"]
        self.save_path = pathlib.Path('/content/gdrive/My Drive/data/models/liif/') / save_name
        if "resume" in self.config and self.config["resume"] is not None:
            self.comet.log_asset(self.config['resume'])
            
        self.save_path.mkdir(parents=True, exist_ok=True)
        with open(self.save_path / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        if self.config.get('data_norm') is None:
            self.config['data_norm'] = {
                'inp': {'sub': [0], 'div': [1]},
                'gt': {'sub': [0], 'div': [1]}
            }
        self.comet.log_parameters(self.config)
        
        comp = self.build()
        self.model = comp["model"]
        self.optimizer = comp["optimizer"]
        self.d_model = comp["d_model"]
        self.d_optimizer = comp["d_optimizer"]
        self.epoch = comp["epoch"]
        self.lr_scheduler = comp["lr_scheduler"]
        
    def build(self):
        comp = {}
        d_model = None
        d_optimizer = None
        max_val_v = 0
        if self.config.get('resume') is not None:
            comp = load_checkpoint()
        else:
            comp["model"] = models.make(self.config['model'])
            if torch.cuda.is_available():
                comp["model"] = comp["model"].cuda()        
            comp["optimizer"] = optimizers.make(comp["model"].parameters(), self.config['optimizer'])

            epoch_start = 1
            if self.config.get('multi_step_lr') is None:
                comp["lr_scheduler"] = None
            else:
                comp["lr_scheduler"] = MultiStepLR(comp["optimizerz"], **self.config['multi_step_lr'])
            
            if "d_model" in self.config:
                comp["d_model"] = models.make(self.config['d_model'])
                if torch.cuda.is_available():            
                    comp["d_model"] = d_model.cuda()
                comp["d_optimizer"] = optimizers.make(list(model.parameters()) + list(comp["d_model"].parameters()), self.config['d_optimizer'])
            if "max_val_psnr" in self.config:
                comp["max_val_psnr"] = self.config["max_val_psnr"]

        if self.comet: self.comet.set_model_graph(str(comp['model']))
        self.log(f"model: #params={utils.compute_num_params(comp['model'], text=True)}")
        self.log(f"epoch start: {comp['epoch_star']}")
        
        if comp["d_model"] is not None:
            self.log('discriminator model: #params={}'.format(utils.compute_num_params(comp["d_model"], text=True)))

        return comp

    def load_checkpoint(self):
        if torch.cuda.is_available():
            sv_file = torch.load(self.config['resume'])
        else:            
            sv_file = torch.load(self.config['resume'], map_location=torch.device('cpu'))            
        
        self.config['model'] = sv_file['model_spec']
    
        model = sv_file['model']       
        if torch.cuda.is_available():
            model = model.cuda()
        if 'sd' in self.config['model']:
            del self.config['model']['sd']   
        
        self.config['optimizer'] = sv_file['optimizer_spec']
        optimizer = sv_file['optimizer']
        if 'sd' in self.config['optimizer']:
            del self.config['optimizer']['sd']
        
        epoch_start = sv_file['epoch'] + 1        
        
        # Load Discriminator Model if defined
        if 'd_model' in sv_file:
            d_model = models.make(sv_file['d_model'], load_sd=True)
            if torch.cuda.is_available():            
                d_model = d_model.cuda()
            d_optimizer = optimizers.make(list(model.parameters()) + list(d_model.parameters()), 
                                        sv_file['d_optimizer'], 
                                        load_sd=True)
        # Set previous max value of PSNR metric tracker
        if "max_val_v" in sv_file:
            max_val_v = sv_file["max_val_v"]
        elif "max_val_v" in self.config:
            max_val_v = self.config["max_val_v"]
        else:
            max_val_v = 1e-13

        # Resume at previous epoch
        epoch_start = sv_file['epoch'] + 1
        # Set LR to previous value

        if 'lr_scheduler' in sv_file:
            lr_scheduler = sv_file['lr_scheduler']
        else:
            if self.config.get('multi_step_lr') is None:
                lr_scheduler = None
            else:
                lr_scheduler = MultiStepLR(optimizer, **self.config['multi_step_lr'])
            for _ in range(epoch_start - 1):
                lr_scheduler.step()
        return {
            "model": model, 
            "optimizer": optimizer, 
            "epoch": epoch_start, 
            "lr_scheduler": lr_scheduler, 
            "d_model": d_model, 
            "d_optimizer": d_optimizer
        }
        
    def make_discriminator(self):
        d_model, d_optimizer = None, None
        if 'd_model' in self.config:
            d_model = models.make(self.config['d_model'])        
            if torch.cuda.is_available():            
                d_model = d_model.cuda()
            d_optimizer = optimizers.make(list(model.parameters()) + list(d_model.parameters()), self.config['d_optimizer'])
            self.log('model: #params={}'.format(utils.compute_num_params(d_model, text=True)))
        return d_model, d_optimizer
        
    def make_data_loader(self, spec, tag=''):
        if spec is None:
            return None

        dataset = datasets.make(spec['dataset'])
        dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

        log_msg('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            if hasattr(v, "shape"):
                log_msg('  {}: shape={}'.format(k, tuple(v.shape)))
            else:
                log_msg('  {}: len={}'.format(k, len(v)))

        loader = DataLoader(dataset, 
                            batch_size=spec['batch_size'],
                            shuffle=(tag == 'train'), 
                            num_workers=spec['num_workers'], 
                            pin_memory=spec['pin_memory'])
        return loader

    def make_data_loaders(self, config):
        train_loader = make_data_loader(config.get('train_dataset'), tag='train')
        val_loader = make_data_loader(config.get('val_dataset'), tag='val')
        return train_loader, val_loader
        
    def step(self, epoch=None, use_loss_tracker=False):
        self.model.train()

        val_results = {}
        for k,v in self.metric_fns.items():
            val_results[k] = 0

        for k,v in self.loss_dict.items():
            for name,args in v.items():
                self.loss_dict[k][name]["fn"] = losses.make(args["fn"])
                if use_loss_tracker:
                    self.loss_dict[k][name]["tracker"] = utils.LossTracker(name, 
                                fn=losses.make(args["fn"]),
                                experiment=self.comet,
                                weight=args["weight"],
                                warmup=1000)
                else:    
                    self.loss_dict[k][name]["tracker"] = utils.Averager()

        if use_loss_tracker:
            train_loss = utils.LossTracker("train", 
                                experiment=self.comet)
        else:
            train_loss = utils.Averager()
        

        data_norm = config['data_norm']
        norms = utils.make_img_coeff(data_norm)

        batch_num = 0
        with tqdm(train_loader, leave=False, desc='train') as bar:
            for batch in bar:
                results = {}
                if torch.cuda.is_available():
                    for k, v in batch.items():
                        batch[k] = v.cuda()

                inp = (batch['inp'] / norms["inp"]["div"]) - norms["inp"]["sub"]
                pred = self.model(inp, batch['coord'].clone(), batch['cell'])
                pred = torch.nan_to_num(pred)
                gt = (batch['gt'] / norms["gt"]["div"]) - norms["gt"]["sub"]
                
                pred_out = pred.unsqueeze(0)
                gt_inp = gt.unsqueeze(0)

                for k,v in self.loss_dict["q"].items():
                    loss = v["fn"](pred_out, gt_inp) * v["weight"]
                    v["tracker"].add(loss.item())
                    results[k] = loss
                    if self.comet: self.comet.log_metric(k, loss.item())

                # teacher
                if self.t_model is not None:
                    t_feat = self.t_model(inp).detach()
                    for k,v in self.loss_dict["teacher"].items():
                        loss = v["fn"](self.model.feat, t_feat) * v["weight"]
                        v["tracker"].add(loss.item())
                        results[k] = loss
                        if self.comet: self.comet.log_metric(k, loss.item())
                    
                target_shape = batch['hr'].shape[-2:]
                batch_size = batch['hr'].shape[0]
                coord, cell = utils.make_coord_cell(target_shape=target_shape, batch_size=batch_size)
                if torch.cuda.is_available():
                    coord = coord.cuda()
                    cell = cell.cuda()
                pred = self.model(inp, coord, cell)
                pred = torch.nan_to_num(pred)
                pred = pred.clamp_(0, 1)
                pred_out = utils.reshape(pred, target_shape)

                hr_inp = batch['hr'].clamp_(0, 1)

                for k,v in self.loss_dict["img"].items():
                    loss = v["fn"](pred_out, hr_inp) * v["weight"]
                    v["tracker"].add(loss.item())
                    results[k] = loss
                    if self.comet: self.comet.log_metric(k, loss.item())

                self.optimizer.zero_grad()

                loss_q = sum([v for k,v in results.items() if k in self.loss_dict["q"].keys()]) / \
                        len([v for k,v in results.items() if k in self.loss_dict["q"].keys()])
                loss_img = sum([v for k,v in results.items() if k in self.loss_dict["img"].keys()]) / \
                        len([v for k,v in results.items() if k in self.loss_dict["img"].keys()])
                loss_t = sum([v for k,v in results.items() if k in self.loss_dict["teacher"].keys()]) / \
                        len([v for k,v in results.items() if k in self.loss_dict["teacher"].keys()])

                loss = loss_q + loss_img + loss_t
                if use_loss_tracker:
                    results["TRAIN"] = train_loss(loss, None)
                else:
                    train_loss.add(loss.item())

                    loss.backward()
                self.optimizer.step()

                batch_num += 1
                if val_loader is not None:
                    if batch_num == config.get('batch_eval_first', 100) or batch_num % config.get('batch_eval', 1000) == 0:
                        val_results = validate(self.model, val_loader, config, 
                                            metric_fns=self.metric_fns,
                                            epoch=epoch,
                                            experiment=self.comet)
                
                for k,v in val_results.items():
                    results[k] = v

                #results = {k:v.item() for k,v in results.items() if isinstance(v, torch.Tensor)}
                results = {}
                for k,v in self.loss_dict.items():
                    for name, args in v.items():
                        results[f"{k}-{name}"] = args["tracker"].item()
                
                pred = None; loss = None
                bar.set_postfix(results)
        results["TRAIN"] = train_loss.item()
        if self.comet: self.comet.log_metric('loss', train_loss.item())
        return results