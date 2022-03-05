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


class SuperResTrainingEngine:
    def __init__(self, args: dict,
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
        self.root_path = args["root_path"]        
        self.dataset = args["dataset"]
        self.backbone = args["backbone"]
        self.head = args["head"]
        self.id = args["id"]
        self.resume_id = args["resume_id"] if "resume_id" in args else None
        self.teacher_id = args["teach_id"] if "teach_id" in args else None
        
        self.comet.log_parameters({"init_args": args})

        os.environ['CUDA_VISIBLE_DEVICES'] = args["gpu"]

        self.init_config()
        self.make_save_path()
        
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
        
        self.loaders = self._make_data_loaders()
        
    def init_config(self):
        try:
            with open(self.args["config"], 'r') as f:
                self.config = yaml.load(f, Loader=yaml.loader.SafeLoader)
                if self.resume_id:
                    self.config["resume"] = f"{self.root_path}{self.resume_id}/epoch-last.pth"
                self.log(f'Config Initialized!  Loaded from {self.args["config"]}')
        except Exception as e:
            self.log(f"Failed to init config:\n{e}")
            
    def make_save_path(self):
        try:
            save_name = self.args["name"]
            if save_name is None:
                save_name = '_' + self.args["config"].split('/')[-1][:-len('.yaml')]
            if self.args["tag"] is not None:
                save_name += '_' + self.args["tag"]
            self.save_path = pathlib.Path(self.root_path) / save_name
            if "resume" in self.config and self.config["resume"] is not None:
                self.comet.log_asset(self.config['resume'])
                
            self.save_path.mkdir(parents=True, exist_ok=True)
            with open(self.save_path / 'config.yaml', 'w') as f:
                yaml.dump(self.config, f)
            return True
        except Exception as e:
            self.log(f"Failed to make save path:\n{e}")
            return False
           
    def build(self):
        comp = {}
        d_model = None
        d_optimizer = None
        max_val_v = 0
        if self.config.get('resume') is not None:
            comp = self.load_checkpoint()
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
        
    def _make_discriminator(self):
        d_model, d_optimizer = None, None
        if 'd_model' in self.config:
            d_model = models.make(self.config['d_model'])        
            if torch.cuda.is_available():            
                d_model = d_model.cuda()
            d_optimizer = optimizers.make(list(model.parameters()) + list(d_model.parameters()), self.config['d_optimizer'])
            self.log('model: #params={}'.format(utils.compute_num_params(d_model, text=True)))
        return d_model, d_optimizer
        
    def _make_data_loader(self, spec, tag=''):
        if spec is None:
            return None

        dataset = datasets.make(spec['dataset'])
        dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

        self.log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            if hasattr(v, "shape"):
                self.log('  {}: shape={}'.format(k, tuple(v.shape)))
            else:
                self.log('  {}: len={}'.format(k, len(v)))

        loader = DataLoader(dataset, 
                            batch_size=spec['batch_size'],
                            shuffle=(tag == 'train'), 
                            num_workers=spec['num_workers'], 
                            pin_memory=spec['pin_memory'])
        return loader

    def _make_data_loaders(self):
        train_loader = self._make_data_loader(self.config.get('train_dataset'), tag='train')
        val_loader = self._make_data_loader(self.config.get('val_dataset'), tag='val')
        return {"train":train_loader, "val":val_loader}
        
    def train(self, epoch_start=0):
        for epoch in range(epoch_start, self.config.get("epoch_max", epoch_start) + 1):
            timer.s()

            losses_dict = self.config["losses"].copy()
            if epoch > self.config.get("end_teach", 0) and "teacher" in losses_dict:
                del losses_dict["teacher"]
                
            train_results = self.step(epoch=epoch,
                                      use_loss_tracker=False,)
            if lr_scheduler is not None:
                lr_scheduler.step()
            msg = f"[Epoch {epoch}]  [{utils.time_text(timer.t())}]"
            msg = f"{msg}Results: {train_results['TRAIN']:.5f}"
            for k,v in train_results.items():
                if k != "TRAIN":
                    msg = f"{msg}\t[{k}] {v:.5f}"
            self.log(msg)

            self.save_models()
    
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
        

        data_norm = self.config['data_norm']
        norms = utils.make_img_coeff(data_norm)

        batch_num = 0
        with tqdm(self.loaders["train"], leave=False, desc='train') as bar:
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
                if epoch >= config["end_teach"] or "end_teach" not in self.config:
                    loss_q = sum([v for k,v in results.items() if k in self.loss_dict["q"].keys()]) / \
                             len([v for k,v in results.items() if k in self.loss_dict["q"].keys()])
                    loss_img = sum([v for k,v in results.items() if k in self.loss_dict["img"].keys()]) / \
                               len([v for k,v in results.items() if k in self.loss_dict["img"].keys()])
                    loss = loss_q + loss_img
                else:
                    loss_t = sum([v for k,v in results.items() if k in loss_dict["teacher"].keys()]) / \
                             len([v for k,v in results.items() if k in loss_dict["teacher"].keys()])
                    loss = loss_t
                if use_loss_tracker:
                    results["TRAIN"] = train_loss(loss, None)
                else:
                    train_loss.add(loss.item())

                    loss.backward()
                self.optimizer.step()

                batch_num += 1
                if self.val_loader is not None:
                    if batch_num == self.config.get('batch_eval_first', 100) or batch_num % config.get('batch_eval', 1000) == 0:
                        val_results = self.validate(self.model, self.val_loader, self.      config, 
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
    
    def validate(self, epoch, metric_fns):
        self.model.eval()

        norms = utils.make_img_coeff(self.config["data_norm"])

        results = {}
        for name, fn in metric_fns.items():
            results[name] = utils.Averager()

        for batch in self.loaders["val"]:
            if torch.cuda.is_available():
                for k, v in batch.items():
                    batch[k] = v.cuda()

            inp = (batch['inp'] - norms["inp"]["sub"]) / norms["inp"]["div"]

            target_shape = batch['hr'].shape[-2:]
            batch_size = batch['hr'].shape[0]
            coord, cell = utils.make_coord_cell(target_shape=target_shape, batch_size=batch_size)
            if torch.cuda.is_available():
                coord = coord.cuda()
                cell = cell.cuda()
            with torch.no_grad():
                pred = self.model(inp, coord, cell)
            pred = torch.nan_to_num(pred)
            pred = pred.clamp_(0, 1)
            pred = utils.reshape(pred, target_shape)
            for name, fn in metric_fns.items():
                if name in ["NIQE"]:
                    patch_size = target_shape[1] // 2
                    patch_size -= 1
                    results[name].add(fn(pred, patch_size=patch_size).item())
                else:
                    results[name].add(fn(pred, batch['hr']).item())

        if save_plot:
            try:
                self.plot(self.loaders["val"], epoch, 
                        save_dir=savedir,
                        batch_size=pred.shape[0],
                        count=eval_count,
                        data_norm=data_norm)
            except Exception as e:
                self.log(f"Failed to save validation preview\n{e}")
        
        for k, v in results.items():
            results[k] = v.item()
            if self.experiment: self.experiment.log_metric(k, v.item())

        return results
    
    def plot(self, losder, epoch,
                name="testfig", 
                save_dir="/content",
                fig_size=None,
                target_shape=None, 
                batch_size=1,
                count=1, 
                min_var=0.1,
                data_norm=None,
                return_data=False):
        inputs = []
        preds = []
        originals = []
        batches = []
        for i, batch in enumerate(loader):
            try:
                if torch.cuda.is_available():
                    for k, v in batch.items():
                        batch[k] = v.cuda()
                norms = utils.make_img_coeff(data_norm)
                gt = batch["hr"].clamp_(0, 1).contiguous()
                gt_var = gt.var()
                if gt_var < min_var: continue

                batch_shape = gt.shape[-2:] if target_shape is None else target_shape
                inp = (batch['inp'] / norms["inp"]["div"]) - norms["inp"]["sub"]
                self.model.eval()
                
                inp = inp.clamp_(0, 1)
                coord, cell = utils.make_coord_cell(target_shape=batch_shape, 
                                                    batch_size=batch_size)
                if torch.cuda.is_available():
                    coord = coord.cuda()
                    cell = cell.cuda()
                with torch.no_grad():
                    pred = self.model(inp, coord, cell)
                overflow_count = pred.abs()[pred > 1].float().sum().item()
                if experiment: experiment.log_metric("overflow output pixels", overflow_count)
                underflow_count = pred[pred < 0].float().sum().item()
                if experiment: experiment.log_metric("underflow output pixels", underflow_count)
                nan_count = pred.isnan().float().sum().item()
                if experiment: experiment.log_metric("nan output pixels", nan_count)
                pred = pred.clamp_(0, 1)
                pred = torch.nan_to_num(pred)
                pred = utils.reshape(pred, batch_shape)
                inp = batch['inp'].clamp_(0, 1).contiguous()
                
                inputs.append(inp[0])
                preds.append(pred[0])
                originals.append(gt[0])
                batches.append(batch)

                # can't use i because of potential skips
                if len(inputs) > count: break
            except Exception as e:
                log(f"Failed to calculate preview batch!\n{e}")
                try:
                    del batch
                except:
                    pass
                try:
                    del pred
                except:
                    pass
        
        try:
            plt.figure(figsize=fig_size if fig_size else (10, count * 2))
            for i, (inp, p, g) in enumerate(zip(inputs, preds, originals)): 
                if i >= count: break
                if torch.cuda.is_available():
                    inp = inp.cpu()
                    p = p.cpu()
                    g = g.cpu()
            
                plt.subplot(count,3,(i * 3) + 1)
                plt.imshow(inp.numpy().transpose(1,2,0))
                plt.subplot(count,3,(i * 3) + 2)
                plt.imshow(p.numpy().transpose(1,2,0))
                plt.subplot(count,3,(i * 3) + 3)
                plt.imshow(g.numpy().transpose(1,2,0))
            plt.savefig(f"{save_dir}/{name}_{epoch}.png")
            plt.close()
            experiment.log_image(f"{save_dir}/{name}_{epoch}.png")
        except Exception as e:
            self.log(f"Failed to save preview plot!\n{e}")
        if return_data:
            return preds, batches
        
    def save_models(self, epoch):
        try:
            model_spec = self.config['model']
            model_spec['sd'] = self.model.state_dict()
            optimizer_spec = self.config['optimizer']
            optimizer_spec['sd'] = self.optimizer.state_dict()
            sv_file = { 
                'model_spec': self.model.pec.copy(),
                'optimizer_spec': optimizer_spec.copy(),        
                'epoch': epoch,
                'max_val_v': max_val_v,
                'model': self.model,
                'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler,
                'config': self.config,
            }
            if self.d_model is not None:
                d_model_spec = self.config['d_model'].copy()
                d_model_spec['sd'] = self.d_model.state_dict()
                d_optimizer_spec = self.config['d_optimizer'].copy()
                d_optimizer_spec['sd'] = self.d_optimizer.state_dict()
                sv_file['d_model_spec'] = self.d_model_spec
                sv_file['d_optimizer_spec'] = d_optimizer_spec
                sv_file['d_model'] = self.d_model
                sv_file['d_optimizer'] = self.d_optimizer

            save_location = os.path.join(self.save_path, 'epoch-last.pth')
            torch.save(sv_file, save_location)
            self.comet.log_model(f"{epoch}", save_location)

            if epoch % self.config.get('epoch_save', 10) == 0:
                torch.save(sv_file,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
                
            return True
        except Exception as e:
            print(f"Failed to save models!\n{e}")
            return False