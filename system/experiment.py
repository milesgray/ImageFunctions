import os
import yaml
import pathlib

class TrainingExperiment:
    def __init__(self, args, comet=None, log=print):
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
        
    #@title Prepare Training
    def prepare_training(self):
        d_model = None
        d_optimizer = None
        max_val_v = 0
        if self.config.get('resume') is not None:
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
        else:
            model = models.make(self.config['model'])
            if torch.cuda.is_available():
                model = model.cuda()        
            optimizer = optimizers.make(model.parameters(), self.config['optimizer'])

            epoch_start = 1
            if self.config.get('multi_step_lr') is None:
                lr_scheduler = None
            else:
                lr_scheduler = MultiStepLR(optimizer, **self.config['multi_step_lr'])
            
            if "d_model" in self.config:
                d_model = models.make(self.config['d_model'])
                if torch.cuda.is_available():            
                    d_model = d_model.cuda()
                d_optimizer = optimizers.make(list(model.parameters()) + list(d_model.parameters()), self.config['d_optimizer'])
            if "max_val_v" in self.config:
                max_val_v = self.config["max_val_v"]

        if self.comet: self.comet.set_model_graph(str(model))
        self.log(f"model: #params={utils.compute_num_params(model, text=True)}")
        self.log(f"epoch start: {epoch_start}")
        
        if d_model is not None:
            self.log('Discriminator model: #params={}'.format(utils.compute_num_params(d_model, text=True)))

        return model, optimizer, epoch_start, lr_scheduler, max_val_v, d_model, d_optimizer

    def make_discriminator(self):
        d_model, d_optimizer = None, None
        if 'd_model' in self.config:
            d_model = models.make(self.config['d_model'])        
            if torch.cuda.is_available():            
                d_model = d_model.cuda()
            d_optimizer = optimizers.make(list(model.parameters()) + list(d_model.parameters()), self.config['d_optimizer'])
            self.log('model: #params={}'.format(utils.compute_num_params(d_model, text=True)))
        return d_model, d_optimizer
        
    