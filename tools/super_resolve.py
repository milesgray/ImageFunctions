import time
import pathlib
import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class SuperResolver:
    def __init__(self, model,
                 name="test",
                 extension="png",
                 directory="/content/results",
                 amount=2.0,
                 padding=2,
                 logger=print,
                 experiment=None):
        self.log = logger
        self.model = model
        self.name = name
        self.ext = extension
        self.count = 0
        self.path = pathlib.Path(directory)
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            self.log(f"Created {self.path.absolute()}")
        self.amount = amount
        self.padding = int(padding * self.amount)
        self.experiment = experiment
        self.result_container = None
        self.img = None
        self.num_x_tiles = None
        self.num_y_tiles = None
        self.frames = []

    def load_img(self, img_path, 
                 patch_shape=(32,32),
                 verbose=True):
        try:
            if verbose: self.log(f"Using a patch size {patch_shape[0]}x{patch_shape[1]}")
            self.target_crop_size = (round(patch_shape[0] * self.amount), 
                                    round(patch_shape[1] * self.amount))
            if verbose: self.log(f"Target output is a patch size {self.target_crop_size[0]}x{self.target_crop_size[1]}")
            self.img = torchvision.io.read_image(img_path).permute(1,2,0).numpy()
            self.img = self.img[:self.img.shape[0]-
                                (self.img.shape[0] % patch_shape[0]),
                                :self.img.shape[1]-
                                (self.img.shape[1] % patch_shape[1]),
                                :]
            self.ratio = self.img.shape[0] / self.img.shape[1]
            if verbose: self.log(f"Loaded image with adjusted shape {self.img.shape}")
            self.num_x_tiles = self.img.shape[0] // patch_shape[0]
            self.num_y_tiles = self.img.shape[1] // patch_shape[1]
            if verbose: self.log(f"Splitting into {self.num_x_tiles} x tiles and {self.num_y_tiles} y tiles")
            final_x = round(self.img.shape[0] * 
                            (self.target_crop_size[0] / patch_shape[0]))
            final_y = round(self.img.shape[1] * 
                            (self.target_crop_size[1] / patch_shape[1]))
            self.result_container = np.empty((final_x, final_y, 3))
            if verbose: self.log(f"Final output to an image of size {final_x}x{final_y}")

            return True
        except Exception as e:
            if verbose: self.log(f"Failed to load image:\n{e}")
        
            return False
        
    def apply(self, img_path=None,
              patch_shape=(32,32),
              verbose=True):
        if img_path is not None:
            if not self.load_img(img_path, patch_shape=patch_shape, verbose=verbose):
                return False
        else:
            if verbose: self.log(f"Using already loaded image")
        
        try:
            for x in tqdm(range(self.num_x_tiles)):
                for y in range(self.num_y_tiles):
                    start_x = round(x * patch_shape[0])
                    start_y = round(y * patch_shape[1])

                    crop_start_x = max(0, start_x - self.padding)
                    crop_end_x = min(self.img.shape[0], 
                                    start_x + patch_shape[0] + self.padding)
                    crop_start_y = max(0, start_y - self.padding)
                    crop_end_y = min(self.img.shape[1],
                                    start_y + patch_shape[1] + self.padding)

                    img_crop = self.img[crop_start_x:crop_end_x, 
                                        crop_start_y:crop_end_y, 
                                        :]
                    img_crop = img_crop.transpose(2,1,0)

                    pred = self.pred(img_crop,
                                     target_shape=(self.target_crop_size[0] + 
                                                   (self.padding * 2),
                                                   self.target_crop_size[1] + 
                                                   (self.padding * 2)
                                    ),
                                    return_data=True)
                    pred = pred.squeeze().permute(1,2,0).cpu().numpy()
                    pred = pred[self.padding:-self.padding,
                                self.padding:-self.padding,
                                :]

                    result_start_x = round(x * self.target_crop_size[0])
                    result_start_y = round(y * self.target_crop_size[1])

                    result_start_x = max(0, result_start_x)
                    result_end_x = min(self.result_container.shape[0], 
                                       result_start_x + self.target_crop_size[0])
                    result_start_y = max(0, result_start_y)
                    result_end_y = min(self.result_container.shape[1],
                                       result_start_y + self.target_crop_size[1])
                    
                    self.result_container[result_start_x:result_end_x, 
                                          result_start_y:result_end_y, 
                                          :] = np.fliplr(np.rot90(pred, k=3))
        except Exception as e:
            if verbose: self.log(f"Failed to build output:\n{e}")
            return False

        self.count += 1

        return self.result_container

    def image(self, img_path=None,
              patch_shape=(32,32),
              do_naive=False,
              show=True,
              save=True,
              verbose=True):
        start_time = time.time()
        img = self.apply(img_path=img_path,
                         patch_shape=patch_shape,
                         verbose=verbose)
        try:
            show_img_dict = {}
            if do_naive:
                self.naive_img = torchvision.transforms.Resize((self.img.shape[1] * self.amount, self.img.shape[2] * self.amount))(
                    torch.Tensor(self.img)).numpy()
                show_img_dict["naive"] = self.naive_img

            if verbose: self.log(f"Finished model inference, saving result...")
            self.save_image(self.result_container, "upsampled")
            show_img_dict["upsampled"] = self.result_container

            if verbose: self.log(f"Saved upsampled image, now saving original image...")
            self.save_image(self.img, "original")
            show_img_dict["original"] = self.img

            if show: self.show_images(show_img_dict)
        except Exception as e:
            if verbose: self.log(f"Failed to save or show images:\n{e}")
            return False
        
        dt = time.time() - start_time
        if verbose: self.log(f"Complete! Took {dt // 60} minutes, {dt % 60:.2f} seconds")
        return True

    def video(self, filename, paths,
              patch_shape=(32,32), 
              verbose=False):
        try:
            timer = utils.Timer()
            total_imgs = len(paths)
            out_path = self.path / "video"/ filename
            out_path.mkdir(parents=True, exist_ok=True)
            self.frames = []
            inner_timer = utils.Timer()
            for i, p in enumerate(paths):
                img = self.apply(img_path=p,
                         patch_shape=patch_shape,
                         verbose=verbose)
            
                img = self.prep_image(img)
                if self.experiment: self.experiment.log_image(img.numpy(), pathlib.Path(p).stem)
                self.frames.append(img)
                if i % 10 == 0:
                    total_dt = timer.t()
                    total_dt_per = total_dt / i
                    est_remain = total_dt_per * (total_imgs - i)
                    dt = inner_timer.t()
                    dt_per = dt / 10
                    inner_timer.s()
                    self.log(f"{i}/{total_imgs} @ {utils.log.time_text(dt_per)}/img - {utils.time_text(est_remain)} estimated remaining")
            framestack = torch.stack(self.frames)
            torchvision.io.write_video(out_path, framestack)
            return True
        except Exception as e:
            if verbose: self.log(f"Video failed:\n{e}")
            return False

    def prep_image(self, data, verbose=False):
        try:
            if isinstance(data, np.ndarray):
                data = torch.Tensor(data)
            img = data.mul(255).to(torch.uint8)
            return img
        except Exception as e:
            if verbose: self.log(f"Failed to prep img ({data.shape})\n{e}\nReturning data instead!")
            return data

    def make_filename(self, prefix):
        result = f"{prefix}"
        result = f"{result}_{self.amount}"
        result = f"{result}_{self.name}"
        result = f"{result}_{self.count}"
        result = f"{result}.{self.ext}"
        return result

    def save_image(self, data, prefix,
                   verbose=False):
        try:
            img = self.prep_image(data)        
            img_name = self.make_filename(prefix)    
            img_path = str(self.path / img_name)
            torchvision.io.write_png(img.permute(2,0,1), img_path)
            if self.experiment: self.experiment.log_image(img_path, img_name)
        except Exception as e:
            self.log(f"Failed to save img ({data.shape})\n{e}\nDisplaying data instead!")
            self.show_fig(data)

    def show_images(self, imgdict, figsize=10):
        for title, img in imgdict.items():
            self.show_fig(img, title=title, figsize=figsize)

    def show_fig(self, data, title="", figsize=(10,8)):
        if isinstance(figsize, int):
            figsize = (figsize, int(figsize*self.ratio))
        plt.figure(figsize=figsize)
        plt.title(title)
        if isinstance(data, torch.Tensor):
            data = data.squeeze().cpu().to(torch.uint8).permute((2,1,0)).numpy()
        plt.imshow(data)

    def save_fig(self, pred, inp, figsize, name):
        try:
            plt.figure(figsize=figsize)
            if torch.cuda.is_available():
                plt.subplot(1,2,1)
                plt.imshow(pred.cpu().squeeze().numpy().transpose(1,2,0))
                plt.subplot(1,2,2)
                plt.imshow(inp.cpu().squeeze().numpy().transpose(1,2,0))
            else:
                plt.subplot(1,2,1)
                plt.imshow(pred.squeeze().numpy().transpose(1,2,0))
                plt.subplot(1,2,2)
                plt.imshow(inp.squeeze().numpy().transpose(1,2,0))
            path = f"/content/{name}_fig.png"
            plt.savefig(path)
            if self.experiment: self.experiment.log_image(path, name)
        except Exception as e:
            self.log(f"Failed save\n{e}")

    def pred(self, img,
             name="test",
             target_shape=(256,256),
             figsize=(4,4),
             batch_size=1,
             return_data=False,
             save_fig=False,
             show_fig=False):
        inp = torch.from_numpy(img).unsqueeze(0)
        data_norm = {
                'inp': {'sub': [0], 'div': [255]},
                'gt': {'sub': [0], 'div': [1]}
            }
        t = data_norm['inp']
        inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1)
        inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1)
        t = data_norm['gt']
        gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1)
        gt_div = torch.FloatTensor(t['div']).view(1, 1, -1)
        if torch.cuda.is_available():
            inp = inp.cuda()
            inp_sub = inp_sub.cuda()
            inp_div = inp_div.cuda()
            gt_sub = gt_sub.cuda()
            gt_div = gt_div.cuda()
        inp = (inp / inp_div) - inp_sub
        inp = inp.clamp_(0, 1)
        self.model.eval()
        coord, cell = self.make_coord_cell(target_shape=target_shape, batch_size=batch_size)
        if torch.cuda.is_available():
            inp = inp.cuda()
            coord = coord.cuda()
            cell = cell.cuda()
        with torch.no_grad():
            pred = self.model(inp, coord, cell)

        pred = (pred * gt_div) + gt_sub
        pred = pred.clamp_(0, 1)
        pred = self.reshape(pred, target_shape)

        if save_fig:
            self.save_fig(pred, inp, figsize, name)
        if show_fig:
            self.show_fig(pred)
        if return_data:
            return pred
        
    def make_coord(self, shape, ranges=None, flatten=True):
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

    def make_coord_cell(self, target_shape=(32, 32), batch_size=8):
        coord = self.make_coord(target_shape).repeat(batch_size, 1, 1)
        cell = torch.ones_like(coord)
        cell[..., 0] *= 2 / target_shape[1]
        cell[..., 1] *= 2 / target_shape[0]
        return coord, cell

    def reshape(self, pred, target_shape):
        ih, iw = target_shape
        s = math.sqrt(pred.shape[1] / (ih * iw))
        shape = [pred.shape[0], round(ih * s), round(iw * s), 3]
        pred = pred.view(*shape) \
            .permute(0, 3, 1, 2).contiguous()
        return pred