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
                 name="test.png",
                 directory="/content/",
                 amount=2.0,
                 logger=print,
                 experiment=None):
        self.log = logger
        self.model = model
        self.name = name
        self.path = pathlib.Path(directory)
        if not self.path.exists():
            self.path.mkdir(parents=True)
            self.log(f"Created {self.path.absolute()}")
        self.amount = amount
        self.experiment = experiment
        self.result_container = None
        self.img = None

    def apply(self, img_path,
              patch_shape=(32,32),
              save_intermediate=False,
              intermediate_size=(4,4),
              show=True,
              verbose=True):
        start_time = time.time()
        target_crop_size = (round(patch_shape[0] * self.amount), round(patch_shape[1] * self.amount))
        self.img = cv2.imread(img_path)
        self.img = self.img[:self.img.shape[0]-(self.img.shape[0] % patch_shape[0]),:self.img.shape[1]-(self.img.shape[1] % patch_shape[1]),:]
        num_x_tiles = self.img.shape[0] // patch_shape[0]
        num_y_tiles = self.img.shape[1] // patch_shape[1]
        if verbose: self.log(f"Loaded image with adjusted shape {self.img.shape}, splitting into {num_x_tiles} x tiles and {num_y_tiles} y tiles")
        final_x = round(self.img.shape[0] * (target_crop_size[0] / patch_shape[0]))
        final_y = round(self.img.shape[1] * (target_crop_size[1] / patch_shape[1]))
        self.result_container = np.empty((final_x, final_y, 3))
        if verbose: self.log(f"Upsampling to an image of size {final_x}x{final_y}...")

        for x in tqdm(range(num_x_tiles)):
            for y in range(num_y_tiles):
                start_x = round(x * patch_shape[0])
                start_y = round(y * patch_shape[1])

                img_crop = self.img[start_x:start_x+patch_shape[0], start_y:start_y+patch_shape[1], :]
                pred = self.pred(img_crop.transpose(2,1,0),
                                 name=f"plots/temp_{x}_{y}",
                                 figsize=(6,3),
                                 target_shape=target_crop_size,
                                 batch_size=1,
                                 return_data=True,
                                 save_fig=False,
                                 show_fig=False)
                pred = pred.cpu().numpy().transpose(0,2,3,1)

                start_x = round(x * target_crop_size[0])
                start_y = round(y * target_crop_size[1])
                self.result_container[start_x:start_x+target_crop_size[0], start_y:start_y+target_crop_size[1], :] = np.fliplr(np.rot90(pred.squeeze(), k=3))

        if verbose: self.log(f"Finished model inference, saving result...")
        self.save_image(self.result_container, "upsampled")

        if verbose: self.log(f"Saved upsampled image, now saving original image...")
        self.save_image(self.img, "original")

        if show:
            self.show_images()

        dt = time.time() - start_time
        if verbose: self.log(f"Complete! Took {dt // 60} minutes, {dt % 60} seconds")

    def save_image(self, data, prefix):
        try:
            if isinstance(data, np.ndarray):
                data = torch.Tensor(data)
            img = data.mul(255).to(torch.uint8).permute((2,1,0))
            img_name = f"{prefix}_{self.amount}_{self.name}"
            img_path = str(self.path / img_name)
            torchvision.io.write_png(img, img_path)
            if self.experiment: self.experiment.log_image(img_path, img_name)
        except Exception as e:
            self.log(f"Failed to save img ({data.shape})\n{e}\nDisplaying data instead!")
            self.show(data)

    def show_images(self, figsize=10):
        self.show(self.img, figsize=figsize)
        self.show(self.result_container, figsize=figsize)

    def show(self, data, figsize=(10,8)):
        if isinstance(figsize, int):
            figsize = (figsize, int(figsize*0.8))
        plt.figure(figsize=figsize)
        if isinstance(data, torch.Tensor):
            data = data.squeeze().cpu().to(torch.uint8).permute((2,1,0)).numpy()
        plt.imshow(data)

    def save(self, pred, inp, figsize, name):
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
            self.save(pred, inp, figsize, name)
        if show_fig:
            self.show(pred)
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