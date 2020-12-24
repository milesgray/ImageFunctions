import math, pathlib

import numpy as np
import torch

import utils

__all__ = ['SuperResManager']


class _MeshGenerator:
    class MGException(Exception):
        def __init__(self, message: str):
            self.__message = message

        def __str__(self):
            return self.__message

    def __init__(self, region: list, cell_size: list, overlap: list = None):
        region = np.array(region)
        if np.linalg.norm(region[1] - region[0]) == 0:
            raise self.MGException("Region size is zero!")

        if region[0][0] >= region[1][0] or region[0][1] >= region[1][1]:
            raise self.MGException("Bad region coordinates")

        if len(cell_size) < 2 or cell_size[0] <= 0 or cell_size[1] <= 0:
            raise self.MGException("Bad cell size")

        self.__region = region
        self.__cell_size = cell_size
        if overlap is not None:
            self.__overlap = 0.5 * np.array([overlap[0], overlap[1]])
        else:
            cells_cnt = np.array(np.abs(self.__region[1] - self.__region[0]) / self.__cell_size)
            if np.array_equal(cells_cnt, [0, 0]):
                self.__overlap = np.array([0, 0])
            else:
                self.__overlap = 2 * (np.ceil(cells_cnt) * cell_size - np.abs(self.__region[1] - self.__region[0])) / np.round(
                    cells_cnt)

    def generate_cells(self):
        result = []

        def walk_cells(callback: callable):
            y_start = self.__region[0][1]
            x_start = self.__region[0][0]

            y = y_start

            step_cnt = np.array(np.ceil(np.abs(self.__region[1] - self.__region[0]) / self.__cell_size), dtype=np.uint64)

            for i in range(step_cnt[1]):
                x = x_start

                for j in range(step_cnt[0]):
                    callback([np.array([x, y]),
                              np.array([x + self.__cell_size[0], y + self.__cell_size[1]])], i, j)
                    x += self.__cell_size[0]

                y += self.__cell_size[1]

        def on_cell(coords, i, j):
            offset = self.__overlap * np.array([j, i], dtype=np.float32)
            coords[0] = coords[0] - offset
            coords[1] = coords[1] - offset
            result.append(np.array(coords, dtype=np.uint32))

        walk_cells(on_cell)
        return result


class TilesManager:
    def __init__(self):
        self._tiles = []
        self._size = None

    def generate_tiles(self, size: [], tiles_size: [], overlap: list = None) -> 'TilesManager':
        self._size = size
        self._tiles = _MeshGenerator(size, tiles_size, overlap).generate_cells()
        return self

    def get_tiles(self) -> []:
        return self._tiles

    def cut_image_by_tiles(self, image: np.ndarray) -> []:
        res = []
        for tile in self._tiles:
            res.append(TilesManager._cut_image_by_tile(image, tile))
        return res

    def merge_images_by_tiles(self, images: [np.ndarray]) -> np.ndarray:
        res = np.empty(list(self._size[1]) + [3], dtype=images[0].dtype)
        for i, tile in enumerate(self._tiles):
            if i > len(images):
                break
            TilesManager._insert_tile_to_image(res, images[i], tile)
        return res

    @staticmethod
    def _cut_image_by_tile(image: np.ndarray, tile: []) -> np.ndarray:
        return image[tile[0][1]: tile[1][1], tile[0][0]: tile[1][0], :]

    @staticmethod
    def _insert_tile_to_image(image: np.ndarray, image_part: np.ndarray, tile: []) -> None:
        image[tile[0][1]: tile[1][1], tile[0][0]: tile[1][0], :] = image_part

class SuperResManager:
    def __init__(self, model, scale=4.0, log=print):
        if isinstance(model, (str, pathlib.Path)):
            self.model = self.load_model(model, log=log)
        else:        
            self.model = model
        self.log = log
        self.scale = scale
        self.orig_tile_mgr = TilesManager()
        self.zoom_tile_mgr = TilesManager()
        self.img = None
        self.zoom_img = None

    @staticmethod
    def load_model(path, log=print):
        sv_file = torch.load(path)
        model = models.make(sv_file['model'], load_sd=True).cuda()
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
        return model

    @staticmethod
    def _make_coord_cell(target_shape=(32, 32), batch_size=8):
        coord = SuperResManager._make_coord(target_shape).repeat(batch_size, 1, 1)
        cell = torch.ones_like(coord)
        cell[..., 0] *= 2 / target_shape[1]
        cell[..., 1] *= 2 / target_shape[0]
        return coord, cell

    @staticmethod
    def _make_coord(shape, ranges=None, flatten=True):
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
    
    @staticmethod
    def _reshape(pred, t_shape):
        ih, iw = t_shape
        s = math.sqrt(pred.shape[1] / (ih * iw))
        shape = [pred.shape[0], round(ih * s), round(iw * s), 3]
        pred = pred.view(*shape) \
            .permute(0, 3, 1, 2).contiguous()
        return pred

    def load_image(self, img, tile_size=32, log=print):
        if img.shape[-1] != 3:
            img = img.squeeze().transpose(1,2,0)
        if self.log is None:
            self.log = log
        self.log(f"Image shape: {img.shape}, tile size: {tile_size}")
        self.target_img_shape = [round(img.shape[0] * self.scale), round(img.shape[1] * self.scale)]
        self.log(f"Target img shape: {self.target_img_shape}")
        self.target_tile_shape = [round(tile_size * self.scale), round(tile_size * self.scale)]
        self.log(f"Target tile shape: {self.target_tile_shape}")
        self.orig_tile_mgr = self.orig_tile_mgr.generate_tiles([[0,0],img.shape[:2]], [tile_size, tile_size])        
        self.zoom_tile_mgr = self.zoom_tile_mgr.generate_tiles([[0,0],self.target_img_shape], 
                                                               self.target_tile_shape)
        self.img = img
        self.log('Image loaded')

    def apply(self, img=None, tile_size=32, log=print):
        if self.img is None and img is None:
            self.log(f"Must run load_image before apply or pass an img to apply")
            return
        elif self.img is None:
            self.load_image(img, tile_size=tile_size, log=log)
        
        results = []
        for i, tile in enumerate(self.orig_tile_mgr.cut_image_by_tiles(self.img)):
            coord, cell = self._make_coord_cell(target_shape=self.target_tile_shape, batch_size=1)
            tile = torch.Tensor(tile.transpose(2,0,1))
            if len(tile.shape) == 3:
                tile = tile.unsqueeze(0)
            if any([d == 0 for d in tile.shape]):
                self.log(f"Skipping tile {i} - {tile.shape} (0 dimension detected)")
                continue
            self.log(f"Processing tile {i} - {tile.shape}")
            with torch.no_grad():
                result = self.model(tile.cuda(), coord.cuda(), cell.cuda())
            result = self._reshape(result, self.target_tile_shape)
            results.append(result.squeeze().cpu().numpy().transpose(1,2,0))
        self.log(f"Finished generating zoomed tiles - {len(results)} total. Now merging into final output.")
        self.zoom_img = self.zoom_tile_mgr.merge_images_by_tiles(results)
        self.log(f"Successfully created new image with shape {self.zoom_img.shape}!")
        return self.zoom_img
