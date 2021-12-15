import os
import json
import random
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.notebook import tqdm

from datasets import register


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 last_k=None, skip_every=1, repeat=1, cache='none', 
                 shuffle_mapping=False, forced_mapping=None,
                 mean_color_threshold=90):
        self.repeat = repeat
        self.cache = cache
        self.mean_color_threshold = mean_color_threshold

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]
        elif last_k is not None:
            filenames = filenames[-last_k:]
        filenames = filenames[::skip_every]

        if shuffle_mapping:
            self.init_mapping = np.random.permutation(len(filenames))
        else:
            self.init_mapping = [i for i in range(len(filenames))]
        if forced_mapping is not None:
            self.init_mapping = forced_mapping

        self.files = []
        self.filenames = []
        for file_idx in tqdm(self.init_mapping):
            try:
                filename = filenames[file_idx]
                file = os.path.join(root_path, filename)
                
                use_file = True

                if cache == 'none':
                    img = transforms.ToTensor()(
                        Image.open(file).convert('RGB'))
                    if img.mean() > self.mean_color_threshold:
                        self.files.append(file)
                    else:
                        use_file = False

                elif cache == 'bin':
                    bin_root = os.path.join(os.path.dirname(root_path),
                        '_bin_' + os.path.basename(root_path))
                    if not os.path.exists(bin_root):
                        os.mkdir(bin_root)
                        print('mkdir', bin_root)
                    bin_file = os.path.join(
                        bin_root, filename.split('.')[0] + '.pkl')
                    if not os.path.exists(bin_file):
                        with open(bin_file, 'wb') as f:
                            pickle.dump(imageio.imread(file), f)
                        print('dump', bin_file)
                    self.files.append(bin_file)

                elif cache == 'in_memory':
                    img = transforms.ToTensor()(
                        Image.open(file).convert('RGB'))
                    if img.mean() > self.mean_color_threshold:
                        self.files.append(img)
                    else:
                        use_file = False
                
                if use_file: self.filenames.append(file)
            except Exception as e:
                print(f"Failed to load image {filename}: {e}")

        self.mapping = [i for i in range(len(self.files))]

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        idx = self.mapping[idx % len(self.files)]
        x = self.files[idx]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x

    def shuffle_mapping(self):
        random.shuffle(self.mapping)


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, shuffle_mapping=True, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, forced_mapping=self.dataset_1.init_mapping, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
