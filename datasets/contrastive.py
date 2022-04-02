
class ImageFolderDataset(Dataset):
    def __init__(self, root_path, transform=None, target_window=100, first_k=None, 
                 last_k=None, skip_every=1, repeat=1, cache='in_memory', 
                 shuffle_mapping=False, forced_mapping=None, real_shuffle=False,
                 batch_size=args.batch_size, logger=print, config={ 
                    'train': {
                        'transform':None,
                        'repeat':10},
                    'eval': {
                        'transform': None,
                        'repeat': 1
                    }
                  }):
        self.configs = config
        self.current_mode = list(config.keys())[0]
        self.repeat = repeat #config[self.current_mode]['repeat']
        self.cache = cache
        self.batch_size = batch_size
        self.logger = logger

        filenames = sorted(os.listdir(root_path))
        self.sets = [f.split('_')[-1].split('--')[0] for f in filenames]
        self.set_lookup = {f: s for f, s in zip(filenames, self.sets)}

        self.set_list = []
        total_labels = 0

        for setname in np.unique(self.sets):
            set_seen = 0
            for i, f in enumerate([f for f in filenames if self.set_lookup[f] == setname]):
                self.set_list.append({
                    "label": total_labels,
                    "file": f,
                    "order": i
                })
                set_seen += 1
                if set_seen % target_window == 0:
                    total_labels += 1
            total_labels += 1

        self.target_lookup = {s["file"]: s["label"] for s in self.set_list}
        self.order_lookup = {s["file"]: s["order"] for s in self.set_list}
        
        if first_k is not None:
            filenames = filenames[:first_k]
        elif last_k is not None:
            filenames = filenames[-last_k:]
        filenames = filenames[::skip_every]

        self.logger(f"Found {len(filenames)} files in {root_path}")

        self.do_shuffle = real_shuffle
        if shuffle_mapping:
            self.init_mapping = np.random.permutation(len(filenames))
        else:
            self.init_mapping = [i for i in range(len(filenames))]
        if forced_mapping is not None:
            self.init_mapping = forced_mapping

        self.logger(f"Using cache strategy '{cache}'")
        self.files = []
        self.filenames = []
        with tqdm(self.init_mapping) as pbar:
            for file_idx in pbar:
                try:
                    filename = filenames[file_idx]
                    filepath = os.path.join(root_path, filename)
                    
                    self.filenames.append(filename)

                    if cache == 'none':
                        self.files.append(filepath)

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
                                pickle.dump(imageio.imread(filepath), f)
                            print('dump', bin_file)
                        self.files.append(bin_file)

                    elif cache == 'in_memory':
                        self.files.append(Image.open(filepath).convert('RGB'))
                except Exception as e:
                    print(f"Failed to load image {filepath}: {e}")

        self.mapping = [i for i in range(len(self.files))]
        self.actual_size = len(self.mapping)
        self.targets = [self.target_lookup[f] for f in self.filenames]
        self.classes = np.unique(self.targets)
        self.num_class = len(self.classes)        
        self.transform = transform

    def __len__(self):        
        return len(self.files) * self.repeat

    def __getitem__(self, i):
        i = self.mapping[i % len(self.files)]
        x = self.files[i]
        l = self.targets[i]

        if self.cache == 'none':
            return Image.open(x).convert('RGB'), l, i

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x, l, i

        elif self.cache == 'in_memory':
            return x, l, i

    def shuffle_mapping(self):
        # can't easily shuffle and use indexes as labels
        if self.do_shuffle:
            random.shuffle(self.mapping)

    def set_mode(self, name):
        if name in self.configs:
            try:
                for a, v in self.configs[name].items():                    
                    if a in self.__dict__: 
                        self.logger(f"[INFO]\tSetting {a} to {v}")
                        self.__dict__[a] = v
                    else:
                        self.logger(f"[INFO]\tSetting {a} not found")
                self.logger(f"[INFO]\tMode switched from '{self.current_mode}' to '{name}'")                
                self.current_mode = name                
            except Exception as e:
                self.logger(f"[INFO]\tMode switched from '{self.current_mode}' to 'error'")
                self.current_mode = "error"
                raise e
        else:
            raise AttributeError(f"No {name} config profile found")


class FractalLabelPair(ImageFolderDataset):
    def __getitem__(self, index):
        img, label, order = super().__getitem__(index)

        if self.transform is None:
            self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(data_stats["mean"], 
                                                     data_stats["std"])])
        img_1 = self.transform(img)
        img_2 = self.transform(img)

        return { 
            "images": (img_1, img_2), 
            "label": label, 
            "order": order
        }
class FractalPair(ImageFolderDataset):
    def __getitem__(self, index):
        img, label, order = super().__getitem__(index)

        if self.transform is None:
            self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(data_stats["mean"], 
                                                     data_stats["std"])])
        img_1 = self.transform(img)
        img_2 = self.transform(img)
        
        return { 
            "images": (img_1, img_2), 
            "order": order
        }
class FractalLabel(ImageFolderDataset):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        if self.transform is None:
            self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(data_stats["mean"], 
                                                     data_stats["std"])])
        img = self.transform(img)        

        return { 
            "images": (img), 
            "label": label, 
            "order": order
        }
class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img_1 = self.transform(img)
            img_2 = self.transform(img)

        return img_1, img_2

class ConfigurableDataLoader(DataLoader):
    def set_mode(self, name):
        try:
            self.dataset.set_mode(name)            
        except Exception as e:
            print(f"[ERROR]\tFailed to switch to mode {name}:\n{e}")