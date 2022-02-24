# ImageFunctions
PyTorch based experiments on representing images as neural networks

## TODO
**This readme is a work in progress**

### Data

`mkdir load` for putting the dataset folders.

- **DIV2K**: `mkdir` and `cd` into `load/div2k`. Download HR images and bicubic validation LR images from [DIV2K website](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (i.e. [Train_HR](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip), [Valid_HR](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip), [Valid_LR_X2](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip), [Valid_LR_X3](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip), [Valid_LR_X4](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip)). `unzip` these files to get the image folders.

- **benchmark datasets**: `mkdir` and `cd` into `load/benchmark`. Download and `tar -xf` the [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) (provided by [this repo](https://github.com/thstkdgus35/EDSR-PyTorch)), get the image folders `Set5/, Set14/, B100/, Urban100/`.

- **celebAHQ**: `mkdir load/celebAHQ` and `cp scripts/resize.py load/celebAHQ/`, then `cd load/celebAHQ/`. Download and `unzip` data1024x1024.zip from the [Google Drive link](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P?usp=sharing) (provided by [this repo](github.com/suvojit-0x55aa/celebA-HQ-dataset-download)). Run `python resize.py` and get image folders `256/, 128/, 64/, 32/`. Download the [split.json](https://www.dropbox.com/s/2qeijojdjzvp3b9/split.json?dl=0).

### Running the code

**0. Preliminaries**

- For `train_liif.py` or `test.py`, use `--gpu [GPU]` to specify the GPU IDs for running (e.g. `--gpu 0` or `--gpu 0,1`).

- For `train_liif.py`, by default the saving folder is at `save/_[CONFIG_NAME]`. We can use `--name` to specify a name if needed.

- For dataset args in configs, `cache: in_memory` denotes pre-loading into memory (may require large memory, e.g. ~40GB for DIV2K), `cache: bin` denotes creating binary files (in the same folder) for the first time, `cache: none` denotes direct loading. We can modify it according to the hardware resources before running the training scripts.

**1. DIV2K experiments**

Train: `python train_liif.py --config configs/train-div2k/train_edsr-baseline-liif.yaml` (with EDSR-baseline backbone, for RDN replace `edsr-baseline` with `rdn`). We use 1 GPU for training EDSR-baseline-LIIF and 4 GPUs for RDN-LIIF.

Test: `bash scripts/test-div2k.sh [MODEL_PATH] [GPU]` for div2k validation set, `bash scripts/test-benchmark.sh [MODEL_PATH] [GPU]` for benchmark datasets. `[MODEL_PATH]` is the path to a `.pth` file, we use `epoch-last.pth` in corresponding saving folder.

Name|Pretrained model
:-:|:-:
EDSR-baseline-LIIF|[Download (19M)](https://www.dropbox.com/s/6f402wcn4v83w2v/edsr-baseline-liif.pth?dl=0)
RDN-LIIF|[Download (256M)](https://www.dropbox.com/s/mzha6ll9kb9bwy0/rdn-liif.pth?dl=0)

**2. celebAHQ experiments**

Train: `python train_liif.py --config configs/train-celebAHQ/[CONFIG_NAME].yaml`.

Test: `python test.py --config configs/test/test-celebAHQ-32-256.yaml --model [MODEL_PATH]` (or `test-celebAHQ-64-128.yaml`). We use `epoch-best.pth` in corresponding saving folder.
