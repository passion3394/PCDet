# PCDet: 3D Point Cloud Detection

`PCDet` is a general PyTorch-based codebase for 3D object detection from point cloud.
<br>

<p align="middle">
  <img src="docs/demo_01.png" width="430" height="380"/>
  <img src="docs/demo_02.png" width="430" height="380"/> 
</p>

## laymu's experiment environment
_libgcc_mutex             0.1                 conda_forge    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
_openmp_mutex             4.5                       0_gnu    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
ca-certificates           2020.4.5.1           hecc5488_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
certifi                   2020.4.5.1       py37hc8dfbb8_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
cycler                    0.10.0                   pypi_0    pypi<br>
decorator                 4.4.2                    pypi_0    pypi<br>
easydict                  1.9                      pypi_0    pypi<br>
imageio                   2.8.0                    pypi_0    pypi<br>
kiwisolver                1.2.0                    pypi_0    pypi<br>
ld_impl_linux-64          2.34                 h53a641e_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
libffi                    3.2.1                         1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free<br>
libgcc-ng                 9.2.0                h24d8f2e_2    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
libgomp                   9.2.0                h24d8f2e_2    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
libstdcxx-ng              9.2.0                hdf63c60_2    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
llvmlite                  0.31.0                   pypi_0    pypi<br>
matplotlib                3.2.1                    pypi_0    pypi<br>
ncurses                   6.1               hf484d3e_1002    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
networkx                  2.4                      pypi_0    pypi<br>
numba                     0.48.0                   pypi_0    pypi<br>
numpy                     1.18.2                   pypi_0    pypi<br>
openssl                   1.1.1f               h516909a_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
pcdet                     0.1.0+f4acdff             dev_0    <develop><br>
pillow                    7.1.1                    pypi_0    pypi<br>
pip                       20.0.2                     py_2    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
protobuf                  3.11.3                   pypi_0    pypi<br>
pyparsing                 2.4.7                    pypi_0    pypi<br>
python                    3.7.6           h8356626_5_cpython    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
python-dateutil           2.8.1                    pypi_0    pypi<br>
python_abi                3.7                     1_cp37m    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
pywavelets                1.1.1                    pypi_0    pypi<br>
pyyaml                    5.3.1                    pypi_0    pypi<br>
readline                  8.0                  hf8c457e_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
scikit-image              0.16.2                   pypi_0    pypi<br>
scipy                     1.4.1                    pypi_0    pypi<br>
setuptools                46.1.3           py37hc8dfbb8_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
six                       1.14.0                   pypi_0    pypi<br>
spconv                    1.0                      pypi_0    pypi<br>
sqlite                    3.30.1               hcee41ef_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
tensorboardx              2.0                      pypi_0    pypi<br>
tk                        8.6.10               hed695b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
torch                     1.4.0                    pypi_0    pypi<br>
tqdm                      4.45.0                   pypi_0    pypi<br>
wheel                     0.34.2                     py_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
xz                        5.2.5                h516909a_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge<br>
zlib                      1.2.11                        0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free<br>
  
tips:
- [x] must use spconv1.0
- [x] when compiling the tag of spconv, should replace torch::jit::RegisterOperators in src/spconv/all.cc to  torch::RegisterOperators
- [x] after install spconv 1.0, reinstall pcdet

## Introduction
`PCDet` is a general PyTorch-based codebase for 3D object detection from point cloud. 
It currently supports several state-of-the-art 3D object detection methods (`PointPillar`, `SECOND`, `Part-A^2 Net`) with highly refactored codes for both one-stage and two-stage frameworks.

This is also the official code release of [`Part-A^2 net`](https://arxiv.org/abs/1907.03670). 

Note that currently this framework mainly contains the voxel-based approaches and we are going to support more point-based approaches in the future. 

### Currently Supported Features
- [x] Support both one-stage and two-stage 3D object detection frameworks
- [x] Distributed training with multiple GPUs and multiple machines, cost about 5 hours to achieve SoTA results on KITTI
- [x] Clear code structure for supporting more datasets and approaches
- [x] RoI-aware point cloud pooling
- [x] GPU version 3D IoU calculation and rotated NMS

## Model Zoo

### KITTI 3D Object Detection Baselines
Supported methods are shown in the below table. The results are the 3D detection performance of car class on the *val* set of KITTI dataset.
All models are trained with 8 GPUs and are available for download.

|                                             | training time | AP_Easy | AP_Mod. | AP_Hard | download  |
|---------------------------------------------|:-------------:|:-------:|:-------:|:-------:|:---------:|
| [PointPillar](tools/cfgs/pointpillar.yaml)  | ~2hours       | 87.37   | 77.30   | 74.02   | [model-18M](https://drive.google.com/open?id=1EIXknJF3ME8LLvC2chB7L52U6XGU_bxg) | 
| [SECOND](tools/cfgs/second.yaml)            | ~2hours       | 88.46   | 78.46   | 76.63   | [model-20M](https://drive.google.com/open?id=1Nx_STdaItqrnW8EqPHSDIDZXCF2PYYE-) |
| [Part-A^2](tools/cfgs/PartA2_car.yaml)      | ~5hours       | 89.66   | 79.45   | 78.80   | [model-209M](https://drive.google.com/open?id=1D-lxyPww80H-zEdheaDTO6BfxCFiXOEo) |
| [Part-A^2-fc](tools/cfgs/PartA2_fc.yaml)    | ~5hours       | 89.57   | 79.31   | 78.61   | [model-244M](https://drive.google.com/open?id=1HypkHBfcKWEdP5RLh6CmT77L1gXDck6D) |


## Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 14.04/16.04)
* Python 3.6+
* PyTorch 1.1 or higher
* CUDA 9.0 or higher


### Install `pcdet`
1. Clone this repository.
```shell
git clone https://github.com/sshaoshuai/PCDet.git
```

2. Install the dependent libraries as follows:

* Install the dependent python libraries: 
```
pip install -r requirements.txt 
```

* Install the SparseConv library, we use the non-official implementation from [`spconv`](https://github.com/traveller59/spconv). Note that we use an old version of `spconv`, make sure you install the `spconv v1.0` ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)) instead of the latest one.

3. Install this `pcdet` library by running the following command:
```shell
python setup.py develop
```

## Dataset Preparation

Currently we only support KITTI dataset, and contributions are welcomed to support more datasets.

### KITTI Dataset
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [here](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training): 

```
PCDet
├── data
│   ├── kitti
│   │   │──ImageSets
│   │   │──training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │──testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

* Generate the data infos by running the following command in the path `pcdet/datasets/kitti`: 
```python 
python kitti_dataset.py create_kitti_infos
```

## Getting Started
All the config files are within `tools/cfgs/`. 

### Test and evaluate the pretrained models
* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size 4 --ckpt ${CKPT}
```

* For example, for testing with the above provided `Part-A^2` model, please run the following command (here we add `--set` to modify some default parameters to match with the training setting of the provided `Part-A^2` model, and other provided models do not need to add it): 

```shell script 
python test.py --cfg_file cfgs/PartA2_car.yaml --batch_size 4 --ckpt PartA2.pth \ 
    --set MODEL.RPN.BACKBONE.NAME UNetV0 MODEL.RPN.RPN_HEAD.ARGS use_binary_dir_classifier:True
```

* To evaluate all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size 4 --eval_all
```


### Train a model
Currently, to train `PointPillar` or `SECOND` or `PartA2`, the `--batch_size` depends on the number of your training GPUs as we use `${BATCH_SIZE}=4*${NUM_GPUS}`, i.e., `--batch_size 32` for training with 8 GPUs. 

* Train with multiple GPUs:
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} \ 
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```

* Train with multiple machines:
```shell script
sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} \ 
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```

* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```

## Acknowledgement
We would like to thank for [second.pytorch](https://github.com/traveller59/second.pytorch) for providing the original implementation of the one-stage voxel-based framework, 
and there are also some parts of the codes that are modified from [PointRCNN](https://github.com/sshaoshuai/PointRCNN). 

We hope that this repo could serve as a strong and flexible codebase to benefit the research community by speeding up the process of reimplementing previous works and/or developing new methods.


## Citation 
If you find this work useful in your research, please consider cite:
```
@article{shi2020points,
  title={From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network},
  author={Shi, Shaoshuai and Wang, Zhe and Shi, Jianping and Wang, Xiaogang and Li, Hongsheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  publisher={IEEE}
}
```

and / or

```
@inproceedings{shi2019pointrcnn,
  title={PointRCNN: 3d Object Progposal Generation and Detection from Point Cloud},
  author={Shi, Shaoshuai and Wang, Xiaogang and Li, Hongsheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={770--779},
  year={2019}
}
```

## Contact
Should you have any question, please contact Shaoshuai Shi ([@sshaoshuai](http://github.com/sshaoshuai)).

