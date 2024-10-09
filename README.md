# distill

该仓库为图神经网络领域第一个知识蒸馏算法库，该算法为基于 `TensorLayerX` 和 `GammaGL` 设计的，并且支持在torch、tensorflow、paddle、mindspore和jittor后端下运行。

## How to Run

目前，我们支持python>=3.9, 并仅支持在linux系统上运行该算法库。

1. python环境

```bash
$ conda create -n distill python=3.10
$ source activate distill
```

2. 安装深度学习后端

```bash
# For tensorflow == 2.10.0
$ pip install tensorflow-gpu # GPU version
$ pip install tensorflow # CPU version

# For torch, version >= 2.1
# 可在该网站安装对应版本的torch：https://pytorch.org/get-started/previous-versions/
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For paddle, any latest stable version
# https://www.paddlepaddle.org.cn/
$ python -m pip install paddlepaddle-gpu

# For mindspore, GammaGL supports version 2.2.0, GPU-CUDA 11.6
# https://www.mindspore.cn/install
$ pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/unified/x86_64/mindspore-2.2.0-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3. 安装TensorLayerX

```
pip install git+https://github.com/dddg617/tensorlayerx.git@nightly 
```

**Note**

> - 当安装TensorLayerx时，需要首先安装PyTorch

4. 安装GammaGL

可通过pypi进行pip安装，也可以通过编译安装的方式进行安装，由于每位用户的操作系统环境各异，推荐用户使用编译安装的方式进行安装，下面将介绍编译安装的命令。

```
$ git clone --recursive https://github.com/BUPT-GAMMA/GammaGL.git
$ pip install pybind11 pyparsing
$ python setup.py install build_ext --inplace
```

