#!/bin/bash

set -euo pipefail

DIR=$(pwd)
mkdir -p $HOME/.local/nvidia/hpc_sdk

echo "Configuring python"
conda create -n project8 python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate project8

echo "Installing NVHPC SDK"
cd /scratch/$USER
wget --no-verbose https://developer.download.nvidia.com/hpc-sdk/24.7/nvhpc_2024_247_Linux_x86_64_cuda_12.5.tar.gz
tar xvpzf nvhpc_2024_247_Linux_x86_64_cuda_12.5.tar.gz
cd nvhpc_2024_247_Linux_x86_64_cuda_12.5
NVHPC_SILENT=true NVHPC_INSTALL_DIR=$HOME/.local/nvidia/hpc_sdk NVHPC_INSTALL_TYPE=single ./install

echo "Installing AmgX"
cd /scratch/$USER
git clone --recursive https://github.com/nvidia/amgx
cd amgx
mkdir build
cd build
cmake ../ -DCMAKE_NO_MPI=True -DCMAKE_INSTALL_PREFIX=$HOME/.local
make -j$(nproc) all
make install

echo "Installing pyamgx"
cd /scratch/$USER
git clone https://github.com/shwina/pyamgx
cd pyamgx
git checkout 6229ff008ee5a264cfc1799eeb2f83d96da0aadc
patch -p1 < $DIR/pyamgx.patch
pip install --upgrade\
 cython\
 pip\
 scipy\
 setuptools\
 wheel
pip install .

echo "Installing python dependencies"
cd $DIR/..
pip install .

echo "Done"
