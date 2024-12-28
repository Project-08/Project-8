#!/bin/bash

set -euo pipefail

mkdir -p $HOME/.local
python3 -m venv $HOME/venv
. $HOME/venv/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib
mkdir build
cd build

git clone --recursive https://github.com/nvidia/amgx
cd amgx
mkdir build
cd build
cmake ../ -DCMAKE_NO_MPI=True -DCMAKE_INSTALL_PREFIX=$HOME/.local
make -j`nproc` all
make install
cd ../..

git clone https://github.com/shwina/pyamgx
cd pyamgx
git checkout 6229ff008ee5a264cfc1799eeb2f83d96da0aadc
patch -p1 < ../../pyamgx.patch
pip install --upgrade\
 cython\
 pip\
 scipy\
 setuptools\
 wheel
pip install .
cd ..
