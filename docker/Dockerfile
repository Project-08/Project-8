FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

RUN apt-get update\
 && apt-get upgrade -qq\
 && apt-get install -qq --no-install-recommends\
 cmake\
 git\
 python3-dev\
 python3-pip\
 python3-venv\
 python3\
 sudo\
 && apt-get clean\
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN git clone --recursive https://github.com/nvidia/amgx\
 && cd amgx\
 && mkdir build\
 && cd build\
 && cmake ../ -DCMAKE_NO_MPI=True -DCMAKE_INSTALL_PREFIX=/usr/local\
 && make -j$(nproc) all\
 && make install\
 && cd ../..\
 && rm -rf amgx

COPY pyamgx.patch /root/pyamgx.patch

RUN git clone https://github.com/shwina/pyamgx\
 && cd pyamgx\
 && git checkout 6229ff008ee5a264cfc1799eeb2f83d96da0aadc\
 && patch -p1 < /root/pyamgx.patch\
 && pip install --upgrade\
 cython\
 pip\
 scipy\
 setuptools\
 wheel\
 && pip install .\
 && cd ..\
 && rm -rf pyamgx\
 && rm /root/pyamgx.patch
