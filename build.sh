#!/bin/bash

# peakperf build script
set -e

rm -rf build/ peakperf
mkdir build/
cd build/

# In case you have CUDA installed but it is not detected,
# set CMAKE_CUDA_COMPILER to your nvcc binary:
# cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
cmake ..
make
cd -
ln -s build/peakperf peakperf
