#!/bin/bash

# peakperf build script
set -e

rm -rf build/ peakperf
mkdir build/
cd build/

if [ "$1" == "debug" ]
then
  PKPF_BUILD_TYPE="Debug"
else
  PKPF_BUILD_TYPE="Release"
fi

# In case you have CUDA installed but it is not detected,
# - set CMAKE_CUDA_COMPILER to your nvcc binary:
# - set CMAKE_CUDA_COMPILER_TOOLKIT_ROOT to the CUDA root dir
# cmake -DCMAKE_BUILD_TYPE=$PKPF_BUILD_TYPE -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT=/usr/local/cuda/ ..
cmake -DCMAKE_BUILD_TYPE=$PKPF_BUILD_TYPE ..
make -j$(nproc)
cd -
ln -s build/peakperf peakperf
