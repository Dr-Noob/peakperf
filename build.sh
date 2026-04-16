#!/bin/bash

# peakperf build script
set -e

rm -rf build/ peakperf

if [ "$1" == "debug" ]
then
  PKPF_BUILD_TYPE="Debug"
else
  PKPF_BUILD_TYPE="Release"
fi

INSTALL_PREFIX="/usr/local"
if [ -n "$TERMUX__PREFIX" ]; then
  echo TERMUX__PREFIX environmental variable found: $TERMUX__PREFIX 
  INSTALL_PREFIX="$TERMUX__PREFIX"
fi

# In case you have CUDA installed but it is not detected,
# - set CMAKE_CUDA_COMPILER to your nvcc binary:
# - set CMAKE_CUDA_COMPILER_TOOLKIT_ROOT to the CUDA root dir
# cmake -DCMAKE_BUILD_TYPE=$PKPF_BUILD_TYPE -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT=/usr/local/cuda/ ..
cmake -S . -B build -DCMAKE_BUILD_TYPE=$PKPF_BUILD_TYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX 
cmake --build build -j$(nproc)
cmake --install build
ln -s build/peakperf peakperf
