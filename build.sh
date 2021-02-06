#!/bin/bash

# peakperf build script
set -e

rm -rf build/ peakperf
mkdir build/
cd build/

cmake -DENABLE_GPU_DEVICE=0 ..
make
cd -
ln -s build/peakperf peakperf
