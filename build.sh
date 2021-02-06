#!/bin/bash

# peakperf build script
set -e

rm -rf build/ peakperf
mkdir build/
cd build/

cmake ..
make
cd -
ln -s build/peakperf peakperf
