#!/bin/bash

rm -rf build/
mkdir build/
cd build/
#cmake -DENABLE_CPU_DEVICE=0 ..
#make VERBOSE=1
make
