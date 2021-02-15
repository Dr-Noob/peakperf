#ifndef __COMPUTE_TURING__
#define __COMPUTE_TURING__

#include "kernel.hpp"
#define WORK_TURING 4

__global__ void compute_turing(float *a, float *b, float *c, int n);

#endif
