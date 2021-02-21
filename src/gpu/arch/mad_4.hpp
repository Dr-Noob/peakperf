#ifndef __COMPUTE_MAD_4__
#define __COMPUTE_MAD_4__

#include "kernel.hpp"
#define WORK_MAD_4 4

__global__ void compute_mad_4(float *a, float *b, float *c, int n);

#endif
