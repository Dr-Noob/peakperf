#ifndef __COMPUTE_MAD_6__
#define __COMPUTE_MAD_6__

#include "kernel.hpp"
#define WORK_MAD_6 6

__global__ void compute_mad_6(float *a, float *b, float *c, int n);

#endif
