#ifndef __COMPUTE_KERNEL__
#define __COMPUTE_KERNEL__

#include "arch.hpp"

__device__ unsigned long long totThr;
__global__ void compute_kernel(float *a, float *b, float *c, int n);

#endif
