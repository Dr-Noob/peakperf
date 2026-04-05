#ifndef __COMPUTE_KERNEL_4__
#define __COMPUTE_KERNEL_4__

#include "arch.hpp"

__global__ void compute_kernel_4(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c);

#endif
