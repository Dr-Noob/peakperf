#ifndef __COMPUTE_MAXWELL__
#define __COMPUTE_MAXWELL__

#include "kernel.hpp"
#define WORK_MAXWELL 6

__global__ void compute_maxwell(float *a, float *b, float *c, int n);

#endif
