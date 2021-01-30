#ifndef __COMPUTE_MAXWELL__
#define __COMPUTE_MAXWELL__

#include "kernel.h"
#define WORK_MAXWELL 6

__global__ void matrixMul_maxwell(float *a, float *b, float *c, int n);

#endif
