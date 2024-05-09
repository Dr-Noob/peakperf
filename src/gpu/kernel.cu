#include "kernel.hpp"
#include <stdio.h>
#include <stdint.h>

__global__
void compute_kernel(float *vec_a, float *vec_b, float *vec_c, int n) {
  int cid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  __shared__ float myblockA[4];
  __shared__ float myblockB[4];
  float c1 = vec_c[0];
  float c2 = vec_c[1];
  float c3 = vec_c[2];
  float c4 = vec_c[3];

  myblockA[0] = vec_a[0];
  myblockB[0] = vec_b[0];
  myblockA[1] = vec_a[1];
  myblockB[1] = vec_b[1];
  myblockA[2] = vec_a[2];
  myblockB[2] = vec_b[2];
  myblockA[3] = vec_a[3];
  myblockB[3] = vec_b[3];

  __syncthreads();

  #pragma unroll 2000
  for(long i=0; i < BENCHMARK_GPU_ITERS; i++) {
    c1 = (c1 * myblockA[0]) + myblockB[0];
    c2 = (c2 * myblockA[1]) + myblockB[1];
    c3 = (c3 * myblockA[2]) + myblockB[2];
    c4 = (c4 * myblockA[3]) + myblockB[3];
  }

  vec_c[0] = c1;
  vec_c[1] = c2;
  vec_c[2] = c3;
  vec_c[3] = c4;
}

