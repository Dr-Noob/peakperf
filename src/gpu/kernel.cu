#include "kernel.hpp"
#include <stdio.h>
#include <stdint.h>

__global__
void compute_kernel(float *vec_a, float *vec_b, float *vec_c, int n) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  int off = gid*4;
  __shared__ float myblockA[4];
  __shared__ float myblockB[4];
  __shared__ float myblockC[4];

  myblockA[0] = vec_a[off+0];
  myblockB[0] = vec_b[off+0];
  myblockC[0] = vec_a[off+0];
  myblockA[1] = vec_a[off+1];
  myblockB[1] = vec_b[off+1];
  myblockC[1] = vec_a[off+1];
  myblockA[2] = vec_a[off+2];
  myblockB[2] = vec_b[off+2];
  myblockC[2] = vec_a[off+2];
  myblockA[3] = vec_a[off+3];
  myblockB[3] = vec_b[off+3];
  myblockC[3] = vec_a[off+3];

  __syncthreads();

  #pragma unroll 32
  for(long i=0; i < BENCHMARK_GPU_ITERS; i++) {
    myblockC[0] = (myblockC[0] * myblockA[0]) + myblockB[0];
    myblockC[1] = (myblockC[1] * myblockA[1]) + myblockB[1];
    myblockC[2] = (myblockC[2] * myblockA[2]) + myblockB[2];
    myblockC[3] = (myblockC[3] * myblockA[3]) + myblockB[3];
  }

  vec_c[off+0] = myblockC[0];
  vec_c[off+1] = myblockC[1];
  vec_c[off+2] = myblockC[2];
  vec_c[off+3] = myblockC[3];
}

