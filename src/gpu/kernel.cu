#include "kernel.hpp"

__global__
void compute_kernel(float *vec_a, float *vec_b, float *vec_c, int n) {
  int cid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  __shared__ float myblockA[256];
  __shared__ float myblockB[256];

  myblockA[tid] = vec_a[cid];
  myblockB[tid] = vec_b[cid];

  __syncthreads();

  float a = myblockA[tid];
  float b = myblockB[tid];

  float c0 = vec_c[cid];
  float c1 = c0;
  float c2 = c0;
  float c3 = c0;

  for (long i = 0; i < BENCHMARK_GPU_ITERS; i++) {
    c0 = (c0 * a) + b;
    c1 = (c1 * a) + b;
    c2 = (c2 * a) + b;
    c3 = (c3 * a) + b;
  }

  vec_c[cid] = c0 + c1 + c2 + c3;
}

