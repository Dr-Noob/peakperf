#include "kernel.hpp"

__global__
void compute_kernel(float *vec_a, float *vec_b, float *vec_c, int n) {
  int cid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  __shared__ float myblockA[256];
  __shared__ float myblockB[256];
  float c = vec_c[cid];

  myblockA[tid] = vec_a[tid];
  myblockB[tid] = vec_b[tid];

  __syncthreads();

  #pragma unroll 2000
  for(long i=0; i < BENCHMARK_GPU_ITERS; i++) {
    c = (c * myblockA[tid]) + myblockB[tid];
  }

  vec_c[cid] = c;
}
