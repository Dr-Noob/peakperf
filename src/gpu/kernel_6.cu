#include "kernel_6.hpp"

__global__
void compute_kernel_6(const float * __restrict__ vec_a, const float * __restrict__ vec_b, float * __restrict__ vec_c) {
  int cid = threadIdx.x + blockIdx.x * blockDim.x;

  float a = vec_a[cid];
  float b = vec_b[cid];

  float c0 = vec_c[cid];
  float c1 = c0;
  float c2 = c0;
  float c3 = c0;
  float c4 = c0;
  float c5 = c0;

  #pragma unroll 32
  for (long i = 0; i < BENCHMARK_GPU_ITERS; i++) {
    c0 = (c0 * a) + b;
    c1 = (c1 * a) + b;
    c2 = (c2 * a) + b;
    c3 = (c3 * a) + b;
    c4 = (c4 * a) + b;
    c5 = (c5 * a) + b;
  }

  vec_c[cid] = c0 + c1 + c2 + c3 + c4 + c5;
}
