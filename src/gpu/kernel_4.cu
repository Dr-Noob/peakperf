#include "kernel_4.hpp"

__global__
void compute_kernel_4(float *vec_a, float *vec_b, float *vec_c) {
  int cid = threadIdx.x + blockIdx.x * blockDim.x;

  float a = vec_a[cid];
  float b = vec_b[cid];

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
