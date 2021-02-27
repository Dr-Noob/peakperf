#include "kernel.hpp"

__global__
void compute_kernel(float *vec_a, float *vec_b, float *vec_c, int n) {
  float a = vec_a[0];
  float b = vec_b[0];
  float c = 0.0;

  #pragma unroll 2000
  for(long i=0; i < BENCHMARK_GPU_ITERS; i++) {
    c = (c * a) + b;
  }

  vec_c[0] = c;
}
