#include "mad_6.hpp"

__global__
void compute_mad_6(float *vec_a, float *vec_b, float *vec_c, int n) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x);
  float a = vec_a[idx];
  float b = vec_b[idx];
  float c = 0.0;

  for(long i=0; i < BENCHMARK_GPU_ITERS; i++) {
    c = (a * b) + c;
  }

  vec_c[idx] = c;
}
