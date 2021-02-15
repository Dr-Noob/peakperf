#include "turing.hpp"

__global__
void compute_turing(float *vec_a, float *vec_b, float *vec_c, int n) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * WORK_TURING;

  float a1 = vec_a[idx+0];
  float a2 = vec_a[idx+1];
  float a3 = vec_a[idx+2];
  float a4 = vec_a[idx+3];

  float b1 = vec_b[idx+0];
  float b2 = vec_b[idx+1];
  float b3 = vec_b[idx+2];
  float b4 = vec_b[idx+3];

  float c1 = 0.0;
  float c2 = 0.0;
  float c3 = 0.0;
  float c4 = 0.0;

  for(long i=0; i < BENCHMARK_GPU_ITERS; i++) {
    c1 = (a1 * b1) + c1;
    c2 = (a2 * b2) + c2;
    c3 = (a3 * b3) + c3;
    c4 = (a4 * b4) + c4;
  }

  vec_c[idx+0] = c1;
  vec_c[idx+1] = c2;
  vec_c[idx+2] = c3;
  vec_c[idx+3] = c4;
}
