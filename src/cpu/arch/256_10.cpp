#include "256_10.hpp"
#define OP_PER_IT B_256_10_OP_IT

TYPE farr_256_10[MAX_NUMBER_THREADS][SIZE] __attribute__((aligned(64)));

void compute_256_10(TYPE *farr, TYPE mult, int index) {
  farr = farr_256_10[index];

  for(long i=0; i < BENCHMARK_CPU_ITERS; i++) {
    farr[0]  = _mm256_fmadd_ps(mult, farr[0], farr[1]);
    farr[2]  = _mm256_fmadd_ps(mult, farr[2], farr[3]);
    farr[4]  = _mm256_fmadd_ps(mult, farr[4], farr[5]);
    farr[6]  = _mm256_fmadd_ps(mult, farr[6], farr[7]);
    farr[8]  = _mm256_fmadd_ps(mult, farr[8], farr[9]);
    farr[10] = _mm256_fmadd_ps(mult, farr[10], farr[11]);
    farr[12] = _mm256_fmadd_ps(mult, farr[12], farr[13]);
    farr[14] = _mm256_fmadd_ps(mult, farr[14], farr[15]);
    farr[16] = _mm256_fmadd_ps(mult, farr[16], farr[17]);
    farr[18] = _mm256_fmadd_ps(mult, farr[18], farr[19]);
  }
}
