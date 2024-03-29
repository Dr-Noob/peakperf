#include "128_6.hpp"
#define OP_PER_IT B_128_6_OP_IT

TYPE farr_128_6[MAX_NUMBER_THREADS][SIZE] __attribute__((aligned(64)));

void compute_128_6(TYPE *farr, TYPE mult, int index) {
  farr = farr_128_6[index];

  for(long i=0; i < BENCHMARK_CPU_ITERS; i++) {
    farr[0]  = _mm_add_ss(farr[0], farr[1]);
    farr[2]  = _mm_add_ss(farr[2], farr[3]);
    farr[4]  = _mm_add_ss(farr[4], farr[5]);
    farr[6]  = _mm_mul_ss(farr[6], farr[7]);
    farr[8]  = _mm_mul_ss(farr[8], farr[9]);
    farr[10] = _mm_mul_ss(farr[10], farr[11]);
  }
}
