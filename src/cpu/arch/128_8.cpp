#include "128_8.hpp"
#define OP_PER_IT B_128_8_OP_IT

TYPE farr_128_8[MAX_NUMBER_THREADS][SIZE] __attribute__((aligned(64)));

void compute_128_8(TYPE *farr, TYPE mult, int index) {
  farr = farr_128_8[index];

  for(long i=0; i < BENCHMARK_CPU_ITERS; i++) {
    farr[0]  = _mm_add_ss(farr[0], farr[1]);
    farr[2]  = _mm_add_ss(farr[2], farr[3]);
    farr[4]  = _mm_add_ss(farr[4], farr[5]);
    farr[6]  = _mm_add_ss(farr[6], farr[7]);
    farr[8]  = _mm_add_ss(farr[8], farr[9]);
    farr[10] = _mm_add_ss(farr[10], farr[11]);
    farr[12] = _mm_add_ss(farr[12], farr[13]);
    farr[14] = _mm_add_ss(farr[14], farr[15]);
  }
}
