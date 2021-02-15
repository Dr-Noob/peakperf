#include "ivy_bridge.hpp"
#define OP_PER_IT B_256_6_NOFMA_OP_IT

TYPE farr_ivy_bridge[MAX_NUMBER_THREADS][SIZE] __attribute__((aligned(64)));  

void compute_ivy_bridge(TYPE *farr, TYPE mult, int index) {
  farr = farr_ivy_bridge[index];
  
  for(long i=0; i < BENCHMARK_CPU_ITERS; i++) {
    farr[0]  = _mm256_add_ps(farr[0], farr[1]);
    farr[2]  = _mm256_add_ps(farr[2], farr[3]);
    farr[4]  = _mm256_add_ps(farr[4], farr[5]);
    farr[6]  = _mm256_mul_ps(farr[6], farr[7]);
    farr[8]  = _mm256_mul_ps(farr[8], farr[9]);
    farr[10]  = _mm256_mul_ps(farr[10], farr[11]);
  }
}
