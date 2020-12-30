#include "256_3_nofma.h"

#define OP_PER_IT 3
#define FMA_AVAILABLE 1

TYPE farr_256_3_nofma[MAX_NUMBER_THREADS][SIZE] __attribute__((aligned(64)));

double get_gflops_256_3_nofma(int n_threads) {
  return (double)((long)n_threads*MAXFLOPS_ITERS*OP_PER_IT*(BYTES_IN_VECT/4)*FMA_AVAILABLE)/1000000000;        
}

void compute_256_3_nofma(TYPE *farr, TYPE mult, int index) {
  farr = farr_256_3_nofma[index];
  
  for(long i=0; i<MAXFLOPS_ITERS; i++) {
    farr[0]  = _mm256_add_ps(farr[0], farr[1]);
    farr[2]  = _mm256_add_ps(farr[2], farr[3]);
    farr[4]  = _mm256_add_ps(farr[4], farr[5]);
  }
}
