#include "256_5.h"

#define OP_PER_IT B_256_5_OP_IT

TYPE farr_256_5[MAX_NUMBER_THREADS][SIZE] __attribute__((aligned(64)));

void compute_256_5(TYPE *farr, TYPE mult, int index) {
  farr = farr_256_5[index];
    
  for(long i=0; i<MAXFLOPS_ITERS; i++) {
    farr[0]  = _mm256_fmadd_ps(mult, farr[0], farr[1]);
    farr[2]  = _mm256_fmadd_ps(mult, farr[2], farr[3]);
    farr[4]  = _mm256_fmadd_ps(mult, farr[4], farr[5]);
    farr[6]  = _mm256_fmadd_ps(mult, farr[6], farr[7]);
    farr[8]  = _mm256_fmadd_ps(mult, farr[8], farr[9]);
  }
}
