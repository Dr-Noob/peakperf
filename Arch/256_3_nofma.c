#include "256_3_nofma.h"
#define OP_PER_IT B_256_3_NOFMA_OP_IT
#include "data.h"

void compute_256_3_nofma(TYPE *farr, TYPE mult, int index) {
  farr = farr_256_3_nofma[index];
  
  for(long i=0; i<MAXFLOPS_ITERS; i++) {
    farr[0]  = _mm256_add_ps(farr[0], farr[1]);
    farr[2]  = _mm256_add_ps(farr[2], farr[3]);
    farr[4]  = _mm256_add_ps(farr[4], farr[5]);
  }
}
