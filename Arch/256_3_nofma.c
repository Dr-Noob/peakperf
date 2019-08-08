#include "256_3_nofma.h"

void compute(TYPE farr[][SIZE], TYPE mult, int index) {  
  for(long i=0; i<MAXFLOPS_ITERS; i++) {
      farr[index][0]  = _mm256_add_ps(farr[index][0], farr[index][1]);
      farr[index][2]  = _mm256_add_ps(farr[index][2], farr[index][3]);
      farr[index][4]  = _mm256_add_ps(farr[index][4], farr[index][5]);
    }
}
