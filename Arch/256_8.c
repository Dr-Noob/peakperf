#include "256_8.h"

void compute(TYPE farr[][SIZE], TYPE mult, int index) {  
  for(long i=0; i<MAXFLOPS_ITERS; i++) {
      farr[index][0]  = _mm256_fmadd_ps(mult, farr[index][0], farr[index][1]);
      farr[index][2]  = _mm256_fmadd_ps(mult, farr[index][2], farr[index][3]);
      farr[index][4]  = _mm256_fmadd_ps(mult, farr[index][4], farr[index][5]);
      farr[index][6]  = _mm256_fmadd_ps(mult, farr[index][6], farr[index][7]);
      farr[index][8]  = _mm256_fmadd_ps(mult, farr[index][8], farr[index][9]);
      farr[index][10] = _mm256_fmadd_ps(mult, farr[index][10], farr[index][11]);
      farr[index][12] = _mm256_fmadd_ps(mult, farr[index][12], farr[index][13]);
      farr[index][14] = _mm256_fmadd_ps(mult, farr[index][14], farr[index][15]);
    }
}
