#include "512_12.h"

#define OP_PER_IT B_512_12_OP_IT

TYPE farr_512_12[MAX_NUMBER_THREADS][SIZE] __attribute__((aligned(64)));

void compute_512_12(TYPE *farr, TYPE mult, int index) {
  farr = farr_512_12[index];
  
  for(long i=0; i<MAXFLOPS_ITERS; i++) {
    farr[0]  = _mm512_fmadd_ps(mult, farr[0], farr[1]);
    farr[2]  = _mm512_fmadd_ps(mult, farr[2], farr[3]);
    farr[4]  = _mm512_fmadd_ps(mult, farr[4], farr[5]);
    farr[6]  = _mm512_fmadd_ps(mult, farr[6], farr[7]);
    farr[8]  = _mm512_fmadd_ps(mult, farr[8], farr[9]);
    farr[10] = _mm512_fmadd_ps(mult, farr[10], farr[11]);
    farr[12] = _mm512_fmadd_ps(mult, farr[12], farr[13]);
    farr[14] = _mm512_fmadd_ps(mult, farr[14], farr[15]);
    farr[16] = _mm512_fmadd_ps(mult, farr[16], farr[17]);
    farr[18] = _mm512_fmadd_ps(mult, farr[18], farr[19]);
    farr[20] = _mm512_fmadd_ps(mult, farr[20], farr[21]);
    farr[22] = _mm512_fmadd_ps(mult, farr[22], farr[23]);
  }
}
