#ifndef __ARCH__
#define __ARCH__

#include <immintrin.h>
#define SIZE OP_PER_IT*2

#ifdef AVX_512_12
  #include "512_12.h"
#elif defined AVX_256_10
  #include "256_10.h"
#elif defined AVX_256_8
  #include "256_8.h"
#endif

#ifdef AVX_512_12
  #define BYTES_IN_VECT 64
  #define TYPE __m512

  static void initialize(int n_threads, TYPE mult, TYPE farr[][SIZE], float *fa) {
    mult = _mm512_set1_ps(0.1f);

    for(int i=0;i<n_threads;i++) {
      for(int j=0;j<SIZE;j++)
        farr[i][j] = _mm512_set_ps (fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],
                                    fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j]);
    }
  }
#elif defined AVX_256_10 || defined AVX_256_8
  #define BYTES_IN_VECT 32
  #define TYPE __m256
  
  static void initialize(int n_threads, TYPE mult, TYPE farr[][SIZE], float *fa) {
    mult = _mm256_set1_ps(0.1f);
    
    for(int i=0;i<n_threads;i++)
      for(int j=0;j<SIZE;j++)
        farr[i][j] = _mm256_set_ps (fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j]);
  }
#endif

void compute(TYPE farr[][SIZE], TYPE mult, int index);

#endif
