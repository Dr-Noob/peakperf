#ifndef __ARCH__
#define __ARCH__

#include <immintrin.h>

#define SIZE OP_PER_IT*2

#ifdef AVX_512_12
  #define BYTES_IN_VECT 64
  #define TYPE __m512

  static float sum(__m512 x) {
    float *val = (float*) &x;
    float res = 0.0;
    for(int i=0;i<16;i++)res += val[i];
    return res;
  }

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

  static float sum(__m256 x) {
    float *val = (float*) &x;
    float res = 0.0;
    for(int i=0;i<8;i++)res += val[i];
    return res;
  }

  static void initialize(int n_threads, TYPE mult, TYPE farr[][SIZE], float *fa) {
    mult = _mm256_set1_ps(0.1f);
    
    for(int i=0;i<n_threads;i++)
      for(int j=0;j<SIZE;j++)
        farr[i][j] = _mm256_set_ps (fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j]);
  }
#endif

void compute(TYPE farr[][SIZE], TYPE mult, int index);

#endif
