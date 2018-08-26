#ifndef __ARCH__
#define __ARCH__

#include <immintrin.h>

#define FLOPS_ARRAY_SIZE N_THREADS*SIZE
#define SIZE OP_PER_IT*2

#ifdef AVX_512_12
  #define BYTES_IN_VECT 64
  #define TYPE __m512
  TYPE mult;
  TYPE farr[N_THREADS][SIZE] __attribute__((aligned(64)));

  static float sum(__m512 x) {
    float *val = (float*) &x;
    float res = 0.0;
    for(int i=0;i<16;i++)res += val[i];
    return res;
  }

  static void initialize(float fa[FLOPS_ARRAY_SIZE]) {
    int c = 0;
    mult = _mm512_set1_ps(0.1f);

    for(int i=0;i<N_THREADS;i++) {
      for(int j=0;j<SIZE;j++)
        farr[i][j] = _mm512_set_ps (fa[i*SIZE+j],fa[i*SIZE+j+1],fa[i*SIZE+j+2],fa[i*SIZE+j+3],fa[i*SIZE+j+4],fa[i*SIZE+j+5],fa[i*SIZE+j+6],fa[i*SIZE+j+7],
                                    fa[i*SIZE+j+8],fa[i*SIZE+j+9],fa[i*SIZE+j+10],fa[i*SIZE+j+11],fa[i*SIZE+j+12],fa[i*SIZE+j+13],fa[i*SIZE+j+14],fa[i*SIZE+j+15]);
    }
  }

  static float summarize() {
    for(int t=0; t<N_THREADS; t++) {
      farr[t][0]  = _mm512_add_ps(farr[t][0], farr[t][1]);
      farr[t][2]  = _mm512_add_ps(farr[t][2], farr[t][3]);
      farr[t][4]  = _mm512_add_ps(farr[t][4], farr[t][5]);
      farr[t][6]  = _mm512_add_ps(farr[t][6], farr[t][7]);
      farr[t][8]  = _mm512_add_ps(farr[t][8], farr[t][9]);
      farr[t][10] = _mm512_add_ps(farr[t][10], farr[t][11]);
      farr[t][12] = _mm512_add_ps(farr[t][12], farr[t][13]);
      farr[t][14] = _mm512_add_ps(farr[t][14], farr[t][15]);
      farr[t][16] = _mm512_add_ps(farr[t][16], farr[t][17]);
      farr[t][18] = _mm512_add_ps(farr[t][18], farr[t][19]);
      farr[t][20] = _mm512_add_ps(farr[t][20], farr[t][21]);
      farr[t][22] = _mm512_add_ps(farr[t][22], farr[t][23]);

      farr[t][0]  = _mm512_add_ps(farr[t][0], farr[t][2]);
      farr[t][4]  = _mm512_add_ps(farr[t][4], farr[t][6]);
      farr[t][8]  = _mm512_add_ps(farr[t][8], farr[t][10]);
      farr[t][12] = _mm512_add_ps(farr[t][12], farr[t][14]);
      farr[t][16] = _mm512_add_ps(farr[t][16], farr[t][18]);
      farr[t][20] = _mm512_add_ps(farr[t][20], farr[t][22]);

      farr[t][0]  = _mm512_add_ps(farr[t][0], farr[t][4]);
      farr[t][8]  = _mm512_add_ps(farr[t][8], farr[t][12]);
      farr[t][16]  = _mm512_add_ps(farr[t][16], farr[t][20]);

      farr[t][0]  = _mm512_add_ps(farr[t][0], farr[t][8]);
      farr[t][0]  = _mm512_add_ps(farr[t][0], farr[t][16]);
    }

    for(int t=1; t<N_THREADS; t++)farr[0][0] += farr[t][0];

    return sum(farr[0][0]);
  }
#elif defined AVX_256_10 || defined AVX_256_8
  #define BYTES_IN_VECT 32
  #define TYPE __m256
  TYPE mult;
  TYPE farr[N_THREADS][SIZE] __attribute__((aligned(64)));

  static float sum(__m256 x) {
    float *val = (float*) &x;
    float res = 0.0;
    for(int i=0;i<8;i++)res += val[i];
    return res;
  }

  static void initialize(float fa[FLOPS_ARRAY_SIZE]) {
    mult = _mm256_set1_ps(0.1f);

    for(int i=0;i<N_THREADS;i++)
      for(int j=0;j<SIZE;j++)
        farr[i][j] = _mm256_set_ps (fa[i*SIZE+j],fa[i*SIZE+j+1],fa[i*SIZE+j+2],fa[i*SIZE+j+3],fa[i*SIZE+j+4],fa[i*SIZE+j+5],fa[i*SIZE+j+6],fa[i*SIZE+j+7]);
  }

  static float summarize() {
    for(int t=0; t<N_THREADS; t++) {
      farr[t][0]  = _mm256_add_ps(farr[t][0], farr[t][1]);
      farr[t][2]  = _mm256_add_ps(farr[t][2], farr[t][3]);
      farr[t][4]  = _mm256_add_ps(farr[t][4], farr[t][5]);
      farr[t][6]  = _mm256_add_ps(farr[t][6], farr[t][7]);
      farr[t][8]  = _mm256_add_ps(farr[t][8], farr[t][9]);
      farr[t][10] = _mm256_add_ps(farr[t][10], farr[t][11]);
      farr[t][12] = _mm256_add_ps(farr[t][12], farr[t][13]);
      farr[t][14] = _mm256_add_ps(farr[t][14], farr[t][15]);

      farr[t][0]  = _mm256_add_ps(farr[t][0], farr[t][2]);
      farr[t][4]  = _mm256_add_ps(farr[t][4], farr[t][6]);
      farr[t][8]  = _mm256_add_ps(farr[t][8], farr[t][10]);
      farr[t][12] = _mm256_add_ps(farr[t][12], farr[t][14]);

      farr[t][0]  = _mm256_add_ps(farr[t][0], farr[t][4]);
      farr[t][8]  = _mm256_add_ps(farr[t][8], farr[t][12]);

      farr[t][0]  = _mm256_add_ps(farr[t][0], farr[t][8]);
    }

    for(int t=1; t<N_THREADS; t++)farr[0][0] += farr[t][0];

    return sum(farr[0][0]);
  }
#endif

void compute(int index);

#endif
