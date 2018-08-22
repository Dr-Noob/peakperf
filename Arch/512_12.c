#include "512_12.h"

TYPE mult;
TYPE farr[N_THREADS][SIZE] __attribute__((aligned(64)));

float sum(__m512 x) {
  float *val = (float*) &x;
  float res = 0.0;
  for(int i=0;i<16;i++)res += val[i];
  return res;
}

void compute(int index) {
  for(long i=0; i<MAXFLOPS_ITERS; i++) {
      farr[index][0]  = _mm512_fmadd_ps(mult, farr[index][0], farr[index][1]);
      farr[index][2]  = _mm512_fmadd_ps(mult, farr[index][2], farr[index][3]);
      farr[index][4]  = _mm512_fmadd_ps(mult, farr[index][4], farr[index][5]);
      farr[index][6]  = _mm512_fmadd_ps(mult, farr[index][6], farr[index][7]);
      farr[index][8]  = _mm512_fmadd_ps(mult, farr[index][8], farr[index][9]);
      farr[index][10] = _mm512_fmadd_ps(mult, farr[index][10], farr[index][11]);
      farr[index][12] = _mm512_fmadd_ps(mult, farr[index][12], farr[index][13]);
      farr[index][14] = _mm512_fmadd_ps(mult, farr[index][14], farr[index][15]);
      farr[index][16] = _mm512_fmadd_ps(mult, farr[index][16], farr[index][17]);
      farr[index][18] = _mm512_fmadd_ps(mult, farr[index][18], farr[index][19]);
      farr[index][20] = _mm512_fmadd_ps(mult, farr[index][20], farr[index][21]);
      farr[index][22] = _mm512_fmadd_ps(mult, farr[index][22], farr[index][23]);
    }
}

void initialize(float fa[FLOPS_ARRAY_SIZE]) {
  int c = 0;
  mult = _mm512_set1_ps(0.1f);

  for(int i=0;i<N_THREADS;i++) {
    for(int j=0;j<SIZE;j++)
      farr[i][j] = _mm512_set_ps (fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],
                                  fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++]);
  }
}

float summarize() {
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
