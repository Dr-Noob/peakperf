#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <omp.h>

#define MAXFLOPS_ITERS 1000000000
#define FLOPS_ARRAY_SIZE (1024*1024)
#define LOOP_COUNT 64
#define N_THREADS 8

float sum(__m256 x) {
  float *val = (float*) &x;
  float res = 0.0;
  for(int i=0;i<8;i++)res += val[i];
  return res;
}

float fa[FLOPS_ARRAY_SIZE] __attribute__((aligned(64)));
float fb[FLOPS_ARRAY_SIZE] __attribute__((aligned(64)));

__m256 fa0[N_THREADS];
__m256 fa1[N_THREADS];
__m256 fa2[N_THREADS];
__m256 fa3[N_THREADS];
__m256 fa4[N_THREADS];
__m256 fa5[N_THREADS];
__m256 fa6[N_THREADS];
__m256 fa7[N_THREADS];

__m256 fb0[N_THREADS];
__m256 fb1[N_THREADS];
__m256 fb2[N_THREADS];
__m256 fb3[N_THREADS];
__m256 fb4[N_THREADS];
__m256 fb5[N_THREADS];
__m256 fb6[N_THREADS];
__m256 fb7[N_THREADS];

void compute(int index, __m256 mult) {
  for(int i=0; i<MAXFLOPS_ITERS; i++) {
      fa0[index] = _mm256_fmadd_ps(mult, fa0[index], fb0[index]);
      fa1[index] = _mm256_fmadd_ps(mult, fa1[index], fb1[index]);
      fa2[index] = _mm256_fmadd_ps(mult, fa2[index], fb2[index]);
      fa3[index] = _mm256_fmadd_ps(mult, fa3[index], fb3[index]);
      fa4[index] = _mm256_fmadd_ps(mult, fa4[index], fb4[index]);
      fa5[index] = _mm256_fmadd_ps(mult, fa5[index], fb5[index]);
      fa6[index] = _mm256_fmadd_ps(mult, fa6[index], fb6[index]);
      fa7[index] = _mm256_fmadd_ps(mult, fa7[index], fb7[index]);
  }
}

int main() {
  omp_set_num_threads(N_THREADS);
  int i = 0;
  int t = 0;
  __m256 mult = _mm256_set1_ps(0.1f);

  for(i=0; i<FLOPS_ARRAY_SIZE; i++) {
    fa[i] = (float)i + 0.1f;
    fb[i] = (float)i + 0.2f;
  }

  for(i=0;i<N_THREADS;i++) {
    fa0[i] = _mm256_set_ps (fa[LOOP_COUNT*i],fa[LOOP_COUNT*i+1],fa[LOOP_COUNT*i+2],fa[LOOP_COUNT*i+3],fa[LOOP_COUNT*i+4],fa[LOOP_COUNT*i+5],fa[LOOP_COUNT*i+6],fa[LOOP_COUNT*i+7]);
    fa1[i] = _mm256_set_ps (fa[LOOP_COUNT*i+8],fa[LOOP_COUNT*i+9],fa[LOOP_COUNT*i+10],fa[LOOP_COUNT*i+11],fa[LOOP_COUNT*i+12],fa[LOOP_COUNT*i+13],fa[LOOP_COUNT*i+14],fa[LOOP_COUNT*i+15]);
    fa2[i] = _mm256_set_ps (fa[LOOP_COUNT*i+16],fa[LOOP_COUNT*i+17],fa[LOOP_COUNT*i+18],fa[LOOP_COUNT*i+19],fa[LOOP_COUNT*i+20],fa[LOOP_COUNT*i+21],fa[LOOP_COUNT*i+22],fa[LOOP_COUNT*i+23]);
    fa3[i] = _mm256_set_ps (fa[LOOP_COUNT*i+24],fa[LOOP_COUNT*i+25],fa[LOOP_COUNT*i+26],fa[LOOP_COUNT*i+27],fa[LOOP_COUNT*i+28],fa[LOOP_COUNT*i+29],fa[LOOP_COUNT*i+30],fa[LOOP_COUNT*i+31]);
    fa4[i] = _mm256_set_ps (fa[LOOP_COUNT*i+32],fa[LOOP_COUNT*i+33],fa[LOOP_COUNT*i+34],fa[LOOP_COUNT*i+35],fa[LOOP_COUNT*i+36],fa[LOOP_COUNT*i+37],fa[LOOP_COUNT*i+38],fa[LOOP_COUNT*i+39]);
    fa5[i] = _mm256_set_ps (fa[LOOP_COUNT*i+40],fa[LOOP_COUNT*i+41],fa[LOOP_COUNT*i+42],fa[LOOP_COUNT*i+43],fa[LOOP_COUNT*i+44],fa[LOOP_COUNT*i+45],fa[LOOP_COUNT*i+46],fa[LOOP_COUNT*i+47]);
    fa6[i] = _mm256_set_ps (fa[LOOP_COUNT*i+48],fa[LOOP_COUNT*i+49],fa[LOOP_COUNT*i+50],fa[LOOP_COUNT*i+51],fa[LOOP_COUNT*i+52],fa[LOOP_COUNT*i+53],fa[LOOP_COUNT*i+54],fa[LOOP_COUNT*i+55]);
    fa7[i] = _mm256_set_ps (fa[LOOP_COUNT*i+56],fa[LOOP_COUNT*i+57],fa[LOOP_COUNT*i+58],fa[LOOP_COUNT*i+59],fa[LOOP_COUNT*i+60],fa[LOOP_COUNT*i+61],fa[LOOP_COUNT*i+62],fa[LOOP_COUNT*i+63]);

    fb0[i] = _mm256_set_ps (fb[LOOP_COUNT*i+0],fb[LOOP_COUNT*i+1],fb[LOOP_COUNT*i+2],fb[LOOP_COUNT*i+3],fb[LOOP_COUNT*i+4],fb[LOOP_COUNT*i+5],fb[LOOP_COUNT*i+6],fb[LOOP_COUNT*i+7]);
    fb1[i] = _mm256_set_ps (fb[LOOP_COUNT*i+8],fb[LOOP_COUNT*i+9],fb[LOOP_COUNT*i+10],fb[LOOP_COUNT*i+11],fb[LOOP_COUNT*i+12],fb[LOOP_COUNT*i+13],fb[LOOP_COUNT*i+14],fb[LOOP_COUNT*i+15]);
    fb2[i] = _mm256_set_ps (fb[LOOP_COUNT*i+16],fb[LOOP_COUNT*i+17],fb[LOOP_COUNT*i+18],fb[LOOP_COUNT*i+19],fb[LOOP_COUNT*i+20],fb[LOOP_COUNT*i+21],fb[LOOP_COUNT*i+22],fb[LOOP_COUNT*i+23]);
    fb3[i] = _mm256_set_ps (fb[LOOP_COUNT*i+24],fb[LOOP_COUNT*i+25],fb[LOOP_COUNT*i+26],fb[LOOP_COUNT*i+27],fb[LOOP_COUNT*i+28],fb[LOOP_COUNT*i+29],fb[LOOP_COUNT*i+30],fb[LOOP_COUNT*i+31]);
    fb4[i] = _mm256_set_ps (fb[LOOP_COUNT*i+32],fb[LOOP_COUNT*i+33],fb[LOOP_COUNT*i+34],fb[LOOP_COUNT*i+35],fb[LOOP_COUNT*i+36],fb[LOOP_COUNT*i+37],fb[LOOP_COUNT*i+38],fb[LOOP_COUNT*i+39]);
    fb5[i] = _mm256_set_ps (fb[LOOP_COUNT*i+40],fb[LOOP_COUNT*i+41],fb[LOOP_COUNT*i+42],fb[LOOP_COUNT*i+43],fb[LOOP_COUNT*i+44],fb[LOOP_COUNT*i+45],fb[LOOP_COUNT*i+46],fb[LOOP_COUNT*i+47]);
    fb6[i] = _mm256_set_ps (fb[LOOP_COUNT*i+48],fb[LOOP_COUNT*i+49],fb[LOOP_COUNT*i+50],fb[LOOP_COUNT*i+51],fb[LOOP_COUNT*i+52],fb[LOOP_COUNT*i+53],fb[LOOP_COUNT*i+54],fb[LOOP_COUNT*i+55]);
    fb7[i] = _mm256_set_ps (fb[LOOP_COUNT*i+56],fb[LOOP_COUNT*i+57],fb[LOOP_COUNT*i+58],fb[LOOP_COUNT*i+59],fb[LOOP_COUNT*i+60],fb[LOOP_COUNT*i+61],fb[LOOP_COUNT*i+62],fb[LOOP_COUNT*i+63]);
  }

/*
[gcc isnt able to vectorize properly
instead, call the function compute, which
can be vectorized]

[icc could be able to do so but it is
really bad when using instrinsics]

#pragma omp parallel for private(i)
  for(t=0; t<N_THREADS; t++) {
    for(i=0; i<MAXFLOPS_ITERS; i++) {
        fa0[t] = _mm256_fmadd_ps(mult, fa0[t], fb0[t]);
        fa1[t] = _mm256_fmadd_ps(mult, fa1[t], fb1[t]);
        fa2[t] = _mm256_fmadd_ps(mult, fa2[t], fb2[t]);
        fa3[t] = _mm256_fmadd_ps(mult, fa3[t], fb3[t]);
        fa4[t] = _mm256_fmadd_ps(mult, fa4[t], fb4[t]);
        fa5[t] = _mm256_fmadd_ps(mult, fa5[t], fb5[t]);
        fa6[t] = _mm256_fmadd_ps(mult, fa6[t], fb6[t]);
        fa7[t] = _mm256_fmadd_ps(mult, fa7[t], fb7[t]);
    }
  }
*/

#pragma omp parallel for private(i)
  for(t=0; t<N_THREADS; t++)
    compute(t,mult);

  for(int t=0; t<N_THREADS; t++) {
    fa0[t] = _mm256_add_ps(fa0[t], fa1[t]);
    fa2[t] = _mm256_add_ps(fa2[t], fa3[t]);
    fa4[t] = _mm256_add_ps(fa4[t], fa5[t]);
    fa6[t] = _mm256_add_ps(fa6[t], fa7[t]);

    fa0[t] = _mm256_add_ps(fa0[t], fa2[t]);
    fa4[t] = _mm256_add_ps(fa4[t], fa6[t]);

    fa0[t] = _mm256_add_ps(fa0[t], fa4[t]);
  }

  for(int t=1; t<N_THREADS; t++)fa0[0] += fa0[t];

  fprintf(stderr,"%f\n",sum(fa0[0]));
}
