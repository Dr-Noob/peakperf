#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <omp.h>
#include <sys/time.h>

#define N_FA_ARRAYS 8 // from fa0 to fa7
#define MAXFLOPS_ITERS 1000000000
#define FLOPS_ARRAY_SIZE N_THREADS*17*16

#ifdef AVX_512
  #define BYTES_IN_VECT 64
  #define TYPE __m512
  #define INTR1 _mm512_fmadd_ps
  #define INTR2 _mm512_set1_ps
  #define INTR3 _mm512_set_ps
  #define INTR4 _mm512_add_ps
  float sum(__m512 x) {
    float *val = (float*) &x;
    float res = 0.0;
    for(int i=0;i<16;i++)res += val[i];
    return res;
  }
#elif defined AVX_256
  #define BYTES_IN_VECT 32
  #define TYPE __m256
  #define INTR1 _mm256_fmadd_ps
  #define INTR2 _mm256_set1_ps
  #define INTR3 _mm256_set_ps
  #define INTR4 _mm256_add_ps
  float sum(__m256 x) {
    float *val = (float*) &x;
    float res = 0.0;
    for(int i=0;i<8;i++)res += val[i];
    return res;
  }
#endif

TYPE farr[N_THREADS][17] __attribute__((aligned(64)));
float fa[FLOPS_ARRAY_SIZE];

void compute(int index, TYPE mult) {
  for(long i=0; i<MAXFLOPS_ITERS; i++) {
      farr[index][0] = INTR1(mult, farr[index][0], farr[index][1]);
      farr[index][2] = INTR1(mult, farr[index][2], farr[index][3]);
      farr[index][4] = INTR1(mult, farr[index][4], farr[index][5]);
      farr[index][6] = INTR1(mult, farr[index][6], farr[index][7]);

      farr[index][8] = INTR1(mult, farr[index][8], farr[index][9]);
      farr[index][10] = INTR1(mult, farr[index][10], farr[index][11]);
      farr[index][12] = INTR1(mult, farr[index][12], farr[index][13]);
      farr[index][14] = INTR1(mult, farr[index][14], farr[index][15]);
    }
}

int main() {
  omp_set_num_threads(N_THREADS);
  int i = 0;
  int t = 0;
  int c = 0;
  TYPE mult = INTR2(0.1f);

  for(i=0; i<FLOPS_ARRAY_SIZE; i++)
    fa[i] = (float)i + 0.1f;

#ifdef AVX_256
  for(i=0;i<N_THREADS;i++) {
    for(int j=0;j<16;j++)
      farr[i][j] = _mm256_set_ps (fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++]);

    farr[i][16] = _mm256_set_ps (0,0,0,0,0,0,0,0);
  }
#elif defined AVX_512
  for(i=0;i<N_THREADS;i++) {
    for(int j=0;j<16;j++)
      farr[i][j] = _mm512_set_ps (fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],
                                  fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++]);

    farr[i][16] = _mm512_set_ps (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
  }
#endif

  struct timeval t0,t1;
  gettimeofday(&t0, 0);
#pragma omp parallel for
  for(t=0; t<N_THREADS; t++)
    compute(t,mult);

  gettimeofday(&t1, 0);
  for(int t=0; t<N_THREADS; t++) {
    farr[t][0] = INTR4(farr[t][0], farr[t][1]);
    farr[t][2] = INTR4(farr[t][2], farr[t][3]);
    farr[t][4] = INTR4(farr[t][4], farr[t][5]);
    farr[t][6] = INTR4(farr[t][6], farr[t][7]);
    farr[t][8] = INTR4(farr[t][8], farr[t][9]);
    farr[t][10] = INTR4(farr[t][10], farr[t][11]);
    farr[t][12] = INTR4(farr[t][12], farr[t][13]);
    farr[t][14] = INTR4(farr[t][14], farr[t][15]);

    farr[t][0] = INTR4(farr[t][0], farr[t][2]);
    farr[t][4] = INTR4(farr[t][4], farr[t][6]);
    farr[t][8] = INTR4(farr[t][8], farr[t][10]);
    farr[t][12] = INTR4(farr[t][12], farr[t][14]);

    farr[t][0] = INTR4(farr[t][0], farr[t][4]);
    farr[t][8] = INTR4(farr[t][8], farr[t][12]);

    farr[t][0] = INTR4(farr[t][0], farr[t][8]);
  }

  for(int t=1; t<N_THREADS; t++)farr[0][0] += farr[t][0];

  fprintf(stderr,"%f\n\n",sum(farr[0][0]));

  double e_time = (double)((t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec)/1000000;
  double gflops = (double)((long)N_THREADS*MAXFLOPS_ITERS*N_FA_ARRAYS*(BYTES_IN_VECT/4)*2)/1000000000;
  fprintf(stderr, "Used %fs\n",e_time);
  fprintf(stderr, "Computed %.3f GFLOPS\n", gflops);
  fprintf(stderr, "%f GFLOPS/s\n",gflops/e_time);
}
