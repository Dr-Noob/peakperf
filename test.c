#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#define MAXFLOPS_ITERS 1000000000
#define TYPE __m256
TYPE farr[3] __attribute__((aligned(64)));
float fa[8*3];

float sum(__m256 x) {
  float *val = (float*) &x;
  float res = 0.0;
  for(int i=0;i<8;i++)res += val[i];
  return res;
}

void compute(TYPE mult) {
  for(long i=0; i<MAXFLOPS_ITERS; i++) {
      farr[0] = _mm256_fmadd_ps(mult, farr[0], farr[1]);
      farr[2] = _mm256_fmadd_ps(mult, farr[2], farr[3]);
      farr[4] = _mm256_fmadd_ps(mult, farr[4], farr[5]);
      farr[6] = _mm256_fmadd_ps(mult, farr[6], farr[7]);
      farr[8] = _mm256_fmadd_ps(mult, farr[8], farr[9]);
      farr[10] = _mm256_fmadd_ps(mult, farr[10], farr[11]);
      farr[12] = _mm256_fmadd_ps(mult, farr[12], farr[13]);
      farr[14] = _mm256_fmadd_ps(mult, farr[14], farr[15]);
      farr[16] = _mm256_fmadd_ps(mult, farr[16], farr[17]);
      farr[18] = _mm256_fmadd_ps(mult, farr[18], farr[19]);
      farr[20] = _mm256_fmadd_ps(mult, farr[20], farr[21]);
    }
}

int main() {
  int i = 0;
  int t = 0;
  int c = 0;
  TYPE mult = _mm256_set1_ps(0.1f);

  for(i=0; i<8*3; i++)
    fa[i] = (float)i + 0.1f;

  farr[0] = _mm256_set_ps (fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++]);
  farr[1] = _mm256_set_ps (fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++]);
  farr[2] = _mm256_set_ps (fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++],fa[c++]);

  struct timeval t0,t1;
  gettimeofday(&t0, 0);

  compute(mult);

  gettimeofday(&t1, 0);
  farr[0] = _mm256_add_ps(farr[0], farr[1]);
  farr[2] = _mm256_add_ps(farr[0], farr[2]);

  fprintf(stderr,"%f\n\n",sum(farr[0]));

  double e_time = (double)((t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec)/1000000;
  fprintf(stderr, "Used %fs\n",e_time);
}
