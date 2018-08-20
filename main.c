#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <omp.h>
#include <sys/time.h>

#ifdef AVX_512
  #include "Knl.h"
#elif defined AVX_256
  #include "Haswell.h"
#endif

float fa[FLOPS_ARRAY_SIZE];

int main(int argc, char* argv[]) {
  omp_set_num_threads(N_THREADS);
  for(int i=0; i<FLOPS_ARRAY_SIZE; i++)
    fa[i] = (float)i + 0.1f;

  initialize(fa);

  printf("Running test '%s' with %d iterations on %d threads...\n",TEST_NAME,MAXFLOPS_ITERS,N_THREADS);
  struct timeval t0,t1;
  gettimeofday(&t0, 0);
#pragma omp parallel for
  for(int t=0; t<N_THREADS; t++)
    compute(t);

  gettimeofday(&t1, 0);

  fprintf(stderr,"Result is '%f'\n",summarize());

  double e_time = (double)((t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec)/1000000;
  double gflops = (double)((long)N_THREADS*MAXFLOPS_ITERS*OP_PER_IT*(BYTES_IN_VECT/4)*2)/1000000000;
  printf("Used %fs\n",e_time);
  printf("Computed %.3f GFLOPS\n", gflops);
  printf("%f GFLOPS/s\n",gflops/e_time);
}
