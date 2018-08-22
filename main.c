#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>
#include "getarg.h"

#define DEFAULT_N_TRIALS      10
#define DEFAULT_WARMUP_TRIALS 2

#define RED   "\x1b[31;1m"
#define BOLD  "\x1b[1m"
#define GREEN "\x1b[42m"
#define RESET "\x1b[0m"

#ifdef AVX_512_12
  #include "Arch/512_12.h"
#elif defined AVX_256_10
  #include "Arch/256_10.h"
#endif

float fa[FLOPS_ARRAY_SIZE];

int main(int argc, char* argv[]) {
  int nTrials = DEFAULT_N_TRIALS;
  int nWarmupTrials = DEFAULT_WARMUP_TRIALS;

  /*** BASIC ARGS PROCESSING ***/
  if(argc > 3 || argc == 2) {
    printf(RED "ERROR: Wrong number of parameters" RESET "\n");
    printf(RED "Usage: %s [number_trials number_warmup_trials]" RESET "\n",argv[0]);
    return EXIT_FAILURE;
  } else if (argc == 3) {
    nTrials = getarg_int(1,argv);
    if(errn != 0) {
      printf(RED "ERROR: number_trials is not valid: ");
      printerror();
      printf(RESET);
      return EXIT_FAILURE;
    }
    nWarmupTrials = getarg_int(2,argv);
    if(errn != 0) {
      printf(RED "ERROR: number_warmup_trials is not valid: ");
      printerror();
      printf(RESET);
      return EXIT_FAILURE;
    }

    if(nTrials <= 0) {
      printf(RED "ERROR: number_trials must be greater than zero" RESET "\n");
      return EXIT_FAILURE;
    }
    if(nWarmupTrials < 0) {
      printf(RED "ERROR: number_warmup_trials cannot be less than zero" RESET "\n");
      return EXIT_FAILURE;
    }
  }

  struct timeval t0,t1;
  /*** NEEDED TO COMPUTE SD ***/
  double gflops = (double)((long)N_THREADS*MAXFLOPS_ITERS*OP_PER_IT*(BYTES_IN_VECT/4)*2)/1000000000;
  double e_time = 0;
  double mean = 0;
  double sd = 0;
  double sum = 0;
  double gflops_list[nTrials];

  omp_set_num_threads(N_THREADS);
  for(int i=0; i<FLOPS_ARRAY_SIZE; i++)
    fa[i] = (float)i + 0.1f;

  initialize(fa);

  printf("\n" BOLD "Benchmarking FLOPS by Dr-Noob(github.com/Dr-Noob/FLOPS)." RESET "\n");
  printf("   Test name: %s\n",TEST_NAME);
  printf("  Iterations: %li\n",MAXFLOPS_ITERS);
  printf("       GFLOP: %.2f\n",gflops);
  printf("     Threads: %d\n\n", N_THREADS);

  printf(BOLD "%6s %8s %8s" RESET "\n","Nº","Time(s)","GFLOP/S");
  for (int trial = 0; trial < nTrials+nWarmupTrials; trial++) {
    /*** COMPUTE TAKES PLACE HERE ***/
    gettimeofday(&t0, 0);
    #pragma omp parallel for
      for(int t=0; t<N_THREADS; t++)
        compute(t);
    gettimeofday(&t1, 0);

    /*** NOW CALCULATE TIME AND PERFORMANCE ***/
    e_time = (double)((t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec)/1000000;
    if ( trial >= nWarmupTrials) {
      mean += gflops/e_time;
      gflops_list[trial-nWarmupTrials] = gflops/e_time;
      printf("%5d %8.5f %8.2f\n",trial+1, e_time, gflops/e_time);
    }
    else
      printf("%5d %8.5f %8.2f *\n",trial+1, e_time, gflops/e_time);
  }

  /*** CALCULATE STANDART DERIVATION ***/
  mean=mean/(double)nTrials;
  for(int i=0;i<nTrials;i++)
    sum += (gflops_list[i] - mean)*(gflops_list[i] - mean);
  sd=sqrt(sum/nTrials);

  printf("-------------------------------------------------\n");
  printf(BOLD "Average performance:      " RESET GREEN "%.2f +- %.2f GFLOP/s" RESET "\n",mean, sd);
  printf("-------------------------------------------------\n");
  if(nWarmupTrials > 0)printf("* - warm-up, not included in average\n\n");
}
