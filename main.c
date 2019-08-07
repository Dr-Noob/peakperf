#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>
#include "cpufetch.h"
#include "getarg.h"
#include "Arch/Arch.h"

#define RED   "\x1b[31;1m"
#define BOLD  "\x1b[1m"
#define GREEN "\x1b[42m"
#define RESET "\x1b[0m"

void printHelp(char *argv[]) {
  printf("Usage: %s [-h] [-r n_trials] [-w warmup_trials] [-t n_threads] \n\
    Options: \n\
      -h      Print this help and exit\n\
      -r      Set the number of trials of the benchmark\n\
      -w      Set the number of warmup trials\n\
      -t      Set the number of threads to use\n",
      argv[0]);
}

int main(int argc, char* argv[]) {
  if(!parseArgs(argc, argv)) return EXIT_FAILURE;
  
  if(showHelp()) {
    printHelp(argv);
    return EXIT_SUCCESS;
  }
  int nTrials = get_n_trials();
  int nWarmupTrials = get_warmup_trials();
  int n_threads = get_n_threads();
  if(n_threads == INVALID_N_THREADS) n_threads = omp_get_max_threads();
  
  int flops_array_size = SIZE*n_threads;
  struct timeval t0,t1;
  
  /*** NEEDED TO COMPUTE SD ***/
  double gflops = (double)((long)n_threads*MAXFLOPS_ITERS*OP_PER_IT*(BYTES_IN_VECT/4)*2)/1000000000;
  double e_time = 0;
  double mean = 0;
  double sd = 0;
  double sum = 0;
  double gflops_list[nTrials];
  char* cpu_name = getString_CPUName();

  omp_set_num_threads(n_threads);
  
  /*** PREPARE DATA ***/
  TYPE mult = {0};
  TYPE farr[n_threads][SIZE] __attribute__((aligned(64)));
  float fa[flops_array_size];
  for(int i=0; i<flops_array_size; i++)
    fa[i] = (float)i + 0.1f;

  initialize(n_threads, mult, farr, fa);

  printf("\n" BOLD "Benchmarking FLOPS by Dr-Noob(github.com/Dr-Noob/FLOPS)." RESET "\n");
  printf("   Test name: %s\n",TEST_NAME);
  printf("         CPU: %s\n",cpu_name);
  printf("  Iterations: %d\n",MAXFLOPS_ITERS);
  printf("       GFLOP: %.2f\n",gflops);
  printf("     Threads: %d\n\n", n_threads);

  printf(BOLD "%6s %8s %8s" RESET "\n","Nº","Time(s)","GFLOP/S");
  for (int trial = 0; trial < nTrials+nWarmupTrials; trial++) {
    /*** COMPUTE TAKES PLACE HERE ***/
    gettimeofday(&t0, 0);
    #pragma omp parallel for
    for(int t=0; t<n_threads; t++)
      compute(farr, mult, t);
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

  /*** CALCULATE STANDARD DEVIATION ***/
  mean=mean/(double)nTrials;
  for(int i=0;i<nTrials;i++)
    sum += (gflops_list[i] - mean)*(gflops_list[i] - mean);
  sd=sqrt(sum/nTrials);

  printf("-------------------------------------------------\n");
  printf(BOLD "Average performance:      " RESET GREEN "%.2f +- %.2f GFLOP/s" RESET "\n",mean, sd);
  printf("-------------------------------------------------\n");
  if(nWarmupTrials > 0)printf("* - warm-up, not included in average\n\n");
}
