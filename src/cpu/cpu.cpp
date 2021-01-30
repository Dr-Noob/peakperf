#include <stdint.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "cpu.h"
#include "cpufetch/cpufetch.h"
#include "cpufetch/uarch.h"
#include "arch/arch.h"

#define RED   "\x1b[31;1m"
#define BOLD  "\x1b[1m"
#define GREEN "\x1b[42m"
#define RESET "\x1b[0m"

int peakperf_cpu(int nTrials, int nWarmupTrials, int n_threads, bool list_benchmarks, bench_type benchmark_type) {
  struct cpu* cpu = get_cpu_info();

  if(list_benchmarks) {
    print_bench_types(cpu);
    return EXIT_SUCCESS;
  }

  if(n_threads == INVALID_N_THREADS) n_threads = omp_get_max_threads();

  struct timeval t0;
  struct timeval t1;
  struct benchmark* bench = init_benchmark(cpu, n_threads, benchmark_type);
  if(bench == NULL) {
    return EXIT_FAILURE;
  }

  double gflops = get_gflops(bench);
  double e_time = 0;
  double mean = 0;
  double sd = 0;
  double sum = 0;
  double* gflops_list = (double*) malloc(sizeof(double) * nTrials);
  char* cpu_name = get_str_cpu_name(cpu);
  char* uarch_name = get_str_uarch(cpu);
  const char* bench_name = get_benchmark_name(bench);
  int line_length = 13 + strlen(cpu_name);

  putchar('\n');
  for(int i=0; i < line_length; i++) putchar('-');
  printf("\n" BOLD "    peakperf (https://github.com/Dr-Noob/peakperf)" RESET "\n");
  for(int i=0; i < line_length; i++) putchar('-');
  putchar('\n');
  printf("        CPU: %s\n", cpu_name);
  printf("  Microarch: %s\n", uarch_name);
  printf("  Benchmark: %s\n", bench_name);
  printf(" Iterations: %d\n", MAXFLOPS_ITERS);
  printf("      GFLOP: %.2f\n", gflops);
  printf("    Threads: %d\n\n", n_threads);

  printf(BOLD "%6s %8s %8s" RESET "\n","Nº","Time(s)","GFLOP/s");
  for (int trial = 0; trial < nTrials+nWarmupTrials; trial++) {
    gettimeofday(&t0, 0);
    compute(bench);
    gettimeofday(&t1, 0);

    e_time = (double)((t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec)/1000000;
    if (trial >= nWarmupTrials) {
      mean += gflops/e_time;
      gflops_list[trial-nWarmupTrials] = gflops/e_time;
      printf("%5d %8.5f %8.2f\n",trial+1, e_time, gflops/e_time);
    }
    else {
      printf("%5d %8.5f %8.2f *\n",trial+1, e_time, gflops/e_time);
    }
  }

  mean=mean/(double)nTrials;
  for(int i=0;i<nTrials;i++)
    sum += (gflops_list[i] - mean)*(gflops_list[i] - mean);
  sd=sqrt(sum/nTrials);

  for(int i=0; i < line_length; i++) putchar('-');
  printf("\n" BOLD " Average performance:      " RESET GREEN "%.2f +- %.2f GFLOP/s" RESET "\n",mean, sd);
  for(int i=0; i < line_length; i++) putchar('-');
  if(nWarmupTrials > 0)
    printf("\n* - warm-up, not included in average");
  printf("\n\n");

  return EXIT_SUCCESS;
}
