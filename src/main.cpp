#include <stdint.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "getarg.h"
#include "benchmark.h"

#define RED   "\x1b[31;1m"
#define BOLD  "\x1b[1m"
#define GREEN "\x1b[42m"
#define RESET "\x1b[0m"

static const char* VERSION = "1.02";

void printHelp(char *argv[]) {
  printf("Usage: %s [-h] [-v] [-l] [-b bench_type] [-d device] [-r n_trials] [-w warmup_trials] [-t n_threads] \n\
    Options: \n\
      -h      Prints this help and exit\n\
      -v      Prints peakperf version and exit\n\
      -l      List the avaiable benchmark types\n\
      -b      Select a specific benchmark to run\n\
      -d      Select the device to run the benchmark on. Possible values are:\n\
        * cpu Run peakperf in the CPU (default)\n\
        * gpu Run peakperf in the GPU\n\
      -r      Set the number of trials of the benchmark\n\
      -w      Set the number of warmup trials\n\
      -t      Set the number of threads to use\n",
      argv[0]);
}

void print_version() {
  printf("peakperf v%s\n", VERSION);
}

int main(int argc, char* argv[]) {
  if(!parseArgs(argc, argv))
    return EXIT_FAILURE;

  if(showHelp()) {
    printHelp(argv);
    return EXIT_SUCCESS;
  }

  if(showVersion()) {
    print_version();
    return EXIT_SUCCESS;
  }

  int n_trials = get_n_trials();
  int n_warmup_trials = get_warmup_trials();
  device_type device = get_device_type();

  struct benchmark* bench = init_benchmark_device(device);
  if(bench == NULL) {
    return EXIT_FAILURE;
  }
  int n_threads = get_n_threads();
  bool list_benchs = list_benchmarks();
  bench_type benchmark_type = get_benchmark_type(bench);
  struct hardware* hw = get_hardware_info();
  struct config* cfg = get_benchmark_config(bench, n_threads);

  /*if(list_benchmarks) {
    print_bench_types(hw, bench);
    return EXIT_SUCCESS;
  }*/

  struct timeval t0;
  struct timeval t1;
  bool flag = init_benchmark(bench, hw, cfg, benchmark_type);
  if(!flag) {
    return EXIT_FAILURE;
  }

  double gflops = get_gflops(bench);
  double e_time = 0;
  double mean = 0;
  double sd = 0;
  double sum = 0;
  double* gflops_list = (double*) malloc(sizeof(double) * n_trials);
  const char* bench_name = get_benchmark_name(bench);
  int line_length = 13; // + strlen(cpu_name);

  putchar('\n');
  for(int i=0; i < line_length; i++) putchar('-');
  printf("\n" BOLD "    peakperf (https://github.com/Dr-Noob/peakperf)" RESET "\n");
  for(int i=0; i < line_length; i++) putchar('-');
  putchar('\n');
  print_hw_info(hw);
  printf("  Benchmark: %s\n", bench_name);
  printf(" Iterations: %d\n", MAXFLOPS_ITERS);
  printf("      GFLOP: %.2f\n", gflops);
  print_bench_cfg(cfg);

  printf(BOLD "%6s %8s %8s" RESET "\n","Nº","Time(s)","GFLOP/s");
  for (int trial = 0; trial < n_trials+n_warmup_trials; trial++) {
    gettimeofday(&t0, 0);
    compute(bench);
    gettimeofday(&t1, 0);

    e_time = (double)((t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec)/1000000;
    if (trial >= n_warmup_trials) {
      mean += gflops/e_time;
      gflops_list[trial-n_warmup_trials] = gflops/e_time;
      printf("%5d %8.5f %8.2f\n",trial+1, e_time, gflops/e_time);
    }
    else {
      printf("%5d %8.5f %8.2f *\n",trial+1, e_time, gflops/e_time);
    }
  }

  mean=mean/(double)n_trials;
  for(int i=0;i<n_trials;i++)
    sum += (gflops_list[i] - mean)*(gflops_list[i] - mean);
  sd=sqrt(sum/n_trials);

  for(int i=0; i < line_length; i++) putchar('-');
  printf("\n" BOLD " Average performance:      " RESET GREEN "%.2f +- %.2f GFLOP/s" RESET "\n",mean, sd);
  for(int i=0; i < line_length; i++) putchar('-');
  if(n_warmup_trials > 0)
    printf("\n* - warm-up, not included in average");
  printf("\n\n");

  return EXIT_SUCCESS;
}
