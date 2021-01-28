#include <stdint.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "getarg.h"
#include "cpu/cpu.h"

#define RED   "\x1b[31;1m"
#define BOLD  "\x1b[1m"
#define GREEN "\x1b[42m"
#define RESET "\x1b[0m"

static const char* VERSION = "1.02";

void printHelp(char *argv[]) {
  printf("Usage: %s [-h] [-v] [-l] [-b bench_type] [-r n_trials] [-w warmup_trials] [-t n_threads] \n\
    Options: \n\
      -h      Prints this help and exit\n\
      -v      Prints peakperf version and exit\n\
      -l      List the avaiable benchmark types\n\
      -b      Select a specific benchmark to run\n\
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

  if(get_mode() == PEAKPERF_MODE_CPU) {
    int nTrials = get_n_trials();
    int nWarmupTrials = get_warmup_trials();
    int n_threads = get_n_threads();
    bool list_benchs = list_benchmarks();
    bench_type benchmark_type = get_benchmark_type();

    return peakperf_cpu(nTrials, nWarmupTrials, n_threads, list_benchs, benchmark_type);
  }
  else {
    printf("ERROR\n");
    return EXIT_FAILURE;
  }
}
