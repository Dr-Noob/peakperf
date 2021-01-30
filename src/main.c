#include <stdint.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "getarg.h"
#include "cpu/cpu.h"
#include "gpu/gpu.h"

static const char* VERSION = "1.02";

void printHelp(char *argv[]) {
  printf("Usage: %s [-h] [-v] [-l] [-b bench_type] [-m bench_mode] [-r n_trials] [-w warmup_trials] [-t n_threads] \n\
    Options: \n\
      -h      Prints this help and exit\n\
      -v      Prints peakperf version and exit\n\
      -l      List the avaiable benchmark types\n\
      -b      Select a specific benchmark to run\n\
      -m      Select the benchmark mode. Possible values are:\n\
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
  bench_mode mode = get_benchmark_mode();
  
  if(mode == BENCH_MODE_CPU) {    
    int n_threads = get_n_threads();
    bool list_benchs = list_benchmarks();
    bench_type benchmark_type = get_benchmark_type();

    return peakperf_cpu(n_trials, n_warmup_trials, n_threads, list_benchs, benchmark_type);
  }
  else if (mode == BENCH_MODE_GPU) {
    return peakperf_gpu(n_trials, n_warmup_trials);
  }
}
