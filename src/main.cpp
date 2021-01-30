#include <stdint.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "getarg.h"
#include "cpu/cpu.h"
#include "gpu/gpu.h"

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
  /*
   * Idea for the future?
   * device == DEVICE_TYPE_HYBRID
   *    Nº  Time(s)  TFLOP/s (CPU +  GPU)
   *     1  2.50984   4.300  (500 + 3800)
   *     2  2.50898   4.310  (500 + 3810)
   */
  if(device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
      int n_threads = get_n_threads();
      bool list_benchs = list_benchmarks();
      bench_type benchmark_type = get_benchmark_type();

      return peakperf_cpu(n_trials, n_warmup_trials, n_threads, list_benchs, benchmark_type);
    #else
      printf(RED "ERROR" RESET " peakperf was not built with CPU support\n");
      return EXIT_FAILURE;
    #endif
  }
  else if (device == DEVICE_TYPE_GPU) {
    #ifdef DEVICE_GPU_ENABLED
      return peakperf_gpu(n_trials, n_warmup_trials);
    #else
      printf(RED "ERROR:" RESET " peakperf was not built with GPU support\n");
      return EXIT_FAILURE;
    #endif
  }
}
