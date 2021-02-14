#include <stdint.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "getarg.hpp"
#include "benchmark.hpp"
#include "global.hpp"

#define MAXFLOPS_ITERS 1000000000

static const char* VERSION = "1.12";

void printHelp(char *argv[]) {
  const char **t = args_str;
  const char *c = args_chr;
  int max_len = -1;
  int len = sizeof(args_str) / sizeof(args_str[0]);
  for(int i=0; i < len; i++) {
    max_len = max(max_len, (int) strlen(args_str[i]));
  }

  printf("Usage: %s [OPTION]...\n\n", argv[0]);

  printf("General options: \n");
  printf("  -%c, --%s %*s Prints this help and exit\n", c[ARG_HELP], t[ARG_HELP], (int) (max_len-strlen(t[ARG_HELP])), "");
  printf("  -%c, --%s %*s Prints peakperf version and exit\n", c[ARG_VERSION], t[ARG_VERSION], (int) (max_len-strlen(t[ARG_VERSION])), "");
  printf("  -%c, --%s %*s List the avaiable benchmark types\n", c[ARG_LISTBENCHS], t[ARG_LISTBENCHS], (int) (max_len-strlen(t[ARG_LISTBENCHS])), "");
  printf("  -%c, --%s %*s Select a specific benchmark to run\n", c[ARG_BENCHMARK], t[ARG_BENCHMARK], (int) (max_len-strlen(t[ARG_BENCHMARK])), "");
  printf("  -%c, --%s %*s Set the number of trials of the benchmark\n", c[ARG_TRIALS], t[ARG_TRIALS], (int) (max_len-strlen(t[ARG_TRIALS])), "");
  printf("  -%c, --%s %*s Set the number of warmup trials\n", c[ARG_WARMUP], t[ARG_WARMUP], (int) (max_len-strlen(t[ARG_WARMUP])), "");
  printf("  -%c, --%s %*s Select the device to run the benchmark on. Possible values are:\n", c[ARG_DEVICE], t[ARG_DEVICE], (int) (max_len-strlen(t[ARG_DEVICE])), "");
  printf("      cpu: Run peakperf in the CPU (default)\n");
  printf("      gpu: Run peakperf in the GPU\n");
  printf("\nCPU device only options:\n");
  printf("  -%c, --%s %*s Set the number of threads to use (default: omp_get_max_threads())\n", c[ARG_CPU_THREADS], t[ARG_CPU_THREADS], (int) (max_len-strlen(t[ARG_CPU_THREADS])), "");
  printf("\nGPU device only options:\n");
  printf("  -%c, --%s %*s Set the number of CUDA blocks to use (default: number of SM in the running GPU)\n", c[ARG_GPU_BLOCKS], t[ARG_GPU_BLOCKS], (int) (max_len-strlen(t[ARG_GPU_BLOCKS])), "");
  printf("  -%c, --%s %*s Set the number of threads per block to use (default: 1024)\n", c[ARG_GPU_TPB], t[ARG_GPU_TPB], (int) (max_len-strlen(t[ARG_GPU_TPB])), "");
}

void print_version() {
  printf("peakperf v%s\n", VERSION);
}

void print_header(struct benchmark* bench, struct hardware* hw, double gflops) {
  struct config_str * cfg_str = get_cfg_str(bench);
  const char* device_type_str = get_device_type_str(bench);
  char* device_name = get_device_name_str(bench, hw);
  const char* device_uarch = get_device_uarch_str(bench, hw);
  const char* bench_name = get_benchmark_name(bench);
  int line_length = 54;

  int max_len = strlen("Iterations");
  for(int i=0; i < cfg_str->num_fields; i++) {
    max_len = max(max_len, (int) strlen(cfg_str->field_name[i]));
  }

  putchar('\n');
  for(int i=0; i < line_length; i++) putchar('-');
  printf("\n" BOLD "    peakperf (https://github.com/Dr-Noob/peakperf)" RESET "\n");
  for(int i=0; i < line_length; i++) putchar('-');
  putchar('\n');

  printf("%*s %s: %s\n",         (int) (max_len-strlen(device_type_str)),        "", device_type_str, device_name);
  printf("%*s Microarch: %s\n",  (int) (max_len-strlen("Microarch")),            "", device_uarch);
  printf("%*s Benchmark: %s\n",  (int) (max_len-strlen("Benchmark")),            "", bench_name);
  printf("%*s Iterations: %d\n", (int) (max_len-strlen("Iterations")),           "", MAXFLOPS_ITERS);
  printf("%*s GFLOP: %.2f\n",    (int) (max_len-strlen("GFLOP")),                "", gflops);
  for(int i=0; i < cfg_str->num_fields; i++) {
    printf("%*s %s: %d\n",       (int) (max_len-strlen(cfg_str->field_name[i])), "", cfg_str->field_name[i], cfg_str->field_value[i]);
  }
  putchar('\n');

  printf(BOLD "%6s %8s %8s" RESET "\n","Nº","Time(s)","GFLOP/s");
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
  char* benchmark_name = get_benchmark_str_args();

  struct benchmark* bench = init_benchmark_device(device);
  if(bench == NULL) {
    return EXIT_FAILURE;
  }
  struct config* cfg = get_config();
  bool list_benchs = list_benchmarks();
  struct hardware* hw = get_hardware_info(bench);
  if(hw == NULL) {
    return EXIT_FAILURE;
  }

  if(list_benchmarks()) {
    print_bench_types(bench, hw);
    return EXIT_SUCCESS;
  }

  struct timeval t0;
  struct timeval t1;
  bool flag = init_benchmark(bench, hw, cfg, benchmark_name);
  if(!flag) {
    return EXIT_FAILURE;
  }

  double gflops = get_gflops(bench);
  double e_time = 0;
  double mean = 0;
  double sd = 0;
  double sum = 0;
  double* gflops_list = (double*) malloc(sizeof(double) * n_trials);

  int line_length = 54;
  print_header(bench, hw, gflops);

  for (int trial = 0; trial < n_trials+n_warmup_trials; trial++) {
    gettimeofday(&t0, 0);
    if(!compute(bench)) return EXIT_FAILURE;
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

  exit_benchmark(bench);

  return EXIT_SUCCESS;
}
