#include <stdio.h>
#include <stdbool.h>
#include <omp.h>

#include "benchmark.h"

#ifdef DEVICE_CPU_ENABLED
  #include "cpu/cpufetch/cpufetch.h"
  #include "cpu/cpufetch/uarch.h"
  #include "cpu/arch/arch.h"
#endif

#ifdef DEVICE_GPU_ENABLED
  #include "gpu/arch/kernel.h"
#endif

#define RED   "\x1b[31;1m"
#define BOLD  "\x1b[1m"
#define GREEN "\x1b[42m"
#define RESET "\x1b[0m"

struct benchmark* init_benchmark_device(device_type device) {
  struct benchmark* bench = (struct benchmark *) malloc(sizeof(struct benchmark));
  bench->device = device;

  if(device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
      return bench;
    #else
      printf(RED "ERROR" RESET " peakperf was not built with CPU support\n");
      return NULL;
    #endif
  }
  else if (device == DEVICE_TYPE_GPU) {
    #ifdef DEVICE_GPU_ENABLED
      return bench;
    #else
      printf(RED "ERROR:" RESET " peakperf was not built with GPU support\n");
      return NULL;
    #endif
  }
  else {
    printf(RED "ERROR:" RESET " Invalid device found: %d\n", device);
    return NULL;
  }
}

bench_type get_benchmark_type(struct benchmark* bench) {
  return get_benchmark_type_args();
}

struct hardware* get_hardware_info(struct benchmark* bench) {
  struct hardware* hw = (struct hardware *) malloc(sizeof(struct hardware));

  if(bench->device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
      hw->cpu = get_cpu_info();
    #endif
  }

  return hw;
}

struct config* get_benchmark_config(struct benchmark* bench, int n_threads) {
  struct config* cfg = (struct config *) malloc(sizeof(struct config));

  if(bench->device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
    if(n_threads == INVALID_N_THREADS)
      cfg->n_threads = omp_get_max_threads();
    else
      cfg->n_threads = n_threads;
    #endif
  }

  return cfg;
}

bool init_benchmark(struct benchmark* bench, struct hardware* hw, struct config* cfg, bench_type type) {
  if(bench->device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
      bench->cpu_bench = init_benchmark_cpu(hw->cpu, cfg->n_threads, type);

      if(bench->cpu_bench == NULL)
        return false;
      return true;
    #endif
    return false;
  }
  else if(bench->device == DEVICE_TYPE_GPU) {
    #ifdef DEVICE_GPU_ENABLED
      bench->gpu_bench = init_benchmark_gpu();

      if(bench->gpu_bench == NULL)
        return false;
      return true;
    #endif
  }
  return false;
}

void print_hw_info(struct hardware* hw) {
  printf("=== hw_info ===\n");
}

void print_bench_cfg(struct config* cfg) {
  printf("=== bench_cfg ===\n");
}

bool compute(struct benchmark* bench) {
  if(bench->device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
    return compute_cpu(bench->cpu_bench);
    #endif
  }
  else if(bench->device == DEVICE_TYPE_GPU) {
    #ifdef DEVICE_GPU_ENABLED
    return compute_gpu(bench->gpu_bench);
    #endif
  }
  return false;
}

double get_gflops(struct benchmark* bench) {
  if(bench->device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
      return get_gflops_cpu(bench->cpu_bench);
    #endif
  }
  else if(bench->device == DEVICE_TYPE_GPU) {
    #ifdef DEVICE_GPU_ENABLED
      return get_gflops_gpu(bench->gpu_bench);
    #endif
  }
  return 0.0;
}

const char* get_benchmark_name(struct benchmark* bench) {
  if(bench->device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
      return get_benchmark_name_cpu(bench->cpu_bench);
    #endif
  }
  else if(bench->device == DEVICE_TYPE_GPU) {
    #ifdef DEVICE_GPU_ENABLED
      return get_benchmark_name_gpu(bench->gpu_bench);
    #endif
  }
  return NULL;
}

void exit_benchmark(struct benchmark* bench) {
  if(bench->device == DEVICE_TYPE_GPU) {
    #ifdef DEVICE_GPU_ENABLED
    exit_benchmark_gpu();
    #endif
  }
}
