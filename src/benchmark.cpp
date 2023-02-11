#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "benchmark.hpp"
#include "global.hpp"

#ifdef DEVICE_CPU_ENABLED
  #include "cpu/cpufetch/cpufetch.hpp"
  #include "cpu/cpufetch/uarch.hpp"
  #include "cpu/arch/arch.hpp"
#endif

#ifdef DEVICE_GPU_ENABLED
  #include "gpu/arch.hpp"
#endif

#define RED   "\x1b[31;1m"
#define BOLD  "\x1b[1m"
#define GREEN "\x1b[42m"
#define RESET "\x1b[0m"

static const char *device_str[] = {
  /*[DEVICE_TYPE_CPU] = */ "CPU",
  /*[DEVICE_TYPE_CPU] = */ "GPU",
};

#define CFG_STR_CPU_1 "Threads"
#define CFG_STR_GPU_1 "Blocks"
#define CFG_STR_GPU_2 "Threads/block"

struct benchmark* init_benchmark_device(device_type device) {
  struct benchmark* bench = (struct benchmark *) malloc(sizeof(struct benchmark));
  bench->device = device;

  if(device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
      return bench;
    #else
      printErr("peakperf was not built with CPU support");
      return NULL;
    #endif
  }
  else if (device == DEVICE_TYPE_GPU) {
    #ifdef DEVICE_GPU_ENABLED
      return bench;
    #else
      printErr("peakperf was not built with GPU support");
      return NULL;
    #endif
  }
  else {
    printf("Invalid device found: %d", device);
    return NULL;
  }
}

struct hardware* get_hardware_info(struct benchmark* bench, struct config* cfg) {
  struct hardware* hw = (struct hardware *) malloc(sizeof(struct hardware));

  if(bench->device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
      if((hw->cpu = get_cpu_info()) == NULL) {
        return NULL;
      }
    #endif
  }
  else if (bench->device == DEVICE_TYPE_GPU) {
    #ifdef DEVICE_GPU_ENABLED
      if((hw->gpu = get_gpu_info(cfg->gpu_idx)) == NULL) {
        return NULL;
      }
    #endif
  }

  return hw;
}

bool init_benchmark(struct benchmark* bench, struct hardware* hw, struct config* cfg, char* bench_type_str) {
  if(bench->device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
      bench->cpu_bench = init_benchmark_cpu(hw->cpu, cfg->n_threads, bench_type_str, use_pcores_only());

      if(bench->cpu_bench == NULL)
        return false;
      return true;
    #endif
    return false;
  }
  else if(bench->device == DEVICE_TYPE_GPU) {
    #ifdef DEVICE_GPU_ENABLED
      bench->gpu_bench = init_benchmark_gpu(hw->gpu, cfg->nbk, cfg->tpb);

      if(bench->gpu_bench == NULL)
        return false;
      return true;
    #endif
  }
  return false;
}

bool compute(struct benchmark* bench, double* e_time) {
  if(bench->device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
    return compute_cpu(bench->cpu_bench, e_time);
    #endif
  }
  else if(bench->device == DEVICE_TYPE_GPU) {
    #ifdef DEVICE_GPU_ENABLED
    return compute_gpu(bench->gpu_bench, e_time);
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
  return NULL;
}

const char* get_hybrid_topology_string(struct benchmark* bench) {
  if(bench->device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
      return get_hybrid_topology_string_cpu(bench->cpu_bench);
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

char* get_device_name_str(struct benchmark* bench, struct hardware* hw) {
  if(bench->device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
      return get_str_cpu_name(hw->cpu);
    #endif
  }
  else if(bench->device == DEVICE_TYPE_GPU) {
    #ifdef DEVICE_GPU_ENABLED
      return get_str_gpu_name(hw->gpu);
    #endif
  }
  return NULL;
}

const char* get_device_uarch_str(struct benchmark* bench, struct hardware* hw) {
  if(bench->device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
      return get_str_uarch(hw->cpu);
    #endif
  }
  else if(bench->device == DEVICE_TYPE_GPU) {
    #ifdef DEVICE_GPU_ENABLED
      return get_str_gpu_uarch(hw->gpu);
    #endif
  }
  return NULL;
}

const char *get_device_type_str(struct benchmark* bench) {
  return device_str[(int) bench->device];
}

struct config_str * get_cfg_str(struct benchmark* bench) {
  struct config_str * cfg_str = (struct config_str *) malloc(sizeof(struct config_str));
  if(bench->device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
      cfg_str->num_fields = 1;
      cfg_str->field_value = (int *) malloc(sizeof(int) * cfg_str->num_fields);
      cfg_str->field_name = (char **) malloc(sizeof(char *) * cfg_str->num_fields);

      cfg_str->field_value[0] = get_n_threads(bench->cpu_bench);
      cfg_str->field_name[0] = (char *) malloc(sizeof(char) * (strlen(CFG_STR_CPU_1) + 1));
      strncpy(cfg_str->field_name[0], CFG_STR_CPU_1, strlen(CFG_STR_CPU_1) + 1);

      return cfg_str;
    #endif
  }
  else if(bench->device == DEVICE_TYPE_GPU) {
    #ifdef DEVICE_GPU_ENABLED
      cfg_str->num_fields = 2;
      cfg_str->field_value = (int *) malloc(sizeof(int) * cfg_str->num_fields);
      cfg_str->field_name = (char **) malloc(sizeof(char *) * cfg_str->num_fields);

      cfg_str->field_value[0] = get_n_blocks(bench->gpu_bench);
      cfg_str->field_name[0] = (char *) malloc(sizeof(char) * (strlen(CFG_STR_GPU_1) + 1));
      strncpy(cfg_str->field_name[0], CFG_STR_GPU_1, strlen(CFG_STR_GPU_1) + 1);

      cfg_str->field_value[1] = get_threads_per_block(bench->gpu_bench);
      cfg_str->field_name[1] = (char *) malloc(sizeof(char) * (strlen(CFG_STR_GPU_2) + 1));
      strncpy(cfg_str->field_name[1], CFG_STR_GPU_2, strlen(CFG_STR_GPU_2) + 1);

      return cfg_str;
    #endif
  }
  return NULL;
}

void print_bench_types(struct benchmark* bench, struct hardware* hw) {
  if(bench->device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
      print_bench_types_cpu(hw->cpu);
    #endif
  }
}

int print_gpus_list(struct benchmark* bench) {
  if(bench->device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
      return EXIT_FAILURE;
    #endif
  }
  else if(bench->device == DEVICE_TYPE_GPU) {
    #ifdef DEVICE_GPU_ENABLED
      print_cuda_gpus_list();
      return EXIT_SUCCESS;
    #endif
  }
  return EXIT_FAILURE;
}

long get_benchmark_iterations(struct benchmark* bench) {
  if(bench->device == DEVICE_TYPE_CPU) {
    #ifdef DEVICE_CPU_ENABLED
      return BENCHMARK_CPU_ITERS;
    #endif
  }
  else if(bench->device == DEVICE_TYPE_GPU) {
    #ifdef DEVICE_GPU_ENABLED
      return BENCHMARK_GPU_ITERS;
    #endif
  }
  return -1;
}
