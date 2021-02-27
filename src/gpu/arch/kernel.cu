#include <cuda_runtime.h>
#include "helper_cuda.h"

#include <stdio.h>
#include <string.h>

#include "../../global.hpp"
#include "../../getarg.hpp"

#include "mad_6.hpp"
#include "mad_4.hpp"

enum {
  ARCH_MAXWELL,
  ARCH_PASCAL,
  ARCH_TURING,
  ARCH_UNKNOWN
};

enum bench_types {
  BENCH_TYPE_MAXWELL,
  BENCH_TYPE_PASCAL,
  BENCH_TYPE_TURING
};

static const char *uarch_str[] = {
  /*[ARCH_MAXWELL]    = */ "Maxwell",
  /*[ARCH_PASCAL]     = */ "Pascal",
  /*[ARCH_TURING]     = */ "Turing",
};

static const char *bench_name[] = {
  /*[BENCH_TYPE_MAXWELL]    = */ "Maxwell",
  /*[BENCH_TYPE_PASCAL]     = */ "Pascal",
  /*[BENCH_TYPE_TURING]     = */ "Turing",
};

static const char *bench_types_str[] = {
  /*[BENCH_TYPE_MAXWELL]    = */ "maxwell",
  /*[BENCH_TYPE_TURING]     = */ "pascal",
  /*[BENCH_TYPE_TURING]     = */ "turing",
};

struct benchmark_gpu {
  int nbk; // Blocks per thread
  int tpb; // Threads per block
  int n;
  double gflops;
  const char* name;
  bench_type benchmark_type;
  void(*compute_function)(float *, float *, float *, int);
  float *d_A;
  float *d_B;
  float *d_C;
};

// We assume only one gpu is present...
struct gpu {
  int compute_capability;
  int sm_count;
  int cc_major;
  int cc_minor;
  char uarch;
  char* name;
};

bool select_benchmark(struct benchmark_gpu* bench) {
  bench->compute_function = NULL;
  switch(bench->benchmark_type) {
    case BENCH_TYPE_MAXWELL:
    case BENCH_TYPE_PASCAL:
      bench->compute_function = compute_mad_6;
      bench->gflops = (double)(BENCHMARK_GPU_ITERS * 2 * (long)bench->n)/(long)1000000000;
      break;
    case BENCH_TYPE_TURING:
      bench->compute_function = compute_mad_4;
      bench->gflops = (double)(BENCHMARK_GPU_ITERS * 2 * (long)bench->n * WORK_MAD_4)/(long)1000000000;
      break;
    default:
      printErr("No valid benchmark! (bench: %d)", bench->benchmark_type);
      return false;
  }

  bench->name = bench_name[bench->benchmark_type];
  return true;
}

bench_type parse_benchmark_gpu(char* str) {
  int len = sizeof(bench_types_str) / sizeof(bench_types_str[0]);
  for(bench_type t = 0; t < len; t++) {
    if(strcmp(str, bench_types_str[t]) == 0) {
      return t;
    }
  }
  return BENCH_TYPE_INVALID;
}

void print_cuda_gpus_list() {
  cudaError_t err = cudaSuccess;
  int num_gpus = -1;
  if ((err = cudaGetDeviceCount(&num_gpus)) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    return;
  }
  printf("GPUs available: %d\n", num_gpus);

  if(num_gpus > 0) {
    cudaDeviceProp deviceProp;
    int max_len = 0;

    for(int idx=0; idx < num_gpus; idx++) {
      cudaGetDeviceProperties(&deviceProp, idx);
      max_len = max(max_len, (int) strlen(deviceProp.name));
    }

    for(int i=0; i < max_len + 28; i++) putchar('-');
    putchar('\n');
    for(int idx=0; idx < num_gpus; idx++) {
      cudaGetDeviceProperties(&deviceProp, idx);
      printf("%d: %s (Compute Capability %d.%d)\n", idx, deviceProp.name, deviceProp.major, deviceProp.minor);
    }
  }
}

struct gpu* get_gpu_info(int gpu_idx) {
  cudaError_t err = cudaSuccess;
  struct gpu* gpu = (struct gpu *) malloc(sizeof(struct gpu));

  int num_gpus = -1;
  if ((err = cudaGetDeviceCount(&num_gpus)) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }
  if(num_gpus <= 0) {
    printErr("No CUDA capable devices found!");
    return NULL;
  }
  if(gpu_idx < 0) {
    printErr("GPU index must be equal or greater than zero");
    return NULL;
  }
  if(gpu_idx+1 > num_gpus) {
    printErr("Requested GPU index %d in a system with %d GPUs", gpu_idx, num_gpus);
    return NULL;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpu_idx);

  int gpu_name_len = strlen(deviceProp.name);
  gpu->cc_major = deviceProp.major;
  gpu->cc_minor = deviceProp.minor;
  gpu->compute_capability = deviceProp.major * 10 + deviceProp.minor;
  gpu->sm_count = deviceProp.multiProcessorCount;
  gpu->name = (char *) malloc(sizeof(char) * (gpu_name_len + 1));
  memset(gpu->name, 0, gpu_name_len + 1);
  strncpy(gpu->name, deviceProp.name, gpu_name_len);

  switch(gpu->compute_capability) {
    case 52:
      gpu->uarch = ARCH_MAXWELL;
      break;
    case 61:
      gpu->uarch = ARCH_PASCAL;
      break;
    case 75:
      gpu->uarch = ARCH_TURING;
      break;
    default:
      printf("GPU: %s\n", gpu->name);
      printf("Invalid uarch: %d.%d\n", deviceProp.major, deviceProp.minor);
      return NULL;
  }

  return gpu;
}

struct benchmark_gpu* init_benchmark_gpu(struct gpu* gpu, int nbk, int tpb, char* bench_type_str) {
  struct benchmark_gpu* bench = (struct benchmark_gpu *) malloc(sizeof(struct benchmark_gpu));
  bench_type benchmark_type;

  if(bench_type_str == NULL) {
    benchmark_type = BENCH_TYPE_INVALID;
  }
  else {
   benchmark_type = parse_benchmark_gpu(bench_type_str);
   if(benchmark_type == BENCH_TYPE_INVALID) {
     printErr("Invalid GPU benchmark specified: '%s'", bench_type_str);
     return NULL;
   }
  }

  bench->nbk = (nbk == INVALID_CFG) ? gpu->sm_count : nbk;
  bench->tpb = (tpb == INVALID_CFG) ? (6 * _ConvertSMVer2Cores(gpu->cc_major, gpu->cc_minor)): tpb;
  bench->n = gpu->sm_count * bench->tpb;

  // Manual benchmark select
  if(benchmark_type != BENCH_TYPE_INVALID) {
    bench->benchmark_type = benchmark_type;
  }
  else {  // Automatic benchmark select
    switch(gpu->uarch) {
      case ARCH_MAXWELL:
        bench->benchmark_type = BENCH_TYPE_MAXWELL;
        break;
      case ARCH_PASCAL:
        bench->benchmark_type = BENCH_TYPE_PASCAL;
        break;
      case ARCH_TURING:
        bench->benchmark_type = BENCH_TYPE_TURING;
        break;
      default:
        return NULL;
    }
  }

  if(!select_benchmark(bench))
    return NULL;

  cudaError_t err = cudaSuccess;
  float *h_A;
  float *h_B;
  int size = bench->n * sizeof(float);

  if ((err = cudaMallocHost((void **)&h_A, size)) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  if ((err = cudaMallocHost((void **)&h_B, size)) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  for (int i = 0; i < bench->n; i++) {
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
  }

  if ((err = cudaMalloc((void **) &(bench->d_A), size)) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  if ((err = cudaMalloc((void **) &(bench->d_B), size)) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  if ((err = cudaMalloc((void **) &(bench->d_C), size)) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  if ((err = cudaMemcpy(bench->d_A, h_A, size, cudaMemcpyHostToDevice)) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  if ((err = cudaMemcpy(bench->d_B, h_B, size, cudaMemcpyHostToDevice)) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  return bench;
}

void print_bench_types_gpu(struct gpu* gpu) {
  int len = sizeof(bench_types_str) / sizeof(bench_types_str[0]);
  long unsigned int longest = 0;
  long unsigned int total_length = 0;
  for(bench_type t = 0; t < len; t++) {
    if(strlen(bench_name[t]) > longest) {
      longest = strlen(bench_name[t]);
      total_length = longest + 16 + strlen(bench_types_str[t]);
    }
  }

  printf("Available benchmark types for GPU:\n");
  for(long unsigned i=0; i < total_length; i++) putchar('-');
  putchar('\n');
  for(bench_type t = 0; t < len; t++) {
    printf("  - %s %*s(Keyword: %s)\n", bench_name[t], (int) (strlen(bench_name[t]) - longest), "", bench_types_str[t]);
  }
}

const char* get_benchmark_name_gpu(struct benchmark_gpu* bench) {
  return bench->name;
}

double get_gflops_gpu(struct benchmark_gpu* bench) {
  return bench->gflops;
}

bool compute_gpu(struct benchmark_gpu* bench) {
  cudaError_t err = cudaSuccess;
  dim3 dimGrid(bench->nbk, 1, 1);
  dim3 dimBlock(bench->tpb, 1, 1);

  bench->compute_function<<<dimGrid, dimBlock>>>(bench->d_A, bench->d_B, bench->d_C, bench->n);

  cudaDeviceSynchronize();

  if ((err = cudaGetLastError()) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    if(err == cudaErrorLaunchTimeout) {
      printf("         NOTE: The GPU used by peakperf is attached to a display.\n");
      printf("         A possible workaround is to stop X server by issuing:\n");
      printf("         sudo systemctl isolate multi-user.target\n");
    }
    return false;
  }
  return true;
}

void exit_benchmark_gpu() {
  cudaDeviceReset();
}

char* get_str_gpu_name(struct gpu* gpu) {
  return gpu->name;
}

const char* get_str_gpu_uarch(struct gpu* gpu) {
  return uarch_str[gpu->uarch];
}

int get_n_blocks(struct benchmark_gpu* bench) {
  return bench->nbk;
}

int get_threads_per_block(struct benchmark_gpu* bench) {
  return bench->tpb;
}

