#include <cuda_runtime.h>
#include "helper_cuda.h"

#include <stdio.h>
#include <string.h>

#include "../global.hpp"
#include "../getarg.hpp"

#include "kernel.hpp"

enum {
  ARCH_FERMI,
  ARCH_KEPLER,
  ARCH_MAXWELL,
  ARCH_PASCAL,
  ARCH_VOLTA,
  ARCH_TURING,
  ARCH_AMPERE,
  ARCH_UNKNOWN
};

static const char *uarch_str[] = {
  /*[ARCH_FERMI]      = */ "Fermi",
  /*[ARCH_KEPLER]     = */ "Kepler",
  /*[ARCH_MAXWELL]    = */ "Maxwell",
  /*[ARCH_PASCAL]     = */ "Pascal",
  /*[ARCH_VOLTA]      = */ "Volta",
  /*[ARCH_TURING]     = */ "Turing",
  /*[ARCH_AMPERE]     = */ "Ampere",
};

struct benchmark_gpu {
  int nbk; // Blocks per thread
  int tpb; // Threads per block
  int n;
  double gflops;
  const char* name;
  float *d_A;
  float *d_B;
  float *d_C;
};

// We assume only one gpu is present...
struct gpu {
  int compute_capability;
  int latency;
  int sm_count;
  int cc_major;
  int cc_minor;
  char uarch;
  char* name;
};

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
  strcpy(gpu->name, deviceProp.name);

  // https://en.wikipedia.org/w/index.php?title=CUDA#GPUs_supported
  switch(gpu->compute_capability) {
    case 20:
    case 21:
      gpu->uarch = ARCH_FERMI;
      break;
    case 30:
    case 32:
    case 35:
    case 37:
      gpu->uarch = ARCH_KEPLER;
      break;
    case 50:
    case 52:
    case 53:
      gpu->uarch = ARCH_MAXWELL;
      break;
    case 60:
    case 61:
    case 62:
      gpu->uarch = ARCH_PASCAL;
      break;
    case 70:
    case 72:
      gpu->uarch = ARCH_VOLTA;
      break;
    case 75:
      gpu->uarch = ARCH_TURING;
      break;
    case 80:
    case 86:
      gpu->uarch = ARCH_AMPERE;
      break;
    default:
      printf("GPU: %s\n", gpu->name);
      printErr("Invalid uarch: %d.%d\n", deviceProp.major, deviceProp.minor);
      return NULL;
  }

  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions (?)
  switch(gpu->uarch) {
    case ARCH_FERMI:      // UNTESTED
    case ARCH_KEPLER:     // UNTESTED
    case ARCH_MAXWELL:
    case ARCH_PASCAL:
    case ARCH_VOLTA:      // UNTESTED
      gpu->latency = 6;
      break;
    case ARCH_TURING:
    case ARCH_AMPERE:     // UNTESTED
      gpu->latency = 4;
      break;
    default:
      printErr("latency unknown for uarch: %d.%d\n", deviceProp.major, deviceProp.minor);
      return NULL;
  }

  return gpu;
}

struct benchmark_gpu* init_benchmark_gpu(struct gpu* gpu, int nbk, int tpb) {
  struct benchmark_gpu* bench = (struct benchmark_gpu *) malloc(sizeof(struct benchmark_gpu));

  // TODO: Warn if nbk or tpb are not optimal values
  if(gpu->compute_capability >= 50) {
    bench->nbk = (nbk == INVALID_CFG) ? gpu->sm_count : nbk;
    bench->tpb = (tpb == INVALID_CFG) ? (gpu->latency * _ConvertSMVer2Cores(gpu->cc_major, gpu->cc_minor)): tpb;
  }
  else {
    // Fix for old architectures where too many tpb were launched (this config is supposed to keep the same performance)
    bench->nbk = (nbk == INVALID_CFG) ? (gpu->latency * gpu->sm_count) : nbk;
    bench->tpb = (tpb == INVALID_CFG) ? _ConvertSMVer2Cores(gpu->cc_major, gpu->cc_minor) : tpb;
  }
  bench->n = bench->nbk * bench->tpb;
  bench->gflops = (double)(BENCHMARK_GPU_ITERS * 2 * (long)bench->n)/(long)1000000000;

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

double get_gflops_gpu(struct benchmark_gpu* bench) {
  return bench->gflops;
}

bool compute_gpu(struct benchmark_gpu* bench, double* e_time) {
  cudaError_t err = cudaSuccess;
  cudaEvent_t start;
  cudaEvent_t stop;
  dim3 dimGrid(bench->nbk, 1, 1);
  dim3 dimBlock(bench->tpb, 1, 1);

  cudaDeviceSynchronize();

  if ((err = cudaEventCreate(&start)) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    return false;
  }
  if ((err = cudaEventCreate(&stop)) != cudaSuccess) {
    printErr("%s: %s", cudaGetErrorName(err), cudaGetErrorString(err));
    return false;
  }

  cudaEventRecord(start, 0);
  compute_kernel<<<dimGrid, dimBlock>>>(bench->d_A, bench->d_B, bench->d_C, bench->n);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float e_time_gpu;
  cudaEventElapsedTime(&e_time_gpu, start, stop);
  *e_time = e_time_gpu/1000;

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
