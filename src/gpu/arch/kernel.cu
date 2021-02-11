#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#include "maxwell.hpp"

// Avoid ArchLinux package warning "WARNING: Package contains reference to $srcdir"
#define __FILENAME__ "kernel.cu"

enum {
  ARCH_MAXWELL,
  ARCH_UNKNOWN
};

static const char *uarch_str[] = {
  /*[ARCH_MAXWELL]    = */ "Maxwell",
};

struct benchmark_gpu {
  int nbk; // Blocks per thread
  int tpb; // Threads per block
  int n;
  double gflops;
  void(*compute_function)(float *, float *, float *, int);
  float *d_A;
  float *d_B;
  float *d_C;
};

// We assume only one gpu is present...
struct gpu {
  int compute_capability;
  int sm_count;
  char uarch;
  char* name;
};

struct gpu* get_gpu_info() {
  struct gpu* gpu = (struct gpu *) malloc(sizeof(struct gpu));

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  int gpu_name_len = strlen(deviceProp.name);
  gpu->compute_capability = deviceProp.major * 10 + deviceProp.minor;
  gpu->sm_count = deviceProp.multiProcessorCount;
  gpu->name = (char *) malloc(sizeof(char) * (gpu_name_len + 1));
  memset(gpu->name, 0, gpu_name_len + 1);
  strncpy(gpu->name, deviceProp.name, gpu_name_len);

  switch(gpu->compute_capability) {
    case 52:
      gpu->uarch = ARCH_MAXWELL;
      break;
    default:
      printf("Invalid uarch found: %d.%d\n", deviceProp.major, deviceProp.minor);
      return NULL;
  }

  return gpu;
}

struct benchmark_gpu* init_benchmark_gpu(struct gpu* gpu, int nbk, int tpb) {
  struct benchmark_gpu* bench = (struct benchmark_gpu *) malloc(sizeof(struct benchmark_gpu));

  // TODO: Dont ignore nbk, tpb
  bench->nbk = gpu->sm_count;
  bench->tpb = 1024;
  bench->n = gpu->sm_count * bench->tpb;

  switch(gpu->uarch) {
    case ARCH_MAXWELL:
      bench->compute_function = matrixMul_maxwell;
      bench->gflops = (double)(KERNEL_ITERS * 2 * (long)bench->n * WORK_MAXWELL)/(long)1000000000;
      break;
    default:
      return NULL;
  }

  cudaError_t err = cudaSuccess;
  float *h_A;
  float *h_B;
  int size = bench->n * sizeof(float);

  if ((err = cudaMallocHost((void **)&h_A, size)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILENAME__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  if ((err = cudaMallocHost((void **)&h_B, size)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILENAME__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  for (int i = 0; i < bench->n; i++) {
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
  }

  if ((err = cudaMalloc((void **) &(bench->d_A), size)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILENAME__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  if ((err = cudaMalloc((void **) &(bench->d_B), size)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILENAME__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  if ((err = cudaMalloc((void **) &(bench->d_C), size)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILENAME__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  if ((err = cudaMemcpy(bench->d_A, h_A, size, cudaMemcpyHostToDevice)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILENAME__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  if ((err = cudaMemcpy(bench->d_B, h_B, size, cudaMemcpyHostToDevice)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILENAME__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  return bench;
}

const char* get_benchmark_name_gpu(struct benchmark_gpu* bench) {
  return uarch_str[0];
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
    printf("[%s:%d]%s: %s\n", __FILENAME__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
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

