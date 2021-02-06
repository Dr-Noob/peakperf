#include <cuda_runtime.h>
#include <stdio.h>

#include "maxwell.h"

enum {
  ARCH_MAXWELL,
  ARCH_UNKNOWN
};

struct benchmark_gpu {
  int nbk; // Number of blocks
  int tpb; // Threads per block
  int n;
  double gflops;
  int compute_capability;
  void(*compute_function)(float *, float *, float *, int);
  char arch;
  float *d_A;
  float *d_B;
  float *d_C;
};

struct benchmark_gpu* init_benchmark_gpu() {
  struct benchmark_gpu* bench = (struct benchmark_gpu *) malloc(sizeof(struct benchmark_gpu));

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  bench->compute_capability = deviceProp.major * 10 + deviceProp.minor;
  bench->nbk = deviceProp.multiProcessorCount;
  bench->tpb = 1024;
  bench->n = bench->nbk * bench->tpb;

  switch(bench->compute_capability) {
    case 52:
      bench->arch = ARCH_MAXWELL;
      break;
    default:
      printf("Invalid arch found: %d.%d\n", deviceProp.major, deviceProp.minor);
      return NULL;
  }

  switch(bench->arch) {
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
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  if ((err = cudaMallocHost((void **)&h_B, size)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  for (int i = 0; i < bench->n; i++) {
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
  }

  if ((err = cudaMalloc((void **) &(bench->d_A), size)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  if ((err = cudaMalloc((void **) &(bench->d_B), size)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  if ((err = cudaMalloc((void **) &(bench->d_C), size)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  if ((err = cudaMemcpy(bench->d_A, h_A, size, cudaMemcpyHostToDevice)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  if ((err = cudaMemcpy(bench->d_B, h_B, size, cudaMemcpyHostToDevice)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return NULL;
  }

  return bench;
}

const char* get_benchmark_name_gpu(struct benchmark_gpu* bench) {
  char* str = (char *) malloc(sizeof(char) * 10);
  memset(str, 0, sizeof(char) * 10);
  sprintf(str, "bench_gpu");
  return str;
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
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return false;
  }
  return true;
}

void exit_benchmark_gpu() {
  cudaDeviceReset();
}
