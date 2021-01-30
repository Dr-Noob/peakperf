#include <stdint.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>
//#include "helper_cuda.h"
#include "gpu.h"

int peakperf_gpu(int n_trials, int n_warmup_trials) {
  cudaError_t err = cudaSuccess;
  struct timeval t1, t2;

  int n = 10;
  int size = n * sizeof(float);

  float *h_A;
  float *h_B;
  float *h_C;

  float *d_A;
  float *d_B;
  float *d_C;

  if ((err = cudaMallocHost((void **)&h_A, size)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return EXIT_FAILURE;
  }

  if ((err = cudaMallocHost((void **)&h_B, size)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return EXIT_FAILURE;
  }

   if ((err = cudaMallocHost((void **)&h_C, size)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return EXIT_FAILURE;
  }

  for (int i = 0; i < n; i++) {
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
  }

  if ((err = cudaMalloc((void **) &d_A, size)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return EXIT_FAILURE;
  }

  if ((err = cudaMalloc((void **) &d_B, size)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return EXIT_FAILURE;
  }

  if ((err = cudaMalloc((void **) &d_C, size)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return EXIT_FAILURE;
  }

  if ((err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return EXIT_FAILURE;
  }

  if ((err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return EXIT_FAILURE;
  }

  // TODO

  cudaDeviceReset();

  return EXIT_SUCCESS;
}
