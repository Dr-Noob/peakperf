#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
//#include "helper_cuda.h"
#include "arch/kernel.h"

int peakperf_gpu(int n_trials, int n_warmup_trials) {
  cudaError_t err = cudaSuccess;
  struct timeval t1, t2;

  struct benchmark* bench = init_benchmark();
  if(bench == NULL) {
    return EXIT_FAILURE;
  }

  int n = get_n(bench);
  int size = n * sizeof(float);
  int nTrials = 4;
  int nWarmupTrials = 1;
  double e_time = 0;
  double mean = 0;
  double sd = 0;
  double sum = 0;
  double tflops = get_tflops(bench);
  double tflops_list[nTrials];

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

  print_benchmark(bench);
  printf("%6s %8s %8s\n", "Nº", "Time(s)", "TFLOP/s");
  for (int trial = 0; trial < nTrials + nWarmupTrials; trial++) {
    gettimeofday(&t1, NULL);
    compute(bench, d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    if ((err = cudaGetLastError()) != cudaSuccess) {
      printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
      return EXIT_FAILURE;
    }

    e_time = (double)((t2.tv_sec-t1.tv_sec)*1000000 + t2.tv_usec-t1.tv_usec)/1000000;
    if (trial >= nWarmupTrials) {
      mean += tflops/e_time;
      tflops_list[trial-nWarmupTrials] = tflops/e_time;
      printf("%5d %8.5f %8.5f\n",trial+1, e_time, tflops/e_time);
    }
    else {
      printf("%5d %8.5f %8.5f *\n",trial+1, e_time, tflops/e_time);
    }
  }

  if ((err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost)) != cudaSuccess) {
    printf("[%s:%d]%s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
    return EXIT_FAILURE;
  }

  for(int i=0; i < n; i++) {
    if(h_A[i] + h_B[i] != h_C[i])
      fprintf(stderr, "ERROR at i=%d\n", i);
  }

  mean=mean/(double)nTrials;
  for(int i=0;i<nTrials;i++)
    sum += (tflops_list[i] - mean)*(tflops_list[i] - mean);
  sd=sqrt(sum/nTrials);

  printf("\nAverage performance: %.5f +- %.5f TFLOP/s\n", mean, sd);

  cudaDeviceReset();

  return EXIT_SUCCESS;
}
