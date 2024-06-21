#include "kernel.hpp"
#include <stdio.h>
#include <stdint.h>
#define N 16
#define gid threadIdx.x + blockIdx.x * blockDim.x
#define off gid*N


__global__
void compute_kernel(float *vec_a, float *vec_b, float *vec_c, int n) {
  __shared__ float myblockA[N];
  __shared__ float myblockB[N];
  __shared__ float myblockC[N];

  #pragma unroll
  for(int i = 0; i < N; i++){
   myblockA[i] = vec_a[off+i];
   myblockB[i] = vec_b[off+i];
   myblockC[i] = vec_a[off+i];
  }

  __syncthreads();

  #pragma unroll 32
  for(long i=0; i < BENCHMARK_GPU_ITERS; i++) {
    #pragma unroll
    for(int j = 0; j < N; j++){
      myblockC[j] = (myblockC[j] * myblockA[j]) + myblockB[j];
    }
  }

  #pragma unroll
  for(int i = 0; i < N; i++){
    vec_c[off+i] = myblockC[i];
  }

}

