#include <cuda_runtime.h>
#include <stdio.h>

#include "maxwell.h"

enum {
  ARCH_MAXWELL,
  ARCH_UNKNOWN
};

struct benchmark {
  int nbk; // Number of blocks
  int tpb; // Threads per block
  int n;
  double tflops;
  int compute_capability;
  void(*compute_function)(float *, float *, float *, int);
  char arch;
};

struct benchmark* init_benchmark() {
  struct benchmark* bench = (struct benchmark *) malloc(sizeof(struct benchmark));

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
      bench->tflops = (double)(KERNEL_ITERS * 2 * (long)bench->n * WORK_MAXWELL)/(long)1000000000000;
      break;
    default:
      return NULL;
  }

  return bench;
}

void print_benchmark(struct benchmark* bench) {
  printf("  - Architecture: %d\n", bench->arch);
  printf("  - Number of blocks: %d\n", bench->nbk);
  printf("  - Threads per block: %d\n", bench->tpb);
}

int get_n(struct benchmark* bench) {
  return bench->n;
}


double get_tflops(struct benchmark* bench) {
  return bench->tflops;
}

void compute(struct benchmark* bench, float *a, float *b, float *c, int n) {
  dim3 dimGrid(bench->nbk, 1, 1);
  dim3 dimBlock(bench->tpb, 1, 1);

  bench->compute_function<<<dimGrid, dimBlock>>>(a, b, c, n);
}
