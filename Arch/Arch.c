#include <stdio.h>
#include <omp.h>

#define TYPE __m256

#include "Arch.h"
#include "256_8.h"
#include "256_10.h"

struct benchmark {
  int n_threads;
  double gflops;
  void(*compute_function)(TYPE *farr_ptr, TYPE, int);
};

struct benchmark* init_benchmark(struct cpu* cpu, int n_threads) {    
  struct benchmark* bench = malloc(sizeof(struct benchmark));
  
  if(n_threads > MAX_NUMBER_THREADS) {
    printf("ERROR: Max number of threads is %d\n", MAX_NUMBER_THREADS);
    return NULL;
  }
  
  struct uarch* uarch_struct = get_uarch_struct(cpu);
  MICROARCH u = uarch_struct->uarch;  
  bench->n_threads = n_threads;
  
  switch(u) {
    case UARCH_SANDY_BRIDGE:
      bench->compute_function = compute_256_10;
      bench->gflops = get_gflops_256_10(n_threads);
      break;
    case UARCH_KABY_LAKE:
      bench->compute_function = compute_256_8;
      bench->gflops = get_gflops_256_8(n_threads);
      break;
    default:
      printf("ERROR: No valid uarch found!\n");
      return NULL;
  }
  
  return bench;
}

void compute(struct benchmark* bench) {      
  TYPE mult = {0};
  TYPE *farr_ptr = NULL;
  
  #pragma omp parallel for
  for(int t=0; t < bench->n_threads; t++)
    bench->compute_function(farr_ptr, mult, t);
}

double get_gflops(struct benchmark* bench) {
  return bench->gflops;    
}
