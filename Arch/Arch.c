#include <stdio.h>
#include <omp.h>

#define TYPE __m256

#include "Arch.h"
#include "256_8.h"
#include "256_10.h"

void compute(struct cpu* cpu, int n_threads) {    
  if(n_threads > MAX_NUMBER_THREADS) {
    printf("ERROR: Max number of threads is %d\n", MAX_NUMBER_THREADS);
    return;
  }
  
  TYPE mult = {0};
  TYPE *farr_ptr = NULL;
  
  struct uarch* uarch_struct = get_uarch_struct(cpu);
  MICROARCH u = uarch_struct->uarch;  
  void(*compute_function)(TYPE *farr_ptr, TYPE, int) = NULL;
  
  switch(u) {
    case UARCH_SANDY_BRIDGE:
      compute_function = compute_256_10;
      break;
    case UARCH_KABY_LAKE:
      compute_function = compute_256_8;
      break;
    default:
      printf("ERROR: No valid uarch found!\n");
      return;
  }
  
  #pragma omp parallel for
    for(int t=0; t<n_threads; t++)
      compute_function(farr_ptr, mult, t);
}

double get_gflops(struct cpu* cpu, int n_threads) {
   struct uarch* uarch_struct = get_uarch_struct(cpu);
   MICROARCH u = uarch_struct->uarch;  
   double(*get_gflops_function)(int) = NULL;
   
   switch(u) {
    case UARCH_SANDY_BRIDGE:
      get_gflops_function = get_gflops_256_10;
      break;
    case UARCH_KABY_LAKE:
      get_gflops_function = get_gflops_256_8;
      break;
    default:
      printf("ERROR: No valid uarch found!\n");
      return -1.0;
  }
  
  return get_gflops_function(n_threads);
}
