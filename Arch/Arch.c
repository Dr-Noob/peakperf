#include <stdio.h>
#include <omp.h>

#define TYPE __m256

#include "Arch.h"
#include "256_8.h"
#include "256_10.h"

void compute(struct cpu* cpu, int n_threads) {    
  TYPE mult = {0};
  TYPE *farr_ptr = NULL;
  
  struct uarch* uarch_struct = get_uarch_struct(cpu);
  MICROARCH u = uarch_struct->uarch;  
  void(*compute_function)(TYPE *farr_ptr, TYPE, int) = NULL;
  
  switch(u) {
    case UARCH_SANDY_BRIDGE:
      printf("Using UARCH_SANDY_BRIDGE\n");
      compute_function = compute_256_10;
      break;
    case UARCH_KABY_LAKE:
      printf("Using UARCH_KABY_LAKE\n");
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

double get_gflops() {
   // double gflops = (double)((long)n_threads*MAXFLOPS_ITERS*OP_PER_IT*(BYTES_IN_VECT/4)*FMA_AVAILABLE)/1000000000;    
   return 10000000.0;
}
