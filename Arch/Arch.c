#include <stdio.h>
#include <omp.h>

#define TYPE __m256

#include "Arch.h"
#include "256_8.h"
#include "256_10.h"

struct benchmark {
  int n_threads;
  double gflops;
  char* name;
  void(*compute_function)(TYPE *farr_ptr, TYPE, int);
};

enum {
  BENCH_256_3_NOFMA,  
  BENCH_256_5,
  BENCH_256_8,
  BENCH_256_10,
  BENCH_512_8,
  BENCH_512_12,
};

static char *bench_name[] = {
  [UARCH_SANDY_BRIDGE]   = "Sandy Bridge - 256 bits",
  [UARCH_KABY_LAKE]      = "Kaby Lake - 256 bits",
};

double compute_gflops(int n_threads, char bench) {
  int fma_available;
  int op_per_it;
  int bytes_in_vect;
  
  switch(bench) {
    case BENCH_256_3_NOFMA:
      fma_available = B_256_3_NOFMA_FMA_AV;
      op_per_it = B_256_3_NOFMA_OP_IT;
      bytes_in_vect = B_256_3_NOFMA_BYTES;
      break;
    case BENCH_256_5:
      fma_available = B_256_5_FMA_AV;
      op_per_it = B_256_5_OP_IT;
      bytes_in_vect = B_256_5_BYTES;
      break;  
    case BENCH_256_8:
      fma_available = B_256_8_FMA_AV;
      op_per_it = B_256_8_OP_IT;
      bytes_in_vect = B_256_8_BYTES;
      break;
    case BENCH_256_10:
      fma_available = B_256_10_FMA_AV;
      op_per_it = B_256_10_OP_IT;
      bytes_in_vect = B_256_10_BYTES;
      break;
    case BENCH_512_8:
      fma_available = B_512_8_FMA_AV;
      op_per_it = B_512_8_OP_IT;
      bytes_in_vect = B_512_8_BYTES;
      break;
    case BENCH_512_12:
      fma_available = B_512_12_FMA_AV;
      op_per_it = B_512_12_OP_IT;
      bytes_in_vect = B_512_12_BYTES;
      break;  
    default:
      printf("ERROR: Invalid benchmark type!\n");
      return -1.0;
  }
  
  return (double)((long)n_threads*MAXFLOPS_ITERS*op_per_it*(bytes_in_vect/4)*fma_available)/1000000000;        
}

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
      bench->gflops = compute_gflops(n_threads, BENCH_256_10);
      bench->name = bench_name[UARCH_SANDY_BRIDGE];
      break;
    case UARCH_KABY_LAKE:
      bench->compute_function = compute_256_8;
      bench->gflops = compute_gflops(n_threads, BENCH_256_8);
      bench->name = bench_name[UARCH_KABY_LAKE];
      break;
    default:
      printf("ERROR: No valid uarch found!\n");
      return NULL;
  }
  
  if(bench->gflops < 0.0) return NULL;
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

char* get_benchmark_name(struct benchmark* bench) {
  return bench->name;    
}
