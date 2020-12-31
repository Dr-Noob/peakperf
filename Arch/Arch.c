#include <stdio.h>
#include <omp.h>

#define TYPE __m256

#include "Arch.h"

#include "sandy_bridge.h"  
#include "ivy_bridge.h"
#include "haswell.h" 
#include "skylake_256.h"
#include "skylake_512.h"
#include "broadwell.h"
#include "kaby_lake.h"
#include "coffe_lake.h"
#include "cannon_lake_256.h"  
#include "cannon_lake_512.h"
#include "ice_lake_256.h"
#include "ice_lake_512.h"
#include "knl.h"
#include "zen.h"
#include "zen_plus.h"

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
      bench->compute_function = compute_sandy_bridge;
      bench->gflops = compute_gflops(n_threads, BENCH_256_3_NOFMA);
      break;  
    case UARCH_IVY_BRIDGE:
      bench->compute_function = compute_ivy_bridge;
      bench->gflops = compute_gflops(n_threads, BENCH_256_3_NOFMA);
      break;
    case UARCH_HASWELL:
      bench->compute_function = compute_haswell;
      bench->gflops = compute_gflops(n_threads, BENCH_256_10);
      break;  
    case UARCH_SKYLAKE_256:
      bench->compute_function = compute_skylake_256;
      bench->gflops = compute_gflops(n_threads, BENCH_256_8);
      break;  
    case UARCH_SKYLAKE_512:
      bench->compute_function = compute_skylake_512;
      bench->gflops = compute_gflops(n_threads, BENCH_512_8);
      break;  
    case UARCH_BROADWELL:
      bench->compute_function = compute_broadwell;
      bench->gflops = compute_gflops(n_threads, BENCH_256_8);
      break;  
    case UARCH_KABY_LAKE:
      bench->compute_function = compute_kaby_lake;
      bench->gflops = compute_gflops(n_threads, BENCH_256_8);
      break;  
    case UARCH_COFFE_LAKE:
      bench->compute_function = compute_coffe_lake;
      bench->gflops = compute_gflops(n_threads, BENCH_256_8);
      break;  
    case UARCH_CANNON_LAKE_256:
      bench->compute_function = compute_cannon_lake_256;
      bench->gflops = compute_gflops(n_threads, BENCH_256_8);
      break;  
    case UARCH_CANNON_LAKE_512:
      bench->compute_function = compute_cannon_lake_512;
      bench->gflops = compute_gflops(n_threads, BENCH_256_8);
      break;        
    case UARCH_ICE_LAKE_256:
      bench->compute_function = compute_ice_lake_256;
      bench->gflops = compute_gflops(n_threads, BENCH_256_8);
      break; 
    case UARCH_ICE_LAKE_512:
      bench->compute_function = compute_ice_lake_512;
      bench->gflops = compute_gflops(n_threads, BENCH_256_10);
      break;       
    case UARCH_KNIGHTS_LANDING:
      bench->compute_function = compute_knl;
      bench->gflops = compute_gflops(n_threads, BENCH_512_12);
      break;
    case UARCH_ZEN:
      bench->compute_function = compute_zen;
      bench->gflops = compute_gflops(n_threads, BENCH_256_5);
      break;
    case UARCH_ZEN_PLUS:
      bench->compute_function = compute_zen_plus;
      bench->gflops = compute_gflops(n_threads, BENCH_256_5);
      break;    
    default:
      printf("ERROR: No valid uarch found!\n");
      return NULL;
  }
  bench->name = bench_name[UARCH_SANDY_BRIDGE];
  
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
