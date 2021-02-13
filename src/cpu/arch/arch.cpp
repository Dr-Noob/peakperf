#include <stdio.h>
#include <omp.h>
#include <string.h>

#include "../../global.hpp"
#include "../../getarg.hpp"
#include "arch.hpp"

#include "sandy_bridge.hpp"
#include "ivy_bridge.hpp"
#include "haswell.hpp"
#include "skylake_256.hpp"
#include "skylake_512.hpp"
#include "broadwell.hpp"
#include "cannon_lake_256.hpp"
#include "cannon_lake_512.hpp"
#include "ice_lake.hpp"
#include "knl.hpp"
#include "zen.hpp"
#include "zen2.hpp"

struct benchmark_cpu {
  int n_threads;
  double gflops;
  const char* name;
  bench_type benchmark_type;
  void (*compute_function_256)(__m256 *farr_ptr, __m256, int);
  void (*compute_function_512)(__m512 *farr_ptr, __m512, int);
};

enum {
  BENCH_256_6_NOFMA,
  BENCH_256_5,
  BENCH_256_8,
  BENCH_256_10,
  BENCH_512_8,
  BENCH_512_12,
};

static const char *bench_name[] = {
  /*[BENCH_TYPE_SANDY_BRIDGE]    = */ "Sandy Bridge (AVX)",
  /*[BENCH_TYPE_IVY_BRIDGE]      = */ "Ivy Bridge (AVX)",
  /*[BENCH_TYPE_HASWELL]         = */ "Haswell (AVX2)",
  /*[BENCH_TYPE_BROADWELL]       = */ "Broadwell (AVX2)",
  /*[BENCH_TYPE_SKYLAKE_256]     = */ "Skylake (AVX2)",
  /*[BENCH_TYPE_SKYLAKE_512]     = */ "Skylake (AVX512)",
  /*[BENCH_TYPE_KABY_LAKE]       = */ "Kaby Lake (AVX2)",
  /*[BENCH_TYPE_COFFEE_LAKE]     = */ "Coffee Lake (AVX2)",
  /*[BENCH_TYPE_COMET_LAKE]      = */ "Comet Lake (AVX2)",
  /*[BENCH_TYPE_ICE_LAKE]        = */ "Ice Lake (AVX2)",
  /*[BENCH_TYPE_KNIGHTS_LANDING] = */ "Knights Landing (AVX512)",
  /*[BENCH_TYPE_ZEN]             = */ "Zen (AVX2)",
  /*[BENCH_TYPE_ZEN_PLUS]        = */ "Zen+ (AVX2)",
  /*[BENCH_TYPE_ZEN2]            = */ "Zen 2 (AVX2)",
};

static const char *bench_types_str[] = {
  /*[BENCH_TYPE_SANDY_BRIDGE]    = */ "sandy_bridge",
  /*[BENCH_TYPE_IVY_BRIDGE]      = */ "ivy_bridge",
  /*[BENCH_TYPE_HASWELL]         = */ "haswell",
  /*[BENCH_TYPE_BROADWELL]       = */ "broadwell",
  /*[BENCH_TYPE_SKYLAKE_256]     = */ "skylake_256",
  /*[BENCH_TYPE_SKYLAKE_512]     = */ "skylake_512",
  /*[BENCH_TYPE_KABY_LAKE]       = */ "kaby_lake",
  /*[BENCH_TYPE_COFFEE_LAKE]     = */ "coffee_lake",
  /*[BENCH_TYPE_COMET_LAKE]      = */ "comet_lake",
  /*[BENCH_TYPE_ICE_LAKE]        = */ "ice_lake",
  /*[BENCH_TYPE_KNIGHTS_LANDING] = */ "knights_landing",
  /*[BENCH_TYPE_ZEN]             = */ "zen",
  /*[BENCH_TYPE_ZEN_PLUS]        = */ "zen_plus",
  /*[BENCH_TYPE_ZEN2]            = */ "zen2",
};

bench_type parse_benchmark_cpu(char* str) {
  int len = sizeof(bench_types_str) / sizeof(bench_types_str[0]);
  for(bench_type t = 0; t < len; t++) {
    if(strcmp(str, bench_types_str[t]) == 0) {
      return t;    
    }
  }
  return BENCH_TYPE_INVALID;    
}

void print_bench_types_cpu(struct cpu* cpu) {
  int len = sizeof(bench_types_str) / sizeof(bench_types_str[0]);
  long unsigned int longest = 0;
  long unsigned int total_length = 0;
  for(bench_type t = 0; t < len; t++) {
    if(strlen(bench_name[t]) > longest) {
      longest = strlen(bench_name[t]);
      total_length = longest + 16 + strlen(bench_types_str[t]);
    }
  }

  printf("Available benchmark types for CPU:\n");
  for(long unsigned i=0; i < total_length; i++) putchar('-');
  putchar('\n');
  for(bench_type t = 0; t < len; t++) {
    printf("  - %s %*s(Keyword: %s)\n", bench_name[t], (int) (strlen(bench_name[t]) - longest), "", bench_types_str[t]);
  }
}

double compute_gflops(int n_threads, char bench) {
  int fma_available;
  int op_per_it;
  int bytes_in_vect;

  switch(bench) {
    case BENCH_256_6_NOFMA:
      fma_available = B_256_6_NOFMA_FMA_AV;
      op_per_it = B_256_6_NOFMA_OP_IT;
      bytes_in_vect = B_256_6_NOFMA_BYTES;
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
      printErr("Invalid benchmark type!");
      return -1.0;
  }

  return (double)((long)n_threads*MAXFLOPS_ITERS*op_per_it*(bytes_in_vect/4)*fma_available)/1000000000;
}

/*
 * Mapping between architecture and benchmark:
 * 
 * - Sandy Bridge    -> sandy_bridge
 * - Ivy Bridge      -> ivy_bridge
 * - Haswell         -> haswell
 * - Skylake (256)   -> skylake_256
 * - Skylake (512)   -> skylake_512
 * - Broadwell       -> broadwell
 * - Kaby Lake       -> skylake_256
 * - Coffee Lake     -> skylake_256
 * - Comet Lake      -> skylake_256
 * - Ice Lake        -> ice_lake
 * - Knights Landing -> knl
 * - Zen             -> zen
 * - Zen+            -> zen
 * - Zen 2           -> zen2
 */
bool select_benchmark(struct benchmark_cpu* bench) {
  bench->compute_function_256 = NULL;
  bench->compute_function_512 = NULL;

  switch(bench->benchmark_type) {
    case BENCH_TYPE_SANDY_BRIDGE:
      bench->compute_function_256 = compute_sandy_bridge;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_6_NOFMA);
      break;
    case BENCH_TYPE_IVY_BRIDGE:
      bench->compute_function_256 = compute_ivy_bridge;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_6_NOFMA);
      break;
    case BENCH_TYPE_HASWELL:
      bench->compute_function_256 = compute_haswell;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_10);
      break;
    case BENCH_TYPE_SKYLAKE_512:
      bench->compute_function_512 = compute_skylake_512;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_512_8);
      break;
    case BENCH_TYPE_SKYLAKE_256:
      bench->compute_function_256 = compute_skylake_256;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_8);
      break;
    case BENCH_TYPE_BROADWELL:
      bench->compute_function_256 = compute_broadwell;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_8);
      break;
    case BENCH_TYPE_KABY_LAKE:
      bench->compute_function_256 = compute_skylake_256;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_8);
      break;
    case BENCH_TYPE_COFFEE_LAKE:
      bench->compute_function_256 = compute_skylake_256;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_8);
      break;
    case BENCH_TYPE_COMET_LAKE:
      bench->compute_function_256 = compute_skylake_256;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_8);
      break;
    case BENCH_TYPE_ICE_LAKE:
      bench->compute_function_256 = compute_ice_lake;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_8);
      break;
    case BENCH_TYPE_KNIGHTS_LANDING:
      bench->compute_function_512 = compute_knl;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_512_12);
      break;
    case BENCH_TYPE_ZEN:
      bench->compute_function_256 = compute_zen;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_5);
      break;
    case BENCH_TYPE_ZEN_PLUS:
      bench->compute_function_256 = compute_zen;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_5);
      break;
    case BENCH_TYPE_ZEN2:
      bench->compute_function_256 = compute_zen2;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_10);
      break;
    default:
      printErr("No valid benchmark! (bench: %d)", bench->benchmark_type);
      return false;
  }

  bench->name = bench_name[bench->benchmark_type];
  return true;
}

struct benchmark_cpu* init_benchmark_cpu(struct cpu* cpu, int n_threads, bench_type benchmark_type) {    
  struct benchmark_cpu* bench = (struct benchmark_cpu*) malloc(sizeof(struct benchmark_cpu));

  bench->n_threads = n_threads;

  if(bench->n_threads == INVALID_CFG) {
    bench->n_threads = omp_get_max_threads();
  }
  if(bench->n_threads > MAX_NUMBER_THREADS) {
    printErr("Max number of threads is %d", MAX_NUMBER_THREADS);
    return NULL;
  }

  // Manual benchmark select
  if(benchmark_type != BENCH_TYPE_INVALID) {
    bench->benchmark_type = benchmark_type;
  }
  else {  // Automatic benchmark select
    struct uarch* uarch_struct = get_uarch_struct(cpu);
    MICROARCH u = uarch_struct->uarch;  
    bool avx = cpu_has_avx(cpu);
    bool avx512 = cpu_has_avx512(cpu);
    
    switch(u) {
      case UARCH_SANDY_BRIDGE:
        bench->benchmark_type = BENCH_TYPE_SANDY_BRIDGE;
        break;  
      case UARCH_IVY_BRIDGE:
        bench->benchmark_type = BENCH_TYPE_IVY_BRIDGE;
        break;  
      case UARCH_HASWELL:
        bench->benchmark_type = BENCH_TYPE_HASWELL;
        break;  
      case UARCH_SKYLAKE:
        if(avx512)
          bench->benchmark_type = BENCH_TYPE_SKYLAKE_512;
        else
          bench->benchmark_type = BENCH_TYPE_SKYLAKE_256;
        break;
      case UARCH_CASCADE_LAKE:
        bench->benchmark_type = BENCH_TYPE_SKYLAKE_512;
        break;  
      case UARCH_BROADWELL:
        bench->benchmark_type = BENCH_TYPE_BROADWELL;
        break;  
      case UARCH_KABY_LAKE:
        bench->benchmark_type = BENCH_TYPE_KABY_LAKE;
        break;  
      case UARCH_COFFEE_LAKE:
        bench->benchmark_type = BENCH_TYPE_COFFEE_LAKE;
        break;
      case UARCH_COMET_LAKE:
        bench->benchmark_type = BENCH_TYPE_COMET_LAKE;
        break;    
      case UARCH_ICE_LAKE:
        bench->benchmark_type = BENCH_TYPE_ICE_LAKE;
        break;
      case UARCH_KNIGHTS_LANDING:
        bench->benchmark_type = BENCH_TYPE_KNIGHTS_LANDING;
        break;  
      case UARCH_ZEN:
        bench->benchmark_type = BENCH_TYPE_ZEN;
        break;  
      case UARCH_ZEN_PLUS:
        bench->benchmark_type = BENCH_TYPE_ZEN_PLUS;
        break;
      case UARCH_ZEN2:
        bench->benchmark_type = BENCH_TYPE_ZEN2;
        break;    
      default:
        printErr("No valid uarch found! (uarch: %d)", u);
        return NULL;
    }
  }
  
  if(select_benchmark(bench))
    return bench;
  return NULL;
}

bool compute_cpu (struct benchmark_cpu* bench) {
  if(bench->benchmark_type == BENCH_TYPE_SKYLAKE_512 || bench->benchmark_type == BENCH_TYPE_KNIGHTS_LANDING) {
    __m512 mult = {0};
    __m512 *farr_ptr = NULL;

    #pragma omp parallel for
    for(int t=0; t < bench->n_threads; t++)
      bench->compute_function_512(farr_ptr, mult, t);
  }
  else {
    __m256 mult = {0};
    __m256 *farr_ptr = NULL;

    #pragma omp parallel for
    for(int t=0; t < bench->n_threads; t++)
      bench->compute_function_256(farr_ptr, mult, t);
  }
  return true;
}

double get_gflops_cpu(struct benchmark_cpu* bench) {
  return bench->gflops;
}

const char* get_benchmark_name_cpu(struct benchmark_cpu* bench) {
  return bench->name;
}

int get_n_threads(struct benchmark_cpu* bench) {
  return bench->n_threads;
}
