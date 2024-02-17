#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <sys/time.h>

#include "arch.hpp"
#include "../../global.hpp"

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
    case BENCH_128_6:
      fma_available = B_128_6_FMA_AV;
      op_per_it = B_128_6_OP_IT;
      bytes_in_vect = B_128_6_BYTES;
      break;
    case BENCH_128_8:
      fma_available = B_128_8_FMA_AV;
      op_per_it = B_128_8_OP_IT;
      bytes_in_vect = B_128_8_BYTES;
      break;
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
    case BENCH_256_6:
      fma_available = B_256_6_FMA_AV;
      op_per_it = B_256_6_OP_IT;
      bytes_in_vect = B_256_6_BYTES;
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

  return (double)((long)n_threads*BENCHMARK_CPU_ITERS*op_per_it*(bytes_in_vect/4)*fma_available)/1000000000;
}

/*
 * Mapping between architecture and benchmark:
 *
 * - Sandy Bridge    -> sandy_bridge
 * - Ivy Bridge      -> ivy_bridge
 * - Haswell         -> haswell
 * - Skylake (256)   -> skylake_128 / skylake_256
 * - Skylake (512)   -> skylake_512
 * - Broadwell       -> broadwell
 * - Whiskey Lake    -> skylake_256
 * - Kaby Lake       -> skylake_256
 * - Coffe Lake      -> skylake_256
 * - Comet Lake      -> skylake_256
 * - Ice Lake        -> ice_lake
 * - Tiger Lake      -> ice_lake
 * - Knights Landing -> knl
 * - Zen             -> zen
 * - Zen+            -> zen
 * - Zen 2           -> zen2
 * - Zen 3           -> zen3
 * - Zen 4           -> zen4
 */
bool select_benchmark(struct benchmark_cpu* bench) {
  if(bench->benchmark_type == BENCH_TYPE_SKYLAKE_128 || bench->benchmark_type == BENCH_TYPE_NEHALEM || bench->benchmark_type == BENCH_TYPE_AIRMONT || bench->benchmark_type == BENCH_TYPE_WHISKEY_LAKE_128)
    return select_benchmark_sse(bench);
  else if(bench->benchmark_type == BENCH_TYPE_SKYLAKE_512 || bench->benchmark_type == BENCH_TYPE_KNIGHTS_LANDING)
    return select_benchmark_avx512(bench);
  else
    return select_benchmark_avx(bench);
}

struct benchmark_cpu* init_benchmark_cpu(struct cpu* cpu, int n_threads, struct affinity_list* affinity, char *bench_type_str, bool pcores_only) {
  struct benchmark_cpu* bench = (struct benchmark_cpu*) malloc(sizeof(struct benchmark_cpu));
  bench_type benchmark_type;

  if(bench_type_str == NULL) {
    benchmark_type = BENCH_TYPE_INVALID;
  }
  else {
   benchmark_type = parse_benchmark_cpu(bench_type_str);
   if(benchmark_type == BENCH_TYPE_INVALID) {
     printErr("Invalid CPU benchmark specified: '%s'", bench_type_str);
     return NULL;
   }
  }

  int max_threads = omp_get_max_threads();

  bench->affinity = affinity;
  bench->pcores_only = pcores_only;
  bench->hybrid_flag = is_hybrid_cpu(cpu);
  bench->h_topo = get_hybrid_topology(cpu);
  if(bench->hybrid_flag && bench->pcores_only) {
    bench->n_threads = bench->h_topo->p_cores;
  }
  else {
    bench->n_threads = n_threads;
  }

  if (bench->affinity == NULL) {
    if(bench->n_threads == INVALID_CFG) {
      bench->n_threads = max_threads;
    }
    if(bench->n_threads > MAX_NUMBER_THREADS) {
      printErr("Max number of threads is %d", MAX_NUMBER_THREADS);
      return NULL;
    }
  }
  else {
    bool* thread_set = (bool *) malloc(sizeof(bool) * bench->affinity->n);
    memset(thread_set, 0, sizeof(bool) * bench->affinity->n);

    for (int i=0; i < bench->affinity->n; i++) {
      thread_set[i] = true;
      printf("%d\n", bench->affinity->list[i]);
      if (bench->affinity->list[i] > max_threads || bench->affinity->list[i] <= 0) {
        printErr("Affinity value %d is out of range (min=1,max=%d)", bench->affinity->list[i], max_threads);
        return NULL;
      }
    }

    int n_threads_aff = 0;
    for (int i=0; i < bench->affinity->n; i++) {
      if (thread_set[i]) n_threads_aff++;
    }
    bench->n_threads = n_threads_aff;
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
      case UARCH_AIRMONT:
        bench->benchmark_type = BENCH_TYPE_AIRMONT;
        break;
      case UARCH_NEHALEM:
      case UARCH_WESTMERE:
        bench->benchmark_type = BENCH_TYPE_NEHALEM;
        break;
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
        else if(avx)
          bench->benchmark_type = BENCH_TYPE_SKYLAKE_256;
        else
          bench->benchmark_type = BENCH_TYPE_SKYLAKE_128;
        break;
      case UARCH_CASCADE_LAKE:
        bench->benchmark_type = BENCH_TYPE_SKYLAKE_512;
        break;
      case UARCH_BROADWELL:
        bench->benchmark_type = BENCH_TYPE_BROADWELL;
        break;
      case UARCH_WHISKEY_LAKE:
        if(avx)
          bench->benchmark_type = BENCH_TYPE_WHISKEY_LAKE_256;
        else
          bench->benchmark_type = BENCH_TYPE_WHISKEY_LAKE_128;
        break;
      case UARCH_KABY_LAKE:
        bench->benchmark_type = BENCH_TYPE_KABY_LAKE;
        break;
      case UARCH_COFFEE_LAKE:
        bench->benchmark_type = BENCH_TYPE_COFFE_LAKE;
        break;
      case UARCH_COMET_LAKE:
        bench->benchmark_type = BENCH_TYPE_COMET_LAKE;
        break;
      case UARCH_ICE_LAKE:
        bench->benchmark_type = BENCH_TYPE_ICE_LAKE;
        break;
      case UARCH_TIGER_LAKE:
        bench->benchmark_type = BENCH_TYPE_TIGER_LAKE;
        break;
      case UARCH_ROCKET_LAKE:
        bench->benchmark_type = BENCH_TYPE_ROCKET_LAKE;
        break;
      case UARCH_ALDER_LAKE:
        bench->benchmark_type = BENCH_TYPE_ALDER_LAKE;
        break;
      case UARCH_RAPTOR_LAKE:
        bench->benchmark_type = BENCH_TYPE_RAPTOR_LAKE;
        break;
      case UARCH_KNIGHTS_LANDING:
        bench->benchmark_type = BENCH_TYPE_KNIGHTS_LANDING;
        break;
      case UARCH_PILEDRIVER:
        bench->benchmark_type = BENCH_TYPE_PILEDRIVER;
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
      case UARCH_ZEN3:
      case UARCH_ZEN3_PLUS:
        bench->benchmark_type = BENCH_TYPE_ZEN3;
        break;
      case UARCH_ZEN4:
        bench->benchmark_type = BENCH_TYPE_ZEN4;
        break;
      default:
        printErr("Found invalid uarch: '%s'", uarch_struct->uarch_str);
        printErr("peakperf is unable to automatically select the benchmark for your CPU. Please, select the benchmark manually (see peakperf -h) and/or post this error message in https://github.com/Dr-Noob/peakperf/issues");
        return NULL;
    }
  }

  if(select_benchmark(bench))
    return bench;
  return NULL;
}

bool compute_cpu(struct benchmark_cpu* bench, double* e_time) {
  if(bench->benchmark_type == BENCH_TYPE_SKYLAKE_128 || bench->benchmark_type == BENCH_TYPE_NEHALEM || bench->benchmark_type == BENCH_TYPE_AIRMONT)
    return compute_cpu_sse(bench, e_time);
  else if(bench->benchmark_type == BENCH_TYPE_SKYLAKE_512 || bench->benchmark_type == BENCH_TYPE_KNIGHTS_LANDING)
    return compute_cpu_avx512(bench, e_time);
  else
    return compute_cpu_avx(bench, e_time);
}

double get_gflops_cpu(struct benchmark_cpu* bench) {
  return bench->gflops;
}

const char* get_benchmark_name_cpu(struct benchmark_cpu* bench) {
  return bench->name;
}

const char* get_hybrid_topology_string_cpu(struct benchmark_cpu* bench) {
  if(bench->hybrid_flag) {
    /* Fancy
    int str_len = 3 + strlen("(performance)") + 6 + strlen("(efficiency)") + 1;
    char* h_topo_str = (char *) malloc(sizeof(char) * str_len);
    memset(h_topo_str, 0, str_len);
    sprintf(h_topo_str, "%d (performance) + %d (efficiency)", bench->h_topo->p_cores, bench->h_topo->e_cores);
    return h_topo_str;
    */
    int ncores = bench->h_topo->p_cores + bench->h_topo->e_cores;
    char* h_topo_str = (char *) malloc(sizeof(char) * (ncores + 1));
    memset(h_topo_str, 0, (ncores + 1));
    for(int i=0; i < ncores; i++) {
      h_topo_str[i] = bench->h_topo->core_mask[i] ? '1' : '0';
    }
    return h_topo_str;
  }
  return NULL;
}

const char* get_affinity_string_cpu(struct benchmark_cpu* bench) {
  if (bench->affinity == NULL)
    return NULL;

  int aff_str_i = 0;
  char* aff_str = (char *) malloc(sizeof(char) * (1000));
  memset(aff_str, 0, 1000);
  for (int i=0; i < bench->affinity->n; i++) {
    aff_str_i += sprintf(aff_str + aff_str_i, "%d", bench->affinity->list[i]);
    if (i+1 < bench->affinity->n)
      aff_str_i += sprintf(aff_str + aff_str_i, ",");
  }
  return aff_str;
}

int get_n_threads(struct benchmark_cpu* bench) {
  // If no affinity was specified, return the max number of threads
  if (bench->affinity == NULL)
     return bench->n_threads;
  return bench->affinity->n;
}
