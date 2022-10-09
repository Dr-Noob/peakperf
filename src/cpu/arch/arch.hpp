#ifndef __ARCH__
#define __ARCH__

#include <immintrin.h>
#include "../cpufetch/uarch.hpp"
#include "../../getarg.hpp"

#include "arch_sse.hpp"
#include "arch_avx.hpp"
#include "arch_avx512.hpp"

struct benchmark_cpu {
  int n_threads;
  double gflops;
  const char* name;
  bench_type benchmark_type;

  struct benchmark_cpu_sse* bench_sse;
  struct benchmark_cpu_avx* bench_avx;
  struct benchmark_cpu_avx512* bench_avx512;
};

enum {
  BENCH_128_6,
  BENCH_128_8,
  BENCH_256_6_NOFMA,
  BENCH_256_5,
  BENCH_256_8,
  BENCH_256_10,
  BENCH_512_8,
  BENCH_512_12,
};

enum bench_types {
  BENCH_TYPE_AIRMONT,
  BENCH_TYPE_NEHALEM,
  BENCH_TYPE_SANDY_BRIDGE,
  BENCH_TYPE_IVY_BRIDGE,
  BENCH_TYPE_HASWELL,
  BENCH_TYPE_BROADWELL,
  BENCH_TYPE_SKYLAKE_128,
  BENCH_TYPE_SKYLAKE_256,
  BENCH_TYPE_SKYLAKE_512,
  BENCH_TYPE_WHISKEY_LAKE_128,
  BENCH_TYPE_WHISKEY_LAKE_256,
  BENCH_TYPE_KABY_LAKE,
  BENCH_TYPE_COFFE_LAKE,
  BENCH_TYPE_COMET_LAKE,
  BENCH_TYPE_ICE_LAKE,
  BENCH_TYPE_TIGER_LAKE,
  BENCH_TYPE_KNIGHTS_LANDING,
  BENCH_TYPE_PILEDRIVER,
  BENCH_TYPE_ZEN,
  BENCH_TYPE_ZEN_PLUS,
  BENCH_TYPE_ZEN2,
};

static const char *bench_name[] = {
  /*[BENCH_TYPE_AIRMONT]         = */ "Airmont (SSE)",
  /*[BENCH_TYPE_NEHALEM]         = */ "Nehalem (SSE)",
  /*[BENCH_TYPE_SANDY_BRIDGE]    = */ "Sandy Bridge (AVX)",
  /*[BENCH_TYPE_IVY_BRIDGE]      = */ "Ivy Bridge (AVX)",
  /*[BENCH_TYPE_HASWELL]         = */ "Haswell (AVX2)",
  /*[BENCH_TYPE_BROADWELL]       = */ "Broadwell (AVX2)",
  /*[BENCH_TYPE_SKYLAKE_256]     = */ "Skylake (SSE)",
  /*[BENCH_TYPE_SKYLAKE_256]     = */ "Skylake (AVX2)",
  /*[BENCH_TYPE_SKYLAKE_512]     = */ "Skylake (AVX512)",
  /*[BENCH_TYPE_KABY_LAKE]       = */ "Kaby Lake (AVX2)",
  /*[BENCH_TYPE_COFFE_LAKE]      = */ "Coffe Lake (AVX2)",
  /*[BENCH_TYPE_COMET_LAKE]      = */ "Comet Lake (AVX2)",
  /*[BENCH_TYPE_ICE_LAKE]        = */ "Ice Lake (AVX2)",
  /*[BENCH_TYPE_TIGER_LAKE]      = */ "Tiger Lake (AVX2)",
  /*[BENCH_TYPE_KNIGHTS_LANDING] = */ "Knights Landing (AVX512)",
  /*[BENCH_TYPE_PILEDRIVER       = */ "Piledriver (AVX)",
  /*[BENCH_TYPE_ZEN]             = */ "Zen (AVX2)",
  /*[BENCH_TYPE_ZEN_PLUS]        = */ "Zen+ (AVX2)",
  /*[BENCH_TYPE_ZEN2]            = */ "Zen 2 (AVX2)",
};

static const char *bench_types_str[] = {
  /*[BENCH_TYPE_AIRMONT]         = */ "airmont",
  /*[BENCH_TYPE_NEHALEM]         = */ "nehalem",
  /*[BENCH_TYPE_SANDY_BRIDGE]    = */ "sandy_bridge",
  /*[BENCH_TYPE_IVY_BRIDGE]      = */ "ivy_bridge",
  /*[BENCH_TYPE_HASWELL]         = */ "haswell",
  /*[BENCH_TYPE_BROADWELL]       = */ "broadwell",
  /*[BENCH_TYPE_SKYLAKE_256]     = */ "skylake_128",
  /*[BENCH_TYPE_SKYLAKE_256]     = */ "skylake_256",
  /*[BENCH_TYPE_SKYLAKE_512]     = */ "skylake_512",
  /*[BENCH_TYPE_KABY_LAKE]       = */ "kaby_lake",
  /*[BENCH_TYPE_COFFE_LAKE]      = */ "coffe_lake",
  /*[BENCH_TYPE_COMET_LAKE]      = */ "comet_lake",
  /*[BENCH_TYPE_ICE_LAKE]        = */ "ice_lake",
  /*[BENCH_TYPE_TIGER_LAKE]      = */ "tiger_lake",
  /*[BENCH_TYPE_KNIGHTS_LANDING] = */ "knights_landing",
  /*[BENCH_TYPE_PILEDRIVER]      = */ "piledriver",
  /*[BENCH_TYPE_ZEN]             = */ "zen",
  /*[BENCH_TYPE_ZEN_PLUS]        = */ "zen_plus",
  /*[BENCH_TYPE_ZEN2]            = */ "zen2",
};

#define BENCHMARK_CPU_ITERS 1000000000
#define MAX_NUMBER_THREADS 512

/* 
 * Values for each benchmark:
 * =============================
 * > FMA_AV: 
 *   - FMA not available: 1
 *   - FMA available: 2
 * > OP_IT:
 *   - Operations per iteration
 * > BYTES (bytes in vector):
 *   - AVX / AVX2 : 32 bytes
 *   - AVX512 : 64 bytes
 */
//      AVX_128_6            //
#define B_128_6_FMA_AV       1
#define B_128_6_OP_IT        6
#define B_128_6_BYTES        16
//      AVX_128_8            //
#define B_128_8_FMA_AV       1
#define B_128_8_OP_IT        8
#define B_128_8_BYTES        16
//      AVX_256_3_NOFMA      //
#define B_256_6_NOFMA_FMA_AV 1
#define B_256_6_NOFMA_OP_IT  6
#define B_256_6_NOFMA_BYTES  32
//      AVX_256_5            //
#define B_256_5_FMA_AV       2
#define B_256_5_OP_IT        5
#define B_256_5_BYTES        32
//      AVX_256_8            //
#define B_256_8_FMA_AV       2
#define B_256_8_OP_IT        8
#define B_256_8_BYTES        32
//      AVX_256_10           //
#define B_256_10_FMA_AV      2
#define B_256_10_OP_IT       10
#define B_256_10_BYTES       32
//      AVX_512_8            //
#define B_512_8_FMA_AV       2
#define B_512_8_OP_IT        8
#define B_512_8_BYTES        64
//      AVX_512_12           //
#define B_512_12_FMA_AV      2
#define B_512_12_OP_IT       12
#define B_512_12_BYTES       64

#if defined(AVX_512_12) || defined(AVX_512_8)
  #define BYTES_IN_VECT 64
  #define TYPE __m512
  #define SIZE OP_PER_IT*2
#elif defined(AVX_256_10) || defined(AVX_256_8) || defined(AVX_256_5) || defined(AVX_256_6_NOFMA)
  #define BYTES_IN_VECT 32
  #define TYPE __m256
  #define SIZE OP_PER_IT*2
#elif defined(SSE_128_8) || defined(SSE_128_6)
  #define BYTES_IN_VECT 16
  #define TYPE __m128
  #define SIZE OP_PER_IT*2
#endif

struct benchmark_cpu;

struct benchmark_cpu* init_benchmark_cpu(struct cpu* cpu, int n_threads, char* bench_type_str);
bool compute_cpu(struct benchmark_cpu* bench, double* e_time);
double get_gflops_cpu(struct benchmark_cpu* bench);
const char* get_benchmark_name_cpu(struct benchmark_cpu* bench);
bench_type parse_benchmark_cpu(char* str);
void print_bench_types_cpu(struct cpu* cpu);
int get_n_threads(struct benchmark_cpu* bench);
double compute_gflops(int n_threads, char bench);

#endif
