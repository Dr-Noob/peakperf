#ifndef __ARCH__
#define __ARCH__

#include <immintrin.h>
#include "../cpufetch/uarch.hpp"
#include "../../getarg.hpp"

enum bench_types {
  BENCH_TYPE_SANDY_BRIDGE,
  BENCH_TYPE_IVY_BRIDGE,
  BENCH_TYPE_HASWELL,
  BENCH_TYPE_BROADWELL,
  BENCH_TYPE_SKYLAKE_256,
  BENCH_TYPE_SKYLAKE_512,
  BENCH_TYPE_KABY_LAKE,
  BENCH_TYPE_COFFEE_LAKE,
  BENCH_TYPE_COMET_LAKE,
  BENCH_TYPE_ICE_LAKE,
  BENCH_TYPE_KNIGHTS_LANDING,
  BENCH_TYPE_ZEN,
  BENCH_TYPE_ZEN_PLUS,
  BENCH_TYPE_ZEN2,
};

#define MAXFLOPS_ITERS 1000000000
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
#endif

struct benchmark_cpu;

struct benchmark_cpu* init_benchmark_cpu(struct cpu* cpu, int n_threads, char* bench_type_str);
bool compute_cpu(struct benchmark_cpu* bench);
double get_gflops_cpu(struct benchmark_cpu* bench);
const char* get_benchmark_name_cpu(struct benchmark_cpu* bench);
bench_type parse_benchmark_cpu(char* str);
void print_bench_types_cpu(struct cpu* cpu);
int get_n_threads(struct benchmark_cpu* bench);

#endif
