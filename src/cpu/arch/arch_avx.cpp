#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <sys/time.h>

#include "arch.hpp"
#include "../../global.hpp"

#include "sandy_bridge.hpp"
#include "ivy_bridge.hpp"
#include "haswell.hpp"
#include "skylake_256.hpp"
#include "broadwell.hpp"
#include "cannon_lake_256.hpp"
#include "ice_lake.hpp"
#include "zen.hpp"
#include "zen2.hpp"

struct benchmark_cpu_avx {
  void (*compute_function_256)(__m256 *farr_ptr, __m256, int);
};

/*
 * Mapping between architecture and benchmark:
 *
 * - Sandy Bridge       -> sandy_bridge
 * - Ivy Bridge         -> ivy_bridge
 * - Haswell            -> haswell
 * - Skylake (256)      -> skylake_256
 * - Broadwell          -> broadwell
 * - Whiskey Lake (256) -> skylake_256
 * - Kaby Lake          -> skylake_256
 * - Coffe Lake         -> skylake_256
 * - Comet Lake         -> skylake_256
 * - Ice Lake           -> ice_lake
 * - Tiger Lake         -> ice_lake
 * - Piledriver         -> zen
 * - Zen                -> zen
 * - Zen+               -> zen
 * - Zen 2              -> zen2
 */
bool select_benchmark_avx(struct benchmark_cpu* bench) {
  bench->bench_avx = (struct benchmark_cpu_avx *) malloc(sizeof(struct benchmark_cpu));
  bench->bench_avx->compute_function_256 = NULL;

  switch(bench->benchmark_type) {
    case BENCH_TYPE_SANDY_BRIDGE:
      bench->bench_avx->compute_function_256 = compute_sandy_bridge;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_6_NOFMA);
      break;
    case BENCH_TYPE_IVY_BRIDGE:
      bench->bench_avx->compute_function_256 = compute_ivy_bridge;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_6_NOFMA);
      break;
    case BENCH_TYPE_HASWELL:
      bench->bench_avx->compute_function_256 = compute_haswell;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_10);
      break;
    case BENCH_TYPE_SKYLAKE_256:
      bench->bench_avx->compute_function_256 = compute_skylake_256;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_8);
      break;
    case BENCH_TYPE_BROADWELL:
      bench->bench_avx->compute_function_256 = compute_broadwell;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_8);
      break;
    case BENCH_TYPE_WHISKEY_LAKE_256:
      bench->bench_avx->compute_function_256 = compute_skylake_256;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_8);
      break;
    case BENCH_TYPE_KABY_LAKE:
      bench->bench_avx->compute_function_256 = compute_skylake_256;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_8);
      break;
    case BENCH_TYPE_COFFE_LAKE:
      bench->bench_avx->compute_function_256 = compute_skylake_256;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_8);
      break;
    case BENCH_TYPE_COMET_LAKE:
      bench->bench_avx->compute_function_256 = compute_skylake_256;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_8);
      break;
    case BENCH_TYPE_ICE_LAKE:
      bench->bench_avx->compute_function_256 = compute_ice_lake;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_8);
      break;
    case BENCH_TYPE_TIGER_LAKE:
      bench->bench_avx->compute_function_256 = compute_ice_lake;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_8);
      break;
    case BENCH_TYPE_PILEDRIVER: // Piledriver should not use Zen file since it is compiled with AVX2 (piledriver is AVX only)
    case BENCH_TYPE_ZEN:
      bench->bench_avx->compute_function_256 = compute_zen;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_5);
      break;
    case BENCH_TYPE_ZEN_PLUS:
      bench->bench_avx->compute_function_256 = compute_zen;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_5);
      break;
    case BENCH_TYPE_ZEN2:
      bench->bench_avx->compute_function_256 = compute_zen2;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_10);
      break;
    default:
      printErr("No valid benchmark! (bench: %d)", bench->benchmark_type);
      return false;
  }

  bench->name = bench_name[bench->benchmark_type];
  return true;
}

bool compute_cpu_avx (struct benchmark_cpu* bench, double* e_time) {
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  __m256 mult = {0};
  __m256 *farr_ptr = NULL;

  #pragma omp parallel for
  for(int t=0; t < bench->n_threads; t++)
    bench->bench_avx->compute_function_256(farr_ptr, mult, t);

  gettimeofday(&t2, NULL);
  *e_time = (double)((t2.tv_sec-t1.tv_sec)*1000000 + t2.tv_usec-t1.tv_usec)/1000000;

  return true;
}
