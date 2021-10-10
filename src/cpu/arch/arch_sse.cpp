#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <sys/time.h>

#include "arch.hpp"
#include "../../global.hpp"

#include "airmont.hpp"
#include "nehalem.hpp"
#include "skylake_128.hpp"

struct benchmark_cpu_sse {
  void (*compute_function_128)(__m128 *farr_ptr, __m128, int);
};

/*
 * Mapping between architecture and benchmark:
 *
 * - Skylake (128)   -> skylake_128
 */
bool select_benchmark_sse(struct benchmark_cpu* bench) {
  bench->bench_sse = (struct benchmark_cpu_sse*) malloc(sizeof(struct benchmark_cpu_sse));
  bench->bench_sse->compute_function_128 = NULL;

  switch(bench->benchmark_type) {
    case BENCH_TYPE_AIRMONT:
      bench->bench_sse->compute_function_128 = compute_airmont;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_128_6);
      break;
    case BENCH_TYPE_NEHALEM:
      bench->bench_sse->compute_function_128 = compute_nehalem;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_128_8);
      break;
    case BENCH_TYPE_SKYLAKE_128:
      bench->bench_sse->compute_function_128 = compute_skylake_128;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_128_8);
      break;
    default:
      printErr("No valid benchmark! (bench: %d)", bench->benchmark_type);
      return false;
  }

  bench->name = bench_name[bench->benchmark_type];
  return true;
}

bool compute_cpu_sse (struct benchmark_cpu* bench, double* e_time) {
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  __m128 mult = {0};
  __m128 *farr_ptr = NULL;

  #pragma omp parallel for
    for(int t=0; t < bench->n_threads; t++)
      bench->bench_sse->compute_function_128(farr_ptr, mult, t);

  gettimeofday(&t2, NULL);
  *e_time = (double)((t2.tv_sec-t1.tv_sec)*1000000 + t2.tv_usec-t1.tv_usec)/1000000;

  return true;
}

