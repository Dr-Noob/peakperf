#include <sched.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <sys/time.h>

#include "arch.hpp"
#include "../../global.hpp"

#include "256_6_nofma.hpp"
#include "256_6.hpp"
#include "256_5.hpp"
#include "256_8.hpp"
#include "256_10.hpp"

struct benchmark_cpu_avx {
  void (*compute_function_256)(__m256 *farr_ptr, __m256, int);
  void (*compute_function_256_e)(__m256 *farr_ptr, __m256, int);
};

bool select_benchmark_avx(struct benchmark_cpu* bench) {
  bench->bench_avx = (struct benchmark_cpu_avx *) malloc(sizeof(struct benchmark_cpu));
  bench->bench_avx->compute_function_256 = NULL;

  switch(bench->benchmark_type) {
    case BENCH_TYPE_SANDY_BRIDGE:
    case BENCH_TYPE_IVY_BRIDGE:
      bench->bench_avx->compute_function_256 = compute_256_6_nofma;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_6_NOFMA);
      break;
    case BENCH_TYPE_HASWELL:
    case BENCH_TYPE_ZEN2:
      bench->bench_avx->compute_function_256 = compute_256_10;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_10);
      break;
    case BENCH_TYPE_BROADWELL:
    case BENCH_TYPE_SKYLAKE_256:
    case BENCH_TYPE_WHISKEY_LAKE_256:
    case BENCH_TYPE_KABY_LAKE:
    case BENCH_TYPE_COFFE_LAKE:
    case BENCH_TYPE_COMET_LAKE:
    case BENCH_TYPE_ICE_LAKE:
    case BENCH_TYPE_TIGER_LAKE:
    case BENCH_TYPE_ROCKET_LAKE:
    case BENCH_TYPE_ZEN3:
      bench->bench_avx->compute_function_256 = compute_256_8;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_8);
      break;
    case BENCH_TYPE_PILEDRIVER: // Piledriver should not use Zen file since it is compiled with AVX2 (piledriver is AVX only)
    case BENCH_TYPE_ZEN:
    case BENCH_TYPE_ZEN_PLUS:
      bench->bench_avx->compute_function_256 = compute_256_5;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_256_5);
      break;
    case BENCH_TYPE_ALDER_LAKE: // Might be an hybrid architecture
      bench->bench_avx->compute_function_256 = compute_256_8;
      if(bench->hybrid_flag) {
        // We have performance and efficiency cores
        bench->bench_avx->compute_function_256_e = compute_256_6;
        bench->gflops = compute_gflops(bench->h_topo->p_cores, BENCH_256_8) + compute_gflops(bench->h_topo->e_cores, BENCH_256_6);
      }
      else {
        // All cores are performance
        bench->gflops = compute_gflops(bench->n_threads, BENCH_256_8);
      }
      break;
    default:
      printErr("No valid benchmark! (bench: %d)", bench->benchmark_type);
      return false;
  }

  bench->name = bench_name[bench->benchmark_type];
  return true;
}

bool compute_cpu_avx (struct benchmark_cpu* bench, double* e_time) {
  bool sched_failed = false;
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  __m256 mult = {0};
  __m256 *farr_ptr = NULL;

  if(bench->hybrid_flag) {
    // We have a hybrid CPU with performance
    // and efficiency cores
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      cpu_set_t currentCPU;
      CPU_ZERO(&currentCPU);
      CPU_SET(tid, &currentCPU);
      if(sched_setaffinity(0, sizeof(currentCPU), &currentCPU) == -1) {
        perror("compute_cpu_avx: sched_setaffinity");
        #pragma omp critical
        sched_failed = true;
      }
      if(is_performance_core(bench->h_topo, tid) && !sched_failed) {
        #pragma omp for
        for(int t=0; t < bench->n_threads; t++)
          bench->bench_avx->compute_function_256(farr_ptr, mult, t);
      }
      else if(!sched_failed) {
        #pragma omp for
        for(int t=0; t < bench->n_threads; t++)
          bench->bench_avx->compute_function_256_e(farr_ptr, mult, t);
      }
    }
  }
  else {
    #pragma omp parallel for
    for(int t=0; t < bench->n_threads; t++)
      bench->bench_avx->compute_function_256(farr_ptr, mult, t);
  }

  gettimeofday(&t2, NULL);
  *e_time = (double)((t2.tv_sec-t1.tv_sec)*1000000 + t2.tv_usec-t1.tv_usec)/1000000;

  return !sched_failed;
}
