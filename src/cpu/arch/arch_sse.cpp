#include <sched.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <sys/time.h>

#include "arch.hpp"
#include "../../global.hpp"

#include "128_6.hpp"
#include "128_8.hpp"

struct benchmark_cpu_sse {
  void (*compute_function_128)(__m128 *farr_ptr, __m128, int);
};

bool select_benchmark_sse(struct benchmark_cpu* bench) {
  bench->bench_sse = (struct benchmark_cpu_sse*) malloc(sizeof(struct benchmark_cpu_sse));
  bench->bench_sse->compute_function_128 = NULL;

  switch(bench->benchmark_type) {
    case BENCH_TYPE_AIRMONT:
      bench->bench_sse->compute_function_128 = compute_128_6;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_128_6);
      break;
    case BENCH_TYPE_NEHALEM:
    case BENCH_TYPE_SKYLAKE_128:
    case BENCH_TYPE_WHISKEY_LAKE_128:
      bench->bench_sse->compute_function_128 = compute_128_8;
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
  bool sched_failed = false;
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  __m128 mult = {0};
  __m128 *farr_ptr = NULL;

  if(bench->affinity != NULL) {
    #pragma omp parallel num_threads(bench->affinity->n)
    {
      int tid = bench->affinity->list[omp_get_thread_num()]-1;
      cpu_set_t currentCPU;
      CPU_ZERO(&currentCPU);
      CPU_SET(tid, &currentCPU);
      if(sched_setaffinity(0, sizeof(currentCPU), &currentCPU) == -1) {
        perror("compute_cpu_avx: sched_setaffinity");
        #pragma omp critical
        sched_failed = true;
      }
      if(!sched_failed) {
        #pragma omp for
        for(int t=0; t < bench->n_threads; t++)
          bench->bench_sse->compute_function_128(farr_ptr, mult, t);
      }
    }
  }
  else {
    #pragma omp parallel for
      for(int t=0; t < bench->n_threads; t++)
        bench->bench_sse->compute_function_128(farr_ptr, mult, t);
  }

  gettimeofday(&t2, NULL);
  *e_time = (double)((t2.tv_sec-t1.tv_sec)*1000000 + t2.tv_usec-t1.tv_usec)/1000000;

  return true;
}

