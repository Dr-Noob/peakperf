#include <sched.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <sys/time.h>

#include "arch.hpp"
#include "../../global.hpp"

#include "512_8.hpp"
#include "512_12.hpp"

struct benchmark_cpu_avx512 {
  void (*compute_function_512)(__m512 *farr_ptr, __m512, int);
};

bool select_benchmark_avx512(struct benchmark_cpu* bench) {
  bench->bench_avx512 = (struct benchmark_cpu_avx512*) malloc(sizeof(struct benchmark_cpu_avx512));
  bench->bench_avx512->compute_function_512 = NULL;

  switch(bench->benchmark_type) {
    case BENCH_TYPE_SKYLAKE_512:
      bench->bench_avx512->compute_function_512 = compute_512_8;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_512_8);
      break;
    case BENCH_TYPE_KNIGHTS_LANDING:
      bench->bench_avx512->compute_function_512 = compute_512_12;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_512_12);
      break;
    default:
      printErr("No valid benchmark! (bench: %d)", bench->benchmark_type);
      return false;
  }

  bench->name = bench_name[bench->benchmark_type];
  return true;
}

bool compute_cpu_avx512 (struct benchmark_cpu* bench, double* e_time) {
  bool sched_failed = false;
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  __m512 mult = {0};
  __m512 *farr_ptr = NULL;

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
          bench->bench_avx512->compute_function_512(farr_ptr, mult, t);
      }
    }
  }
  else {
    #pragma omp parallel for
    for(int t=0; t < bench->n_threads; t++)
      bench->bench_avx512->compute_function_512(farr_ptr, mult, t);
  }

  gettimeofday(&t2, NULL);
  *e_time = (double)((t2.tv_sec-t1.tv_sec)*1000000 + t2.tv_usec-t1.tv_usec)/1000000;

  return true;
}
