#include <sched.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>

#include "arch.hpp"
#include "../../global.hpp"

#ifdef ARCH_ARM
  #include <arm_neon.h>
  #include "neon_4.hpp"
  #include "neon_6.hpp"
#endif

struct benchmark_cpu_neon {
#ifdef ARCH_ARM
  void (*compute_function_neon)(float32x4_t *farr_ptr, float32x4_t, int);
#else
  void* compute_function_neon;
#endif
};

bool select_benchmark_neon(struct benchmark_cpu* bench) {
  bench->bench_neon = (struct benchmark_cpu_neon*) malloc(sizeof(struct benchmark_cpu_neon));
  bench->bench_neon->compute_function_neon = NULL;

#ifdef ARCH_ARM
  switch(bench->benchmark_type) {
    case BENCH_TYPE_ARM_NEON:
      // Default to neon_6
      bench->bench_neon->compute_function_neon = compute_neon_6;
      bench->gflops = compute_gflops(bench->n_threads, BENCH_NEON_6);
      break;
    default:
      printErr("No valid benchmark! (bench: %d)", bench->benchmark_type);
      return false;
  }
#else
  (void)bench;
#endif

  bench->name = bench_name[bench->benchmark_type];
  return true;
}

bool compute_cpu_neon (struct benchmark_cpu* bench, double* e_time) {
#ifdef ARCH_ARM
  bool sched_failed = false;
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  float32x4_t mult = vdupq_n_f32(0.0f);
  float32x4_t *farr_ptr = NULL;

  if(bench->affinity != NULL) {
    #pragma omp parallel num_threads(bench->affinity->n)
    {
      int tid = bench->affinity->list[omp_get_thread_num()]-1;
      cpu_set_t currentCPU;
      CPU_ZERO(&currentCPU);
      CPU_SET(tid, &currentCPU);
      if(sched_setaffinity(0, sizeof(currentCPU), &currentCPU) == -1) {
        perror("compute_cpu_neon: sched_setaffinity");
        #pragma omp critical
        sched_failed = true;
      }
      if(!sched_failed) {
        #pragma omp for
        for(int t=0; t < bench->n_threads; t++)
          bench->bench_neon->compute_function_neon(farr_ptr, mult, t);
      }
    }
  }
  else {
    #pragma omp parallel for
      for(int t=0; t < bench->n_threads; t++)
        bench->bench_neon->compute_function_neon(farr_ptr, mult, t);
  }

  gettimeofday(&t2, NULL);
  *e_time = (double)((t2.tv_sec-t1.tv_sec)*1000000 + t2.tv_usec-t1.tv_usec)/1000000;

  return true;
#else
  (void)bench;
  (void)e_time;
  return false;
#endif
}
