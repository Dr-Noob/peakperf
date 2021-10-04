#ifndef __ARCH_AVX__
#define __ARCH_AVX__

#include <stdbool.h>

struct benchmark_cpu_avx;

bool select_benchmark_avx(struct benchmark_cpu* bench);
bool compute_cpu_avx(struct benchmark_cpu* bench, double* e_time);

#endif
