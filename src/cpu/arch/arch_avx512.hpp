#ifndef __ARCH_AVX512__
#define __ARCH_AVX512__

#include <stdbool.h>

struct benchmark_cpu_avx512;

bool select_benchmark_avx512(struct benchmark_cpu* bench);
bool compute_cpu_avx512(struct benchmark_cpu* bench, double* e_time);

#endif
