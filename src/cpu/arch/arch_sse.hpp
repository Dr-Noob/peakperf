#ifndef __ARCH_SSE__
#define __ARCH_SSE__

#include <stdbool.h>

struct benchmark_cpu_sse;

bool select_benchmark_sse(struct benchmark_cpu* bench);
bool compute_cpu_sse(struct benchmark_cpu* bench, double* e_time);

#endif
