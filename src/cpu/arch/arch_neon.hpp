#ifndef ARCH_NEON_HPP
#define ARCH_NEON_HPP

#include <stdbool.h>

struct benchmark_cpu;
struct benchmark_cpu_neon;

bool select_benchmark_neon(struct benchmark_cpu* bench);
bool compute_cpu_neon(struct benchmark_cpu* bench, double* e_time);

#endif
