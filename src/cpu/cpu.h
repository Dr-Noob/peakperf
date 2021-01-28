#ifndef __PEAKPERF_CPU_MODE__
#define __PEAKPERF_CPU_MODE__

#include <stdbool.h>
#include "arch/arch.h"

#define INVALID_N_THREADS -1

int peakperf_cpu(int nTrials, int nWarmupTrials, int n_threads, bool list_benchmarks, bench_type benchmark_type);

#endif
