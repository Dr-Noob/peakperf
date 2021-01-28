#ifndef __GETARG__
#define __GETARG__

#include <stdbool.h>
#include "cpu/arch/arch.h"

#define INVALID_N_THREADS -1

typedef char peakperf_mode;

enum {
  PEAKPERF_MODE_CPU,
  PEAKPERF_MODE_INVALID
};

bool parseArgs(int argc, char* argv[]);

bool showVersion();
bool showHelp();
bool list_benchmarks();
int get_n_trials();
int get_warmup_trials();
int get_n_threads();
bench_type get_benchmark_type();
peakperf_mode get_mode();

#endif
