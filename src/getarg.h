#ifndef __GETARG__
#define __GETARG__

#include <stdbool.h>
#include "cpu/arch/arch.h"

#define INVALID_N_THREADS -1

typedef char bench_mode;

enum {
  BENCH_MODE_CPU,
  BENCH_MODE_GPU,
  BENCH_MODE_INVALID
};

bool parseArgs(int argc, char* argv[]);

bool showVersion();
bool showHelp();
bool list_benchmarks();
int get_n_trials();
int get_warmup_trials();
int get_n_threads();
bench_type get_benchmark_type();
bench_mode get_benchmark_mode();

#endif
