#ifndef __GETARG__
#define __GETARG__

#include <stdbool.h>
#include "Arch/Arch.h"

#define INVALID_N_THREADS -1

bool parseArgs(int argc, char* argv[]);

bool showHelp();
bool list_benchmarks();
int get_n_trials();
int get_warmup_trials();
int get_n_threads();
bench_type get_benchmark_type();

#endif
