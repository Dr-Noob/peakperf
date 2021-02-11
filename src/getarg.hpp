#ifndef __GETARG__
#define __GETARG__

#include <stdbool.h>
#include "cpu/arch/arch.hpp"

#define INVALID_CFG -1

typedef char device_type;

enum {
  DEVICE_TYPE_CPU,
  DEVICE_TYPE_GPU,
  DEVICE_TYPE_INVALID
};

struct config {
  // DEVICE_CPU
  int n_threads;

  // DEVICE_GPU
  int nbk;
  int tpb;
};

bool parseArgs(int argc, char* argv[]);

bool showVersion();
bool showHelp();
bool list_benchmarks();
int get_n_trials();
int get_warmup_trials();
bench_type get_benchmark_type_args();
device_type get_device_type();
struct config* get_config();

#endif
