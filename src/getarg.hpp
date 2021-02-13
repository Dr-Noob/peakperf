#ifndef __GETARG__
#define __GETARG__

#include <stdbool.h>
#include "cpu/arch/arch.hpp"

#define INVALID_CFG -1

enum {
  ARG_LISTBENCHS,
  ARG_BENCHMARK,
  ARG_DEVICE,
  ARG_TRIALS,
  ARG_WARMUP,
  ARG_CPU_THREADS,
  ARG_GPU_BLOCKS,
  ARG_GPU_TPB,
  ARG_HELP,
  ARG_VERSION
};

static constexpr char args_chr[] = {
  /*[ARG_LISTBENCHS] = */  'l',
  /*[ARG_BENCHMARK] = */   'b',
  /*[ARG_DEVICE] = */      'd',
  /*[ARG_TRIALS] = */      'r',
  /*[ARG_WARMUP] = */      'w',
  /*[ARG_CPU_THREADS] = */ 't',
  /*[ARG_GPU_BLOCKS] = */  'B',
  /*[ARG_GPU_TPB] = */     'T',
  /*[ARG_HELP] = */        'h',
  /*[ARG_VERSION] = */     'v'
};

static const char *args_str[] = {
  /*[ARG_LISTBENCHS] = */  "list",
  /*[ARG_BENCHMARK] = */   "benchmark",
  /*[ARG_DEVICE] = */      "device",
  /*[ARG_TRIALS] = */      "trials",
  /*[ARG_WARMUP] = */      "warmup-trials",
  /*[ARG_CPU_THREADS] = */ "threads",
  /*[ARG_GPU_BLOCKS] = */  "blocks",
  /*[ARG_GPU_TPB] = */     "threads-per-block",
  /*[ARG_HELP] = */        "help",
  /*[ARG_VERSION] = */     "version"
};

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
