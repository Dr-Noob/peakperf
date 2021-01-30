#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#include <getopt.h>
#include <string.h>

#include "getarg.h"

#define OVERFLOW          -1
#define UNDERFLOW         -2
#define INVALID_ARG       -3

#define ARG_CHAR_HELP       'h'
#define ARG_CHAR_TRIALS     'r'
#define ARG_CHAR_WARMUP     'w'
#define ARG_CHAR_THREADS    't'
#define ARG_CHAR_BENCHMARK  'b'
#define ARG_CHAR_LISTBENCHS 'l'
#define ARG_CHAR_DEVICE     'd'
#define ARG_CHAR_VERSION    'v'

#define ARG_STR_CPU_DEVICE  "cpu"
#define ARG_STR_GPU_DEVICE  "gpu"

#define DEFAULT_N_TRIALS      10
#define DEFAULT_WARMUP_TRIALS  2

struct args_struct {
  bool help_flag;
  bool version_flag;
  bool list_benchmarks_flag;
  int n_trials;
  int n_warmup_trials;
  int n_threads;
  bench_type bench;
  device_type device;
};

int errn = 0;
static struct args_struct args;

device_type parse_device_type(char* str) {
  if(strcmp(str, ARG_STR_CPU_DEVICE) == 0) {
    return DEVICE_TYPE_CPU;    
  }  
  if(strcmp(str, ARG_STR_GPU_DEVICE) == 0) {
    return DEVICE_TYPE_GPU;    
  }
  return DEVICE_TYPE_INVALID;   
}

int getarg_int(char* str) {
  errn = 0;
  
  char* endptr;
  long tmp = strtol(str, &endptr, 10);

  if(*endptr) {
    errn = INVALID_ARG;
    return -1;
  }
  if(tmp == LONG_MIN) {
    errn = UNDERFLOW;
    return -1;
  }
  if(tmp == LONG_MAX) {
    errn = OVERFLOW;
    return -1;
  }
  if(tmp >= INT_MIN && tmp <= INT_MAX) {
    return (int)tmp;
  }
  errn = OVERFLOW;
  return -1;
}

void printerror() {
  switch (errn) {
    case OVERFLOW:
      printf("Overflow detected\n");
      break;
    case UNDERFLOW:
      printf("Underflow detected\n");
      break;
    case INVALID_ARG:
      printf("Invalid arg\n");
      break;
    default:
      printf("Invalid error: %d\n", errn);
      break;
  }
}

bool parseArgs(int argc, char* argv[]) {
  int opt;
  bool n_threads_set = false;
    
  args.help_flag = false;
  args.version_flag = false;
  args.list_benchmarks_flag = false;
  args.n_trials = DEFAULT_N_TRIALS;
  args.n_warmup_trials = DEFAULT_WARMUP_TRIALS;
  args.n_threads = INVALID_N_THREADS;
  args.bench = BENCH_TYPE_INVALID;
  args.device = DEVICE_TYPE_CPU;

  while ((opt = getopt(argc, argv, "hvlr:w:t:b:d:")) != -1) {
    switch (opt) {
    case ARG_CHAR_HELP:
      args.help_flag  = true;
      break;
      
    case ARG_CHAR_VERSION:
      args.version_flag  = true;
      break;
      
    case ARG_CHAR_DEVICE:
      args.device  = parse_device_type(optarg);
      if(args.device == DEVICE_TYPE_INVALID) {
        printf("ERROR: Invalid device: '%s'\n", optarg);
      }
      break;  
      
    case ARG_CHAR_LISTBENCHS:
      args.list_benchmarks_flag  = true;
      break;  
    
    case ARG_CHAR_TRIALS:
      args.n_trials = getarg_int(optarg);
      if(errn != 0) {
        printf("ERROR: Option -r: ");
        printerror();
        args.help_flag  = true;
        return false;
      }
      break;
      
    case ARG_CHAR_WARMUP:
      args.n_warmup_trials = getarg_int(optarg);
      if(errn != 0) {
        printf("ERROR: Option -w: ");
        printerror();
        args.help_flag  = true;    
        return false;
      }
      break;
      
    case ARG_CHAR_THREADS:
      n_threads_set = true;
      args.n_threads = getarg_int(optarg);
      if(errn != 0) {
        printf("ERROR: Option -t: ");
        printerror();
        args.help_flag  = true;   
        return false;
      }
      break;
      
    case ARG_CHAR_BENCHMARK:
      #ifdef DEVICE_CPU_ENABLED
        args.bench = parse_benchmark(optarg);
        if(args.bench == BENCH_TYPE_INVALID) {
          printf("ERROR: Option -b: Invalid benchmark\n");
          args.help_flag  = true;   
          return false;
        }
      #else
        printf("ERROR: Option is only valid with CPU\n");
        return false;
      #endif
      break;  
      
    default:
      printf("WARNING: Invalid options\n");
      args.help_flag  = true;
      return false;
    }
  }
  
  
  // check args //  
  if(args.n_trials <= 0) {
    printf("ERROR: Number of trials must be greater than zero\n");        
    return false;
  }
  if(args.n_warmup_trials < 0) {
    printf("ERROR: Number of warmup trials must be greater or equal to zero\n");        
    return false;
  }
  if(n_threads_set && args.n_threads <= 0) {
    printf("ERROR: Number of threads must be greater than zero\n");        
    return false;
  }
  
  return true;  
}

bool showHelp() {
  return args.help_flag;
}

bool showVersion() {
  return args.version_flag;
}

bool list_benchmarks() {
  return args.list_benchmarks_flag;
}

int get_n_trials() {
  return args.n_trials;
}

int get_warmup_trials() {
  return args.n_warmup_trials;
}

int get_n_threads() {
  return args.n_threads;    
}

bench_type get_benchmark_type() {
  return args.bench;    
}

device_type get_device_type() {
  return args.device;
}
