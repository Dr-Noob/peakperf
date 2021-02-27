#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>

#include "getarg.hpp"
#include "global.hpp"
#include "cpu/arch/arch.hpp"
#include "gpu/arch/arch.hpp"

#define OVERFLOW          -1
#define UNDERFLOW         -2
#define INVALID_ARG       -3

#define ARG_STR_CPU_DEVICE  "cpu"
#define ARG_STR_GPU_DEVICE  "gpu"

#define DEFAULT_N_TRIALS      10
#define DEFAULT_WARMUP_TRIALS  2

struct args_struct {
  bool help_flag;
  bool version_flag;
  bool list_benchmarks_flag;
  bool list_gpus_flag;
  int n_trials;
  int n_warmup_trials;
  char* benchmark_name;
  device_type device;

  int n_threads;
  int nbk;
  int tpb;
  int gpu_idx;
  struct config* cfg;
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
      printf("Overflow detected while parsing the arguments\n");
      break;
    case UNDERFLOW:
      printf("Underflow detected while parsing the arguments\n");
      break;
    case INVALID_ARG:
      printf("Invalid argument\n");
      break;
    default:
      printf("Invalid error: %d\n", errn);
      break;
  }
}

char * build_short_options() {
  const char *c = args_chr;
  int len = sizeof(args_chr) / sizeof(args_chr[0]);
  char* str = (char *) malloc(sizeof(char) * (len*2 + 1));
  memset(str, 0, sizeof(char) * (len*2 + 1));

  sprintf(str, "%c%c:%c:%c:%c:%c:%c:%c:%c%c:%c%c",
  c[ARG_LISTBENCHS], c[ARG_BENCHMARK], c[ARG_DEVICE],
  c[ARG_TRIALS], c[ARG_WARMUP], c[ARG_CPU_THREADS],
  c[ARG_GPU_BLOCKS], c[ARG_GPU_TPB], c[ARG_GPU_LIST],
  c[ARG_GPU_IDX], c[ARG_HELP], c[ARG_VERSION]);

  return str;
}

bool parseArgs(int argc, char* argv[]) {
  int opt;
  int option_index = 0;
  opterr = 0;

  bool n_threads_set = false;
  bool n_blocks_set = false;
  bool n_threads_per_block_set = false;
  bool gpu_idx_set = false;

  args.help_flag = false;
  args.version_flag = false;
  args.list_benchmarks_flag = false;
  args.list_gpus_flag = false;
  args.n_trials = DEFAULT_N_TRIALS;
  args.n_warmup_trials = DEFAULT_WARMUP_TRIALS;
  args.device = DEVICE_TYPE_CPU;
  args.benchmark_name = NULL;

  args.n_threads = INVALID_CFG;
  args.tpb = INVALID_CFG;
  args.nbk = INVALID_CFG;
  args.gpu_idx = 0;
  args.cfg = (struct config *) malloc(sizeof(struct config));

  constexpr char *c = (char *) args_chr;

  static struct option long_options[] = {
    {args_str[ARG_LISTBENCHS],  no_argument,       0, c[ARG_LISTBENCHS]  },
    {args_str[ARG_BENCHMARK],   required_argument, 0, c[ARG_BENCHMARK]   },
    {args_str[ARG_DEVICE],      required_argument, 0, c[ARG_DEVICE]      },
    {args_str[ARG_TRIALS],      required_argument, 0, c[ARG_TRIALS]      },
    {args_str[ARG_WARMUP],      required_argument, 0, c[ARG_WARMUP]      },
    {args_str[ARG_CPU_THREADS], required_argument, 0, c[ARG_CPU_THREADS] },
    {args_str[ARG_GPU_BLOCKS],  required_argument, 0, c[ARG_GPU_BLOCKS]  },
    {args_str[ARG_GPU_TPB],     required_argument, 0, c[ARG_GPU_TPB]     },
    {args_str[ARG_GPU_LIST],    no_argument,       0, c[ARG_GPU_LIST]    },
    {args_str[ARG_GPU_IDX],     required_argument, 0, c[ARG_GPU_IDX]     },
    {args_str[ARG_HELP],        no_argument,       0, c[ARG_HELP]        },
    {args_str[ARG_VERSION],     no_argument,       0, c[ARG_VERSION]     },
    {0, 0, 0, 0}
  };

  char* short_options = build_short_options();
  opt = getopt_long(argc, argv, short_options, long_options, &option_index);

  while(opt != -1) {
    switch (opt) {
      case c[ARG_HELP]:
        args.help_flag  = true;
        break;

      case c[ARG_VERSION]:
        args.version_flag  = true;
        break;

      case c[ARG_DEVICE]:
        args.device  = parse_device_type(optarg);
        if(args.device == DEVICE_TYPE_INVALID) {
          printErr("Invalid device: '%s'", args_str[ARG_DEVICE]);
        }
        break;

      case c[ARG_LISTBENCHS]:
        args.list_benchmarks_flag  = true;
        break;

      case c[ARG_GPU_LIST]:
        args.list_gpus_flag  = true;
        break;

      case c[ARG_TRIALS]:
        args.n_trials = getarg_int(optarg);
        if(errn != 0) {
          printErr("Option %s: ", args_str[ARG_TRIALS]);
          printerror();
          args.help_flag  = true;
          return false;
        }
        break;

      case c[ARG_WARMUP]:
        args.n_warmup_trials = getarg_int(optarg);
        if(errn != 0) {
          printErr("Option %s: ", args_str[ARG_WARMUP]);
          printerror();
          args.help_flag  = true;
          return false;
        }
        break;

      case c[ARG_CPU_THREADS]:
        n_threads_set = true;
        args.n_threads = getarg_int(optarg);
        if(errn != 0) {
          printErr("Option %s: ", args_str[ARG_CPU_THREADS]);
          printerror();
          args.help_flag  = true;
          return false;
        }
        break;

      case c[ARG_GPU_BLOCKS]:
        n_blocks_set = true;
        args.nbk = getarg_int(optarg);
        if(errn != 0) {
          printErr("Option %s: ", args_str[ARG_GPU_BLOCKS]);
          printerror();
          args.help_flag  = true;
          return false;
        }
        break;

      case c[ARG_GPU_TPB]:
        n_threads_per_block_set = true;
        args.tpb = getarg_int(optarg);
        if(errn != 0) {
          printErr("Option %s: ", args_str[ARG_GPU_TPB]);
          printerror();
          args.help_flag  = true;
          return false;
        }
        break;

      case c[ARG_GPU_IDX]:
        gpu_idx_set = true;
        args.gpu_idx = getarg_int(optarg);
        if(errn != 0) {
          printErr("Option %s: ", args_str[ARG_GPU_IDX]);
          printerror();
          args.help_flag  = true;
          return false;
        }
        break;

      case c[ARG_BENCHMARK]:
        args.benchmark_name = (char *) malloc(sizeof(char) * (strlen(optarg) + 1));
        strcpy(args.benchmark_name, optarg);
        break;

      default:
        printf("WARNING: Invalid options\n");
        args.help_flag  = true;
        return true;
    }

    option_index = 0;
    opt = getopt_long(argc, argv, short_options, long_options, &option_index);
  }
  free(short_options);

  // check args //
  if(args.n_trials <= 0) {
    printErr("Number of trials must be greater than zero");
    return false;
  }
  if(args.n_warmup_trials < 0) {
    printErr("Number of warmup trials must be greater or equal to zero");
    return false;
  }
  if(args.device == DEVICE_TYPE_CPU) {
    if(n_threads_set && args.n_threads <= 0) {
      printErr("Number of threads must be greater than zero");
      return false;
    }
    if(n_blocks_set) {
      printErr("Option %s is only available in GPU mode", args_str[ARG_GPU_BLOCKS]);
      return false;
    }
    if(n_threads_per_block_set) {
      printErr("Option %s is only available in GPU mode", args_str[ARG_GPU_TPB]);
      return false;
    }
    if(args.list_gpus_flag) {
      printErr("Option %s is only available in GPU mode", args_str[ARG_GPU_LIST]);
      return false;
    }
    if(gpu_idx_set) {
      printErr("Option %s is only available in GPU mode", args_str[ARG_GPU_IDX]);
      return false;
    }
  }
  else {
    if(args.benchmark_name != NULL) {
      printErr("Option %s is only available in CPU mode", args_str[ARG_BENCHMARK]);
      return false;
    }
    if(args.list_benchmarks_flag) {
      printErr("Option %s is only available in CPU mode", args_str[ARG_LISTBENCHS]);
      return false;
    }
    if(n_threads_set) {
      printErr("Option %s is only available in CPU mode", args_str[ARG_CPU_THREADS]);
      return false;
    }
    if(n_blocks_set && args.nbk <= 0) {
      printErr("Number of blocks must be greater than zero");
      return false;
    }
    if(n_threads_per_block_set && args.tpb <= 0) {
      printErr("Number of threads per block must be greater than zero");
      return false;
    }
  }

  args.cfg->n_threads = args.n_threads;
  args.cfg->tpb = args.tpb;
  args.cfg->nbk = args.nbk;
  args.cfg->gpu_idx = args.gpu_idx;

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

bool list_gpus() {
  return args.list_gpus_flag;
}

int get_n_trials() {
  return args.n_trials;
}

int get_warmup_trials() {
  return args.n_warmup_trials;
}

char* get_benchmark_str_args() {
  return args.benchmark_name;
}

device_type get_device_type() {
  return args.device;
}

struct config* get_config() {
  return args.cfg;
}
