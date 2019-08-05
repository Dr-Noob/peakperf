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

#define ARG_CHAR_HELP     'h'
#define ARG_CHAR_TRIALS   'r'
#define ARG_CHAR_WARMUP   'w'
#define ARG_CHAR_THREADS  't'

#define DEFAULT_N_TRIALS      10
#define DEFAULT_WARMUP_TRIALS  2

struct args_struct {
  bool help_flag;
  int n_trials;
  int n_warmup_trials;
  int n_threads;
};

int errn = 0;
static struct args_struct args;

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
  args.n_trials = DEFAULT_N_TRIALS;
  args.n_warmup_trials = DEFAULT_WARMUP_TRIALS;
  args.n_threads = INVALID_N_THREADS;

  while ((opt = getopt(argc, argv, "hr:w:t:")) != -1) {
    switch (opt) {
    case 'h':
      args.help_flag  = true;
      break;
    
    case 'r':
      args.n_trials = getarg_int(optarg);
      if(errn != 0) {
        printf("ERROR: Option -r: ");
        printerror();
        args.help_flag  = true;
        return false;
      }
      break;
      
    case 'w':
      args.n_warmup_trials = getarg_int(optarg);
      if(errn != 0) {
        printf("ERROR: Option -w: ");
        printerror();
        args.help_flag  = true;    
        return false;
      }
      break;
      
    case 't':
      n_threads_set = true;
      args.n_threads = getarg_int(optarg);
      if(errn != 0) {
        printf("ERROR: Option -t: ");
        printerror();
        args.help_flag  = true;   
        return false;
      }
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

int get_n_trials() {
  return args.n_trials;
}

int get_warmup_trials() {
  return args.n_warmup_trials;
}

int get_n_threads() {
  return args.n_threads;    
}

