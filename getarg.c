#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include <stdio.h>
#include "getarg.h"

#define OVERFLOW          -1
#define UNDERFLOW         -2
#define INVALID_ARG       -3

int errn = 0;

int getArgc(char* argv[]) {
  int i = 1;
  while(argv[i] != NULL)i++;
  return i-1;
}

int getarg_int(int index, char* argv[]) {
  //Clean errn
  errn = 0;

  int argc = getArgc(argv);

  assert(index <= argc);
  assert(argv[index] != NULL);

  char* endptr;
  long tmp = strtol(argv[index], &endptr, 10);

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
    int ret = (int)tmp;
    return ret;
  }
  errn = OVERFLOW;
  return -1;
}

void printerror() {
  switch (errn) {
    case OVERFLOW:
      puts("Overflow detected");
      break;
    case UNDERFLOW:
      puts("Underflow detected");
      break;
    case INVALID_ARG:
      puts("Invalid arg");
      break;
  }
}
