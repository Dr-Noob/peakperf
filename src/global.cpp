#include <stdarg.h>
#include <stdio.h>
#include "global.hpp"

void printErr(const char *fmt, ...) {
  const int buffer_size = 4096;
  char buffer[buffer_size];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer,buffer_size, fmt, args);
  va_end(args);
  fprintf(stderr, RED "[ERROR]: " RESET "%s\n",buffer);
}

int max(int a, int b) {
  return a > b ? a : b;
}
