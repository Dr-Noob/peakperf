#ifndef __GLOBAL__
#define __GLOBAL__

#define RED   "\x1b[31;1m"
#define BOLD  "\x1b[1m"
#define GREEN "\x1b[42m"
#define RESET "\x1b[0m"

void printErr(const char *fmt, ...);
int max(int a, int b);

#endif
