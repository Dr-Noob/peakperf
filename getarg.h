#ifndef __GETARG__
#define __GETARG__

extern int errn;

/***
Returns a int representing arg at the passed index
If index is out of bounts, returns INVALID_INDEX
If arg is not a number, returns INVALID_ARG
If integer is out of limits, returns OVERFLOW, or UNDERFLOW
***/

int getarg_int(int index, char* argv[]);

/***
Prints the error
***/

void printerror();

#endif
