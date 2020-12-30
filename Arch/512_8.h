#ifndef __512_8__
#define __512_8__

#include "Arch.h"

double get_gflops_512_8(int n_threads);
void compute_512_8(TYPE *farr_ptr, TYPE mult, int index);

#endif
