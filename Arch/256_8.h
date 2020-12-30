#ifndef __256_8__
#define __256_8__

#include "Arch.h"

double get_gflops_256_8(int n_threads);
void compute_256_8(TYPE *farr_ptr, TYPE mult, int index);

#endif
