#ifndef __256_3_NOFMA__
#define __256_3_NOFMA__

#include "Arch.h"

double get_gflops_256_3_nofma(int n_threads);
void compute_256_3_nofma(TYPE *farr_ptr, TYPE mult, int index);

#endif
