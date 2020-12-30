#ifndef __256_10__
#define __256_10__

#ifdef BUILDING_OBJECT
  #define OP_PER_IT 10
  #define FMA_AVAILABLE 2
#endif

#include "Arch.h"

void compute_256_10(TYPE *farr_ptr, TYPE mult, int index);

#endif
