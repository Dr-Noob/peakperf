#ifndef __256_8__
#define __256_8__

#ifdef BUILDING_OBJECT
  #define OP_PER_IT 8
  #define FMA_AVAILABLE 2
  #define SIZE OP_PER_IT*2
#endif

#include "Arch.h"

void compute_256_8(TYPE *farr_ptr, TYPE mult, int index);

#endif
