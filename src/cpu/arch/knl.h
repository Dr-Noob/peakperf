#ifndef __KNL__
#define __KNL__

#include "arch.h"

void compute_knl(__m512 *farr_ptr, __m512 mult, int index);

#endif
