#ifndef __KNL__
#define __KNL__

#include "arch.hpp"

void compute_knl(__m512 *farr_ptr, __m512 mult, int index);

#endif
