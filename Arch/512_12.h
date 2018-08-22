#ifndef __512_12__
#define __512_12__

#define MAXFLOPS_ITERS 1000000000
#define OP_PER_IT 12

#include "Arch.h"

/***

SIZE=2*(6/0.5)

2   -> We have to add two operands in a add instruction
6   -> FMA latency
0.5 -> FMA CPI

Used by:
-KNL

***/

#endif
