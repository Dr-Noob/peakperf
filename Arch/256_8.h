#ifndef __256_8__
#define __256_8__

#define MAXFLOPS_ITERS 1000000000
#define OP_PER_IT 8

#include "Arch.h"

/***

SIZE=2*(4/0.5)

2   -> We have to add two operands in a add instruction
4   -> 4 latency
0.5 -> FMA CPI

Used by:
-Skylake

***/

#endif
