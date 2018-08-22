#ifndef __512_12__
#define __512_12__
#include "Arch.h"

#define MAXFLOPS_ITERS 1000000000
#define BYTES_IN_VECT 64
#define TYPE __m512

#define OP_PER_IT 12
#define SIZE 24
#define FLOPS_ARRAY_SIZE N_THREADS*SIZE

/***

SIZE=2*(6/0.5)

2   -> We have to add two operands in a add instruction
6   -> FMA latency
0.5 -> FMA CPI

Used by:
-KNL

***/

#endif
