#ifndef __256_10__
#define __256_10__
#include "Arch.h"

#define MAXFLOPS_ITERS 1000000000
#define BYTES_IN_VECT 32
#define TYPE __m256

#define OP_PER_IT 10
#define SIZE 20
#define FLOPS_ARRAY_SIZE N_THREADS*SIZE

/***

SIZE=2*(5/0.5)

2   -> We have to add two operands in a add instruction
5   -> FMA latency
0.5 -> FMA CPI

Used by:
-Haswell
-Zen
-Zen+

***/

#endif
