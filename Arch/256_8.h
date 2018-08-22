#ifndef __256_8__
#define __256_8__
#include "Arch.h"

#define MAXFLOPS_ITERS 1000000000
#define BYTES_IN_VECT 32
#define TYPE __m256

#define OP_PER_IT 8
#define SIZE 16
#define FLOPS_ARRAY_SIZE N_THREADS*SIZE

/***

SIZE=2*(4/0.5)

2   -> We have to add two operands in a add instruction
5   -> 4 latency
0.5 -> FMA CPI

Used by:
-Skylake

***/

#endif
