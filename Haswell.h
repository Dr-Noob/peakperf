#ifndef __HASWELL__
#define __HASWELL__

#include <immintrin.h>
#define MAXFLOPS_ITERS 1000000000
#define BYTES_IN_VECT 32
#define TYPE __m256
#define TEST_NAME "Haswell-256 bits"

#define OP_PER_IT 10
#define SIZE 20
#define FLOPS_ARRAY_SIZE N_THREADS*SIZE

/***
2*(5/0.5)

2   -> We have to add two operands in a add instruction
5   -> Haswell FMA latency
0.5 -> Haswell FMA CPI
***/

void compute(int index);
void initialize(float fa[FLOPS_ARRAY_SIZE]);
float summarize();

#endif
