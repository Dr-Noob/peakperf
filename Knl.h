#ifndef __NKL__
#define __NKL__

#include <immintrin.h>
#define MAXFLOPS_ITERS 1000000000
#define BYTES_IN_VECT 64
#define TYPE __m512

#define OP_PER_IT 12
#define SIZE 25
/***

2*(6/0.5)+1

2   -> We have to add two operands in a add instruction
6   -> KNL FMA latency
0.5 -> KNL FMA CPI
1   -> Extra space for padding

***/

#define FLOPS_ARRAY_SIZE N_THREADS*SIZE

void compute(int index);
void initialize(float fa[FLOPS_ARRAY_SIZE]);
float summarize();

#endif
