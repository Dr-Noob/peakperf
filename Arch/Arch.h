#ifndef __ARCH__
#define __ARCH__

#include <immintrin.h>
#include "../cpufetch/uarch.h"

#ifdef AVX_512_12
  #include "512_12.h"
#elif defined AVX_512_8
  #include "512_8.h"    
#elif defined AVX_256_10
  #include "256_10.h"
#elif defined AVX_256_8
  #include "256_8.h"
#elif defined AVX_256_5
  #include "256_5.h"  
#elif defined AVX_256_3_NOFMA
  #include "256_3_nofma.h"
#endif

#define MAXFLOPS_ITERS 1000000000
#define MAX_NUMBER_THREADS 512

#if defined(AVX_512_12) || defined(AVX_512_8)
  #define BYTES_IN_VECT 64
  #define TYPE __m512
  #define SIZE OP_PER_IT*2
#elif defined(AVX_256_10) || defined(AVX_256_8) || defined(AVX_256_5) || defined(AVX_256_3_NOFMA) 
  #define BYTES_IN_VECT 32
  #define TYPE __m256
  #define SIZE OP_PER_IT*2
#endif

struct benchmark;

struct benchmark* init_benchmark(struct cpu* cpu, int n_threads);
void compute(struct benchmark* bench);
double get_gflops(struct benchmark* bench);
char* get_benchmark_name(struct benchmark* bench);

#endif
