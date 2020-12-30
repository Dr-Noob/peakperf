#ifndef __ARCH__
#define __ARCH__

#include <stdio.h>
#include <immintrin.h>
#include "../cpufetch/uarch.h"

#define MAX_NUMBER_THREADS 512

#ifdef AVX_512_12
  #include "512_12.h"
#elif defined AVX_256_10
  #include "256_10.h"
#elif defined AVX_256_8
  #include "256_8.h"
#elif defined AVX_512_8
  #include "512_8.h"  
#elif defined AVX_256_5
  #include "256_5.h"  
#elif defined AVX_256_3_NOFMA
  #include "256_3_nofma.h"
#endif

#ifdef BUILDING_OBJECT
#include <omp.h>
#define SIZE OP_PER_IT*2
#endif

#if defined(AVX_512_12) || defined(AVX_512_8)
  #define BYTES_IN_VECT 64
  #define TYPE __m512
#elif defined(BUILDING_OBJECT)
  #define BYTES_IN_VECT 32
  #define TYPE __m256  
#endif

#if defined(AVX_512_12) || defined(AVX_512_8)
  #define BYTES_IN_VECT 64
  #define TYPE __m512

  static void initialize(int n_threads, TYPE mult, TYPE farr[][SIZE], float *fa) {
    mult = _mm512_set1_ps(0.1f);

    for(int i=0;i<n_threads;i++) {
      for(int j=0;j<SIZE;j++)
        farr[i][j] = _mm512_set_ps (fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],
                                    fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j]);
    }
  }
#elif defined(BUILDING_OBJECT)
  #define BYTES_IN_VECT 32
  #define TYPE __m256
  
  static void initialize(int n_threads, TYPE mult, TYPE farr[][SIZE], float *fa) {
    mult = _mm256_set1_ps(0.1f);
    
    for(int i=0;i<n_threads;i++)
      for(int j=0;j<SIZE;j++)
        farr[i][j] = _mm256_set_ps (fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j],fa[i*SIZE+j]);
  }
#endif

#ifdef BUILDING_OBJECT
/*
void (*get_compute_function(struct cpu* cpu))(TYPE [][SIZE], TYPE, int) {
  struct uarch* uarch_struct = get_uarch_struct(cpu);
  MICROARCH u = uarch_struct->uarch;
  
  switch(u) {
    case UARCH_SANDY_BRIDGE:
      printf("UARCH_SANDY_BRIDGE");
      break;
    case UARCH_KABY_LAKE:
      printf("UARCH_KABY_LAKE");
      break;
  }
  
  return NULL;
}*/
#endif


#define MAXFLOPS_ITERS 1000000000

struct benchmark;

struct benchmark* init_benchmark(struct cpu* cpu, int n_threads);
void compute(struct benchmark* bench);
double get_gflops(struct benchmark* bench);

#endif
