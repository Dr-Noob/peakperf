#ifndef __DATA__
#define __DATA__

#ifdef AVX_256_3_NOFMA
  TYPE farr_256_3_NOFMA[MAX_NUMBER_THREADS][SIZE] __attribute__((aligned(64)));  
#elif defined AVX_256_5
  TYPE farr_256_5[MAX_NUMBER_THREADS][SIZE] __attribute__((aligned(64)));  
#elif defined AVX_256_8
  TYPE farr_256_8[MAX_NUMBER_THREADS][SIZE] __attribute__((aligned(64)));  
  #elif defined AVX_256_10
  TYPE farr_256_10[MAX_NUMBER_THREADS][SIZE] __attribute__((aligned(64)));  
#elif defined AVX_512_8
  TYPE farr_512_8[MAX_NUMBER_THREADS][SIZE] __attribute__((aligned(64)));  
#elif AVX_512_12
  TYPE farr_512_12[MAX_NUMBER_THREADS][SIZE] __attribute__((aligned(64)));  
#endif

#endif
