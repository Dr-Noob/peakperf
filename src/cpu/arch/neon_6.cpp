#include "../../global.hpp"
#include "arch.hpp"
#include "neon_6.hpp"

#ifdef ARCH_ARM

#define OP_PER_IT B_NEON_6_OP_IT

float32x4_t farr_neon_6[MAX_NUMBER_THREADS][6] __attribute__((aligned(64)));

void compute_neon_6(float32x4_t *farr, float32x4_t mult, int index) {
  farr = farr_neon_6[index];

  for(long i=0; i < BENCHMARK_CPU_ITERS; i++) {
    farr[0] = vfmaq_f32(farr[0], mult, farr[1]);
    farr[1] = vfmaq_f32(farr[1], mult, farr[2]);
    farr[2] = vfmaq_f32(farr[2], mult, farr[3]);
    farr[3] = vfmaq_f32(farr[3], mult, farr[4]);
    farr[4] = vfmaq_f32(farr[4], mult, farr[5]);
    farr[5] = vfmaq_f32(farr[5], mult, farr[0]);
  }
}
#endif
