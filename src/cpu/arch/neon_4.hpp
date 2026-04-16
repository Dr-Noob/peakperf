#ifndef NEON_4_HPP
#define NEON_4_HPP

#ifdef ARCH_ARM
  #include <arm_neon.h>
  void compute_neon_4(float32x4_t *farr, float32x4_t mult, int index);
#endif

#endif
