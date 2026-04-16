#ifndef NEON_6_HPP
#define NEON_6_HPP

#ifdef ARCH_ARM
  #include <arm_neon.h>
  void compute_neon_6(float32x4_t *farr, float32x4_t mult, int index);
#endif

#endif
