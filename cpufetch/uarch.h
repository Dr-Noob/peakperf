#ifndef __UARCH__
#define __UARCH__

#include <stdint.h>

#include "cpu.h"

typedef uint32_t MICROARCH;

struct uarch {
  MICROARCH uarch;
  char* uarch_str;
};

struct uarch* get_uarch();
char* get_str_uarch(struct cpu* cpu);
void free_uarch_struct(struct uarch* arch);

#endif
