#ifndef __CPU__
#define __CPU__

#include <stdbool.h>

struct cpu;

struct cpu* get_cpu_info();

bool is_cpu_intel(struct cpu* cpu);
bool is_cpu_amd(struct cpu* cpu);
char* get_str_cpu_name(struct cpu* cpu);

#endif
