#ifndef __CPU__
#define __CPU__

#include <stdbool.h>
#include <stdint.h>

struct cpu;

struct hybrid_topology {
  uint32_t e_cores;
  uint32_t p_cores;
  bool* core_mask;
};

struct cpu* get_cpu_info();

bool is_cpu_intel(struct cpu* cpu);
bool is_cpu_amd(struct cpu* cpu);
bool is_hybrid_cpu(struct cpu* cpu);
bool cpu_has_avx(struct cpu* cpu);
bool cpu_has_avx2(struct cpu* cpu);
bool cpu_has_fma(struct cpu* cpu);
bool cpu_has_avx512(struct cpu* cpu);
char* get_str_cpu_name(struct cpu* cpu);
struct uarch* get_uarch_struct(struct cpu* cpu);
struct hybrid_topology* get_hybrid_topology(struct cpu* cpu);
bool is_performance_core(struct hybrid_topology* h_topo, int tid);

#endif
