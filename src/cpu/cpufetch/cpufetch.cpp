/*
 * Code obtained from cpufetch project at
 * https://github.com/Dr-Noob/cpufetch
 */

#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "../../global.hpp"
#include "cpuid.hpp"
#include "uarch.hpp"

#define CPU_VENDOR_INTEL_STRING "GenuineIntel"
#define CPU_VENDOR_AMD_STRING   "AuthenticAMD"
#define MASK 0xFF

typedef int32_t VENDOR;

enum {
  CPU_VENDOR_UNKNOWN,
  CPU_VENDOR_INTEL,
  CPU_VENDOR_AMD,
};

enum {
  CORE_TYPE_EFFICIENCY,
  CORE_TYPE_PERFORMANCE,
  CORE_TYPE_UNKNOWN
};

struct cpu {
  VENDOR cpu_vendor;  
  char* cpu_name; 
  struct uarch* uarch;
  bool avx;
  bool avx2;
  bool fma;
  bool avx512;
  bool hybrid_flag;
  struct hybrid_topology* h_topo;
};

int32_t get_core_type(void) {
#if defined(__x86_64__) || defined(__i386__)
  uint32_t eax = 0x0000001A;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;

  eax = 0x0000001A;
  cpuid(&eax, &ebx, &ecx, &edx);

  int32_t type = eax >> 24 & 0xFF;
  if(type == 0x20) return CORE_TYPE_EFFICIENCY;
  else if(type == 0x40) return CORE_TYPE_PERFORMANCE;
  else {
    printErr("Found invalid core type: 0x%.8X\n", type);
    return CORE_TYPE_UNKNOWN;
  }
#else
  return CORE_TYPE_PERFORMANCE;
#endif
}

bool bind_to_cpu(int cpu_id) {
  cpu_set_t currentCPU;
  CPU_ZERO(&currentCPU);
  CPU_SET(cpu_id, &currentCPU);
  if (sched_setaffinity (0, sizeof(currentCPU), &currentCPU) == -1) {
    return false;
  }
  return true;
}

struct hybrid_topology* get_hybrid_topology_internal(struct cpu* cpu) {
  (void)cpu;
  int ncores;
  if((ncores = sysconf(_SC_NPROCESSORS_ONLN)) == -1) {
    printErr("sysconf(_SC_NPROCESSORS_ONLN): %s", strerror(errno));
    return NULL;
  }

  struct hybrid_topology* h_topo = (struct hybrid_topology*) malloc(sizeof(struct hybrid_topology));
  h_topo->e_cores = 0;
  h_topo->p_cores = 0;
  h_topo->core_mask = (bool *) malloc(sizeof(bool) * ncores);

  int i=0;
  bool invalid_core = false;

  while(!invalid_core) {
    if(bind_to_cpu(i)) {
      int32_t core_type = get_core_type();
      if(core_type == CORE_TYPE_PERFORMANCE) {
        h_topo->p_cores++;
        h_topo->core_mask[i] = true;
      }
      else if(core_type == CORE_TYPE_EFFICIENCY) {
        h_topo->e_cores++;
        h_topo->core_mask[i] = false;
      }
      else {
        printErr("Found invalid core type");
        return NULL;
      }
      i++;
    }
    else {
      invalid_core = true;
    }
  }

  if(i == 0) {
    printErr("Unable to bind to core");
    return NULL;
  }
  return h_topo;
}

void fill_features_cpuid(struct cpu* cpu) {
#if defined(__x86_64__) || defined(__i386__)
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;

  
  cpuid(&eax, &ebx, &ecx, &edx);
  uint32_t maxLevels = eax;
  
  if (maxLevels >= 0x00000001){
    eax = 0x00000001;
    cpuid(&eax, &ebx, &ecx, &edx);
    cpu->avx = (ecx & ((int)1 << 28)) != 0;
    cpu->fma = (ecx & ((int)1 << 12)) != 0;
  }
  else {
    cpu->avx = false;
    cpu->fma = false;
    printErr("Could not determine CPU features");
  }
  
  if (maxLevels >= 0x00000007){
    eax = 0x00000007;
    ecx = 0x00000000;
    cpuid(&eax, &ebx, &ecx, &edx);
    cpu->avx2         = (ebx & ((int)1 <<  5)) != 0;
    cpu->avx512       = (((ebx & ((int)1 << 16)) != 0)  ||
                         ((ebx & ((int)1 << 28)) != 0)  ||
                         ((ebx & ((int)1 << 26)) != 0)  ||
                         ((ebx & ((int)1 << 27)) != 0)  ||
                         ((ebx & ((int)1 << 31)) != 0)  ||
                         ((ebx & ((int)1 << 30)) != 0)  ||
                         ((ebx & ((int)1 << 17)) != 0)  ||
                         ((ebx & ((int)1 << 21)) != 0));
  }
  else {
    cpu->avx2 = false;
    cpu->avx512 = false;
  }

  cpu->h_topo = NULL;
  cpu->hybrid_flag = false;
  if(cpu->cpu_vendor == CPU_VENDOR_INTEL && maxLevels >= 0x00000007) {
    eax = 0x00000007;
    ecx = 0x00000000;
    cpuid(&eax, &ebx, &ecx, &edx);
    cpu->hybrid_flag = (edx >> 15) & 0x1;
    if(cpu->hybrid_flag) {
      cpu->h_topo = get_hybrid_topology_internal(cpu);
    }
  }
#else
  cpu->avx = false;
  cpu->avx2 = false;
  cpu->fma = false;
  cpu->avx512 = false;
  cpu->h_topo = NULL;
  cpu->hybrid_flag = false;
#endif
}

void get_name_cpuid(char* name, uint32_t reg1, uint32_t reg2, uint32_t reg3) {
  uint32_t c = 0;

  name[c++] = reg1       & MASK;
  name[c++] = (reg1>>8)  & MASK;
  name[c++] = (reg1>>16) & MASK;
  name[c++] = (reg1>>24) & MASK;

  name[c++] = reg2       & MASK;
  name[c++] = (reg2>>8)  & MASK;
  name[c++] = (reg2>>16) & MASK;
  name[c++] = (reg2>>24) & MASK;

  name[c++] = reg3       & MASK;
  name[c++] = (reg3>>8)  & MASK;
  name[c++] = (reg3>>16) & MASK;
  name[c++] = (reg3>>24) & MASK;
}

VENDOR cpu_vendor() {
#if defined(__x86_64__) || defined(__i386__)
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;

  cpuid(&eax, &ebx, &ecx, &edx);
  
  char name[13];
  memset(name,0,13);
  get_name_cpuid(name, ebx, edx, ecx);
  
  if(strcmp(CPU_VENDOR_INTEL_STRING,name) == 0)
    return CPU_VENDOR_INTEL;
  else if (strcmp(CPU_VENDOR_AMD_STRING,name) == 0)
    return CPU_VENDOR_AMD;  
  else {
    printf("Unknown CPU vendor: %s", name);
    return CPU_VENDOR_UNKNOWN;
  }    
#else
  return CPU_VENDOR_UNKNOWN;
#endif
}

char* cpu_name() {
#if defined(__x86_64__) || defined(__i386__)
  unsigned eax = 0;
  unsigned ebx = 0;
  unsigned ecx = 0;
  unsigned edx = 0;
  
  char name[64];
  memset(name,0,64);

  //First, check we can use extended
  eax = 0x80000000;
  cpuid(&eax, &ebx, &ecx, &edx);
  if(eax < 0x80000001) {
    char* none = (char*) malloc(sizeof(char)*64);
    sprintf(none,"Unknown");
    return none;
  }

  //We can, fetch name
  eax = 0x80000002;
  cpuid(&eax, &ebx, &ecx, &edx);

  name[0] = eax       & MASK;
  name[1] = (eax>>8)  & MASK;
  name[2] = (eax>>16) & MASK;
  name[3] = (eax>>24) & MASK;
  name[4] = ebx       & MASK;
  name[5] = (ebx>>8)  & MASK;
  name[6] = (ebx>>16) & MASK;
  name[7] = (ebx>>24) & MASK;
  name[8] = ecx       & MASK;
  name[9] = (ecx>>8)  & MASK;
  name[10] = (ecx>>16) & MASK;
  name[11] = (ecx>>24) & MASK;
  name[12] = edx       & MASK;
  name[13] = (edx>>8)  & MASK;
  name[14] = (edx>>16) & MASK;
  name[15] = (edx>>24) & MASK;

  eax = 0x80000003;
  cpuid(&eax, &ebx, &ecx, &edx);

  name[16] = eax       & MASK;
  name[17] = (eax>>8)  & MASK;
  name[18] = (eax>>16) & MASK;
  name[19] = (eax>>24) & MASK;
  name[20] = ebx       & MASK;
  name[21] = (ebx>>8)  & MASK;
  name[22] = (ebx>>16) & MASK;
  name[23] = (ebx>>24) & MASK;
  name[24] = ecx       & MASK;
  name[25] = (ecx>>8)  & MASK;
  name[26] = (ecx>>16) & MASK;
  name[27] = (ecx>>24) & MASK;
  name[28] = edx       & MASK;
  name[29] = (edx>>8)  & MASK;
  name[30] = (edx>>16) & MASK;
  name[31] = (edx>>24) & MASK;

  eax = 0x80000004;
  cpuid(&eax, &ebx, &ecx, &edx);

  name[32] = eax       & MASK;
  name[33] = (eax>>8)  & MASK;
  name[34] = (eax>>16) & MASK;
  name[35] = (eax>>24) & MASK;
  name[36] = ebx       & MASK;
  name[37] = (ebx>>8)  & MASK;
  name[38] = (ebx>>16) & MASK;
  name[39] = (ebx>>24) & MASK;
  name[40] = ecx       & MASK;
  name[41] = (ecx>>8)  & MASK;
  name[42] = (ecx>>16) & MASK;
  name[43] = (ecx>>24) & MASK;
  name[44] = edx       & MASK;
  name[45] = (edx>>8)  & MASK;
  name[46] = (edx>>16) & MASK;
  name[47] = (edx>>24) & MASK;

  name[48] = '\0';

  //Remove unused characters
  int i = 0;
  while(name[i] == ' ')i++;

  char* name_withoutblank = (char *) malloc(sizeof(char)*64);
  strcpy(name_withoutblank,name+i);
  return name_withoutblank;
#else
  char* name = (char*) malloc(sizeof(char)*64);
  sprintf(name,"Unknown CPU");
  return name;
#endif
}

bool is_cpu_intel(struct cpu* cpu) {
  return cpu->cpu_vendor == CPU_VENDOR_INTEL;    
}

bool is_cpu_amd(struct cpu* cpu) {
  return cpu->cpu_vendor == CPU_VENDOR_AMD;  
}

bool is_hybrid_cpu(struct cpu* cpu) {
  return cpu->hybrid_flag;
}

bool cpu_has_avx(struct cpu* cpu) {
  return cpu->avx;    
}

bool cpu_has_avx2(struct cpu* cpu) {
  return cpu->avx2;    
}

bool cpu_has_fma(struct cpu* cpu) {
  return cpu->fma;    
}

bool cpu_has_avx512(struct cpu* cpu) {
  return cpu->avx512;    
}

const char* get_str_uarch(struct cpu* cpu) {
  return cpu->uarch->uarch_str;
}

struct cpu* get_cpu_info() {
  struct cpu* cpu = (struct cpu*) malloc(sizeof(struct cpu));
  memset(cpu, 0, sizeof(struct cpu));
  
  cpu->cpu_name = cpu_name();
  cpu->cpu_vendor = cpu_vendor();
  cpu->uarch = get_uarch(cpu);
  fill_features_cpuid(cpu);
  
  return cpu;
}

void free_cpu_info(struct cpu* cpu) {
  if (cpu == NULL) return;
  if (cpu->cpu_name) free(cpu->cpu_name);
  if (cpu->uarch) free_uarch_struct(cpu->uarch);
  if (cpu->h_topo) {
    if (cpu->h_topo->core_mask) free(cpu->h_topo->core_mask);
    free(cpu->h_topo);
  }
  free(cpu);
}

char* get_str_cpu_name(struct cpu* cpu) {
  return cpu->cpu_name;    
}

struct uarch* get_uarch_struct(struct cpu* cpu) {
  return cpu->uarch;    
}

struct hybrid_topology* get_hybrid_topology(struct cpu* cpu) {
  return cpu->h_topo;
}

bool is_performance_core(struct hybrid_topology* h_topo, int tid) {
  return h_topo->core_mask[tid];
}
