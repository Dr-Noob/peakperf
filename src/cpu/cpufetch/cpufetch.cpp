/*
 * Code obtained from cpufetch project at
 * https://github.com/Dr-Noob/cpufetch
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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

struct cpu {
  VENDOR cpu_vendor;  
  char* cpu_name; 
  struct uarch* uarch;
  bool avx;
  bool avx2;
  bool fma;
  bool avx512;
};

void fill_features_cpuid(struct cpu* cpu) {
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
}

char* cpu_name() {
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

  name[__COUNTER__] = eax       & MASK;
  name[__COUNTER__] = (eax>>8)  & MASK;
  name[__COUNTER__] = (eax>>16) & MASK;
  name[__COUNTER__] = (eax>>24) & MASK;
  name[__COUNTER__] = ebx       & MASK;
  name[__COUNTER__] = (ebx>>8)  & MASK;
  name[__COUNTER__] = (ebx>>16) & MASK;
  name[__COUNTER__] = (ebx>>24) & MASK;
  name[__COUNTER__] = ecx       & MASK;
  name[__COUNTER__] = (ecx>>8)  & MASK;
  name[__COUNTER__] = (ecx>>16) & MASK;
  name[__COUNTER__] = (ecx>>24) & MASK;
  name[__COUNTER__] = edx       & MASK;
  name[__COUNTER__] = (edx>>8)  & MASK;
  name[__COUNTER__] = (edx>>16) & MASK;
  name[__COUNTER__] = (edx>>24) & MASK;

  eax = 0x80000003;
  cpuid(&eax, &ebx, &ecx, &edx);

  name[__COUNTER__] = eax       & MASK;
  name[__COUNTER__] = (eax>>8)  & MASK;
  name[__COUNTER__] = (eax>>16) & MASK;
  name[__COUNTER__] = (eax>>24) & MASK;
  name[__COUNTER__] = ebx       & MASK;
  name[__COUNTER__] = (ebx>>8)  & MASK;
  name[__COUNTER__] = (ebx>>16) & MASK;
  name[__COUNTER__] = (ebx>>24) & MASK;
  name[__COUNTER__] = ecx       & MASK;
  name[__COUNTER__] = (ecx>>8)  & MASK;
  name[__COUNTER__] = (ecx>>16) & MASK;
  name[__COUNTER__] = (ecx>>24) & MASK;
  name[__COUNTER__] = edx       & MASK;
  name[__COUNTER__] = (edx>>8)  & MASK;
  name[__COUNTER__] = (edx>>16) & MASK;
  name[__COUNTER__] = (edx>>24) & MASK;

  eax = 0x80000004;
  cpuid(&eax, &ebx, &ecx, &edx);

  name[__COUNTER__] = eax       & MASK;
  name[__COUNTER__] = (eax>>8)  & MASK;
  name[__COUNTER__] = (eax>>16) & MASK;
  name[__COUNTER__] = (eax>>24) & MASK;
  name[__COUNTER__] = ebx       & MASK;
  name[__COUNTER__] = (ebx>>8)  & MASK;
  name[__COUNTER__] = (ebx>>16) & MASK;
  name[__COUNTER__] = (ebx>>24) & MASK;
  name[__COUNTER__] = ecx       & MASK;
  name[__COUNTER__] = (ecx>>8)  & MASK;
  name[__COUNTER__] = (ecx>>16) & MASK;
  name[__COUNTER__] = (ecx>>24) & MASK;
  name[__COUNTER__] = edx       & MASK;
  name[__COUNTER__] = (edx>>8)  & MASK;
  name[__COUNTER__] = (edx>>16) & MASK;
  name[__COUNTER__] = (edx>>24) & MASK;

  name[__COUNTER__] = '\0';

  //Remove unused characters
  int i = 0;
  while(name[i] == ' ')i++;

  char* name_withoutblank = (char *) malloc(sizeof(char)*64);
  strcpy(name_withoutblank,name+i);
  return name_withoutblank;
}

bool is_cpu_intel(struct cpu* cpu) {
  return cpu->cpu_vendor == CPU_VENDOR_INTEL;    
}

bool is_cpu_amd(struct cpu* cpu) {
  return cpu->cpu_vendor == CPU_VENDOR_AMD;  
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
  
  cpu->cpu_name = cpu_name();
  cpu->cpu_vendor = cpu_vendor();
  cpu->uarch = get_uarch(cpu);
  fill_features_cpuid(cpu);
  
  return cpu;
}

char* get_str_cpu_name(struct cpu* cpu) {
  return cpu->cpu_name;    
}

struct uarch* get_uarch_struct(struct cpu* cpu) {
  return cpu->uarch;    
}
