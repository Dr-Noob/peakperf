/*
 * Code based on cpufetch project at
 * https://github.com/Dr-Noob/cpufetch/blob/master/src/x86/uarch.c
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cpu.h"
#include "cpuid.h"
#include "uarch.h"

// Data not available
#define NA                   -1

#define UARCH_START if (false) {}
#define CHECK_UARCH(arch, ef_, f_, em_, m_, s_, str, uarch) \
   else if (ef_ == ef && f_ == f && (em_ == NA || em_ == em) && (m_ == NA || m_ == m) && (s_ == NA || s_ == s)) fill_uarch(arch, str, uarch);
#define UARCH_END else { printf("Unknown microarchitecture detected: M=0x%.8X EM=0x%.8X F=0x%.8X EF=0x%.8X S=0x%.8X", m, em, f, ef, s); fill_uarch(arch, "Unknown", UARCH_UNKNOWN); }

void fill_uarch(struct uarch* arch, char* str, MICROARCH u) {
  arch->uarch_str = malloc(sizeof(char) * (strlen(str)+1));
  strcpy(arch->uarch_str, str);
  arch->uarch = u;
}

struct uarch* get_uarch_from_cpuid_intel(uint32_t ef, uint32_t f, uint32_t em, uint32_t m, int s) {
  struct uarch* arch = malloc(sizeof(struct uarch));
  
  // EF: Extended Family                                          //
  // F:  Family                                                   //
  // EM: Extended Model                                           //
  // M: Model                                                     //
  // S: Stepping                                                  //
  // ------------------------------------------------------------ //
  //                EF  F  EM   M   S                             //
  UARCH_START  
  CHECK_UARCH(arch, 0,  5,  0,  0, NA, "P5",              UARCH_P5)
  CHECK_UARCH(arch, 0,  5,  0,  1, NA, "P5",              UARCH_P5)
  CHECK_UARCH(arch, 0,  5,  0,  2, NA, "P5",              UARCH_P5)
  CHECK_UARCH(arch, 0,  5,  0,  3, NA, "P5",              UARCH_P5)
  CHECK_UARCH(arch, 0,  5,  0,  4, NA, "P5 MMX",          UARCH_P5)
  CHECK_UARCH(arch, 0,  5,  0,  7, NA, "P5 MMX",          UARCH_P5)
  CHECK_UARCH(arch, 0,  5,  0,  8, NA, "P5 MMX",          UARCH_P5)
  CHECK_UARCH(arch, 0,  5,  0,  9, NA, "P5 MMX",          UARCH_P5)
  CHECK_UARCH(arch, 0,  6,  0,  0, NA, "P6 Pentium II",   UARCH_P6)
  CHECK_UARCH(arch, 0,  6,  0,  1, NA, "P6 Pentium II",   UARCH_P6)
  CHECK_UARCH(arch, 0,  6,  0,  2, NA, "P6 Pentium II",   UARCH_P6)
  CHECK_UARCH(arch, 0,  6,  0,  3, NA, "P6 Pentium II",   UARCH_P6)
  CHECK_UARCH(arch, 0,  6,  0,  4, NA, "P6 Pentium II",   UARCH_P6)
  CHECK_UARCH(arch, 0,  6,  0,  5, NA, "P6 Pentium II",   UARCH_P6)
  CHECK_UARCH(arch, 0,  6,  0,  6, NA, "P6 Pentium II",   UARCH_P6)
  CHECK_UARCH(arch, 0,  6,  0,  7, NA, "P6 Pentium III",  UARCH_P6)
  CHECK_UARCH(arch, 0,  6,  0,  8, NA, "P6 Pentium III",  UARCH_P6)
  CHECK_UARCH(arch, 0,  6,  0,  9, NA, "P6 Pentium M",    UARCH_P6)
  CHECK_UARCH(arch, 0,  6,  0, 10, NA, "P6 Pentium III",  UARCH_P6)
  CHECK_UARCH(arch, 0,  6,  0, 11, NA, "P6 Pentium III",  UARCH_P6)
  CHECK_UARCH(arch, 0,  6,  0, 13, NA, "Dothan",          UARCH_DOTHAN)
  CHECK_UARCH(arch, 0,  6,  0, 14, NA, "Yonah",           UARCH_YONAH)
  CHECK_UARCH(arch, 0,  6,  0, 15, NA, "Merom",           UARCH_MEROM)
  CHECK_UARCH(arch, 0,  6,  1,  5, NA, "Dothan",          UARCH_DOTHAN)
  CHECK_UARCH(arch, 0,  6,  1,  6, NA, "Merom",           UARCH_MEROM)
  CHECK_UARCH(arch, 0,  6,  1,  7, NA, "Penryn",          UARCH_PENYR)
  CHECK_UARCH(arch, 0,  6,  1, 10, NA, "Nehalem",         UARCH_NEHALEM)
  CHECK_UARCH(arch, 0,  6,  1, 12, NA, "Bonnell",         UARCH_BONNELL)
  CHECK_UARCH(arch, 0,  6,  1, 13, NA, "Penryn",          UARCH_PENYR)
  CHECK_UARCH(arch, 0,  6,  1, 14, NA, "Nehalem",         UARCH_NEHALEM)
  CHECK_UARCH(arch, 0,  6,  1, 15, NA, "Nehalem",         UARCH_NEHALEM)
  CHECK_UARCH(arch, 0,  6,  2,  5, NA, "Westmere",        UARCH_WESTMERE)
  CHECK_UARCH(arch, 0,  6,  2 , 6, NA, "Bonnell",         UARCH_BONNELL)
  CHECK_UARCH(arch, 0,  6,  2,  7, NA, "Saltwell",        UARCH_SALTWELL)
  CHECK_UARCH(arch, 0,  6,  2, 10, NA, "Sandy Bridge",    UARCH_SANDY_BRIDGE)
  CHECK_UARCH(arch, 0,  6,  2, 12, NA, "Westmere",        UARCH_WESTMERE)
  CHECK_UARCH(arch, 0,  6,  2, 13, NA, "Sandy Bridge",    UARCH_SANDY_BRIDGE)
  CHECK_UARCH(arch, 0,  6,  2, 14, NA, "Nehalem",         UARCH_NEHALEM)
  CHECK_UARCH(arch, 0,  6,  2, 15, NA, "Westmere",        UARCH_WESTMERE)
  CHECK_UARCH(arch, 0,  6,  3,  5, NA, "Saltwell",        UARCH_SALTWELL)
  CHECK_UARCH(arch, 0,  6,  3,  6, NA, "Saltwell",        UARCH_SALTWELL)
  CHECK_UARCH(arch, 0,  6,  3,  7, NA, "Silvermont",      UARCH_SILVERMONT)
  CHECK_UARCH(arch, 0,  6,  3, 10, NA, "Ivy Bridge",      UARCH_IVY_BRIDGE)
  CHECK_UARCH(arch, 0,  6,  3, 12, NA, "Haswell",         UARCH_HASWELL)
  CHECK_UARCH(arch, 0,  6,  3, 13, NA, "Broadwell",       UARCH_BROADWELL)
  CHECK_UARCH(arch, 0,  6,  3, 14, NA, "Ivy Bridge",      UARCH_IVY_BRIDGE)
  CHECK_UARCH(arch, 0,  6,  3, 15, NA, "Haswell",         UARCH_HASWELL)
  CHECK_UARCH(arch, 0,  6,  4,  5, NA, "Haswell",         UARCH_HASWELL)
  CHECK_UARCH(arch, 0,  6,  4,  6, NA, "Haswell",         UARCH_HASWELL)
  CHECK_UARCH(arch, 0,  6,  4,  7, NA, "Broadwell",       UARCH_BROADWELL)
  CHECK_UARCH(arch, 0,  6,  4, 10, NA, "Silvermont",      UARCH_SILVERMONT)
  CHECK_UARCH(arch, 0,  6,  4, 12, NA, "Airmont",         UARCH_AIRMONT)
  CHECK_UARCH(arch, 0,  6,  4, 13, NA, "Silvermont",      UARCH_SILVERMONT)
  CHECK_UARCH(arch, 0,  6,  4, 14,  8, "Kaby Lake",       UARCH_KABY_LAKE)
  CHECK_UARCH(arch, 0,  6,  4, 14, NA, "Skylake",         UARCH_SKYLAKE)
  CHECK_UARCH(arch, 0,  6,  4, 15, NA, "Broadwell",       UARCH_BROADWELL)
  CHECK_UARCH(arch, 0,  6,  5,  5,  6, "Cascade Lake",    UARCH_CASCADE_LAKE)
  CHECK_UARCH(arch, 0,  6,  5,  5,  7, "Cascade Lake",    UARCH_CASCADE_LAKE)
  CHECK_UARCH(arch, 0,  6,  5,  5, 10, "Cooper Lake",     UARCH_COOPER_LAKE)
  CHECK_UARCH(arch, 0,  6,  5,  5, NA, "Skylake",         UARCH_SKYLAKE)
  CHECK_UARCH(arch, 0,  6,  5,  6, NA, "Broadwell",       UARCH_BROADWELL)
  CHECK_UARCH(arch, 0,  6,  5,  7, NA, "Knights Landing", UARCH_KNIGHTS_LANDING)
  CHECK_UARCH(arch, 0,  6,  5, 10, NA, "Silvermont",      UARCH_SILVERMONT)
  CHECK_UARCH(arch, 0,  6,  5, 12, NA, "Goldmont",        UARCH_GOLDMONT)
  CHECK_UARCH(arch, 0,  6,  5, 13, NA, "Silvermont",      UARCH_SILVERMONT)
  CHECK_UARCH(arch, 0,  6,  5, 14,  8, "Kaby Lake",       UARCH_KABY_LAKE)
  CHECK_UARCH(arch, 0,  6,  5, 14, NA, "Skylake",         UARCH_SKYLAKE)
  CHECK_UARCH(arch, 0,  6,  5, 15, NA, "Goldmont",        UARCH_GOLDMONT)
  CHECK_UARCH(arch, 0,  6,  6,  6, NA, "Palm Cove",       UARCH_PALM_COVE)
  CHECK_UARCH(arch, 0,  6,  6, 10, NA, "Sunny Cove",      UARCH_SUNNY_COVE)
  CHECK_UARCH(arch, 0,  6,  6, 12, NA, "Sunny Cove",      UARCH_SUNNY_COVE)
  CHECK_UARCH(arch, 0,  6,  7,  5, NA, "Airmont",         UARCH_AIRMONT)
  CHECK_UARCH(arch, 0,  6,  7, 10, NA, "Goldmont Plus",   UARCH_GOLDMONT_PLUS)
  CHECK_UARCH(arch, 0,  6,  7, 13, NA, "Sunny Cove",      UARCH_SUNNY_COVE)
  CHECK_UARCH(arch, 0,  6,  7, 14, NA, "Ice Lake",        UARCH_ICE_LAKE)
  CHECK_UARCH(arch, 0,  6,  8,  5, NA, "Knights Mill",    UARCH_KNIGHTS_MILL)
  CHECK_UARCH(arch, 0,  6,  8,  6, NA, "Tremont",         UARCH_TREMONT)
  CHECK_UARCH(arch, 0,  6,  8, 10, NA, "Tremont",         UARCH_TREMONT)
  CHECK_UARCH(arch, 0,  6,  8, 12, NA, "Willow Cove",     UARCH_WILLOW_COVE)
  CHECK_UARCH(arch, 0,  6,  8, 13, NA, "Willow Cove",     UARCH_WILLOW_COVE)
  CHECK_UARCH(arch, 0,  6,  8, 14, NA, "Kaby Lake",       UARCH_KABY_LAKE)
  CHECK_UARCH(arch, 0,  6,  9,  6, NA, "Tremont",         UARCH_TREMONT)
  CHECK_UARCH(arch, 0,  6,  9, 12, NA, "Tremont",         UARCH_TREMONT)
  CHECK_UARCH(arch, 0,  6,  9, 13, NA, "Sunny Cove",      UARCH_SUNNY_COVE)
  CHECK_UARCH(arch, 0,  6,  9, 14,  9, "Kaby Lake",       UARCH_KABY_LAKE)
  CHECK_UARCH(arch, 0,  6,  9, 14, 10, "Coffee Lake",     UARCH_COFFE_LAKE)
  CHECK_UARCH(arch, 0,  6,  9, 14, 11, "Coffee Lake",     UARCH_COFFE_LAKE)
  CHECK_UARCH(arch, 0,  6,  9, 14, 12, "Coffee Lake",     UARCH_COFFE_LAKE)
  CHECK_UARCH(arch, 0,  6,  9, 14, 13, "Coffee Lake",     UARCH_COFFE_LAKE)
  CHECK_UARCH(arch, 0,  6, 10,  5, NA, "Kaby Lake",       UARCH_KABY_LAKE)
  CHECK_UARCH(arch, 0,  6, 10,  6, NA, "Kaby Lake",       UARCH_KABY_LAKE)
  CHECK_UARCH(arch, 0, 11,  0,  0, NA, "Knights Ferry",   UARCH_KNIGHTS_FERRY)
  CHECK_UARCH(arch, 0, 11,  0,  1, NA, "Knights Corner",  UARCH_KNIGHTS_CORNER)
  CHECK_UARCH(arch, 0, 15,  0,  0, NA, "Willamette",      UARCH_WILLAMETTE)
  CHECK_UARCH(arch, 0, 15,  0,  1, NA, "Willamette",      UARCH_WILLAMETTE)
  CHECK_UARCH(arch, 0, 15,  0,  2, NA, "Northwood",       UARCH_NORTHWOOD)
  CHECK_UARCH(arch, 0, 15,  0,  3, NA, "Prescott",        UARCH_PRESCOTT)
  CHECK_UARCH(arch, 0, 15,  0,  4, NA, "Prescott",        UARCH_PRESCOTT)
  CHECK_UARCH(arch, 0, 15,  0,  6, NA, "Cedar Mill",      UARCH_CEDAR_MILL)
  CHECK_UARCH(arch, 1, 15,  0,  0, NA, "Itanium2",        UARCH_ITANIUM2)
  CHECK_UARCH(arch, 1, 15,  0,  1, NA, "Itanium2",        UARCH_ITANIUM2)
  CHECK_UARCH(arch, 1, 15,  0,  2, NA, "Itanium2",        UARCH_ITANIUM2)
  UARCH_END
    
  return arch;
}

// iNApired in Todd Allen's decode_uarch_amd
struct uarch* get_uarch_from_cpuid_amd(uint32_t ef, uint32_t f, uint32_t em, uint32_t m, int s) {
  struct uarch* arch = malloc(sizeof(struct uarch));
  
  // EF: Extended Family                                          //
  // F:  Family                                                   //
  // EM: Extended Model                                           //
  // M: Model                                                     //
  // S: Stepping                                                  //
  // ------------------------------------------------------------ //
  //                 EF  F  EM   M   S                            //
  UARCH_START  
  CHECK_UARCH(arch,  0,  4,  0,  3, NA, "Am486",       UARCH_AM486)
  CHECK_UARCH(arch,  0,  4,  0,  7, NA, "Am486",       UARCH_AM486)
  CHECK_UARCH(arch,  0,  4,  0,  8, NA, "Am486",       UARCH_AM486)
  CHECK_UARCH(arch,  0,  4,  0,  9, NA, "Am486",       UARCH_AM486)
  CHECK_UARCH(arch,  0,  4, NA, NA, NA, "Am5x86",      UARCH_AM5X86)
  CHECK_UARCH(arch,  0,  5,  0,  6, NA, "K6",          UARCH_K6)
  CHECK_UARCH(arch,  0,  5,  0,  7, NA, "K6",          UARCH_K6)
  CHECK_UARCH(arch,  0,  5,  0, 13, NA, "K6",          UARCH_K6)
  CHECK_UARCH(arch,  0,  5, NA, NA, NA, "K6",          UARCH_K6)
  CHECK_UARCH(arch,  0,  6,  0,  1, NA, "K7",          UARCH_K7)
  CHECK_UARCH(arch,  0,  6,  0,  2, NA, "K7",          UARCH_K7)
  CHECK_UARCH(arch,  0,  6, NA, NA, NA, "K7",          UARCH_K7)
  CHECK_UARCH(arch,  0, 15,  0,  4,  8, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  0,  4, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  0,  5, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  0,  7, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  0,  8, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  0, 11, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  0, 12, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  0, 14, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  0, 15, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  1,  4, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  1,  5, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  1,  7, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  1,  8, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  1, 11, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  1, 12, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  1, 15, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  2,  1, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  2,  3, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  2,  4, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  2,  5, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  2,  7, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  2, 11, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  2, 12, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  2, 15, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  4,  1, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  4,  3, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  4,  8, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  4, 11, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  4, 12, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  4, 15, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  5, 13, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  5, 15, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  6,  8, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  6, 11, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  6, 12, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  6, 15, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  7, 12, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15,  7, 15, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  0, 15, 12,  1, NA, "K8",          UARCH_K8)
  CHECK_UARCH(arch,  1, 15,  0,  0, NA, "K10",         UARCH_K10)
  CHECK_UARCH(arch,  1, 15,  0,  2, NA, "K10",         UARCH_K10)
  CHECK_UARCH(arch,  1, 15,  0,  4, NA, "K10",         UARCH_K10)
  CHECK_UARCH(arch,  1, 15,  0,  5, NA, "K10",         UARCH_K10)
  CHECK_UARCH(arch,  1, 15,  0,  6, NA, "K10",         UARCH_K10)
  CHECK_UARCH(arch,  1, 15,  0,  8, NA, "K10",         UARCH_K10)
  CHECK_UARCH(arch,  1, 15,  0,  9, NA, "K10",         UARCH_K10)
  CHECK_UARCH(arch,  1, 15,  0, 10, NA, "K10",         UARCH_K10)
  CHECK_UARCH(arch,  2, 15, NA, NA, NA, "Puma 2008",   UARCH_PUMA_2008)
  CHECK_UARCH(arch,  3, 15, NA, NA, NA, "K10",         UARCH_K10)
  CHECK_UARCH(arch,  5, 15, NA, NA, NA, "Bobcat",      UARCH_BOBCAT)
  CHECK_UARCH(arch,  6, 15,  0,  0, NA, "Bulldozer",   UARCH_BULLDOZER)
  CHECK_UARCH(arch,  6, 15,  0,  1, NA, "Bulldozer",   UARCH_BULLDOZER)
  CHECK_UARCH(arch,  6, 15,  0,  2, NA, "Piledriver",  UARCH_PILEDRIVER)
  CHECK_UARCH(arch,  6, 15,  1,  0, NA, "Piledriver",  UARCH_PILEDRIVER)
  CHECK_UARCH(arch,  6, 15,  1,  3, NA, "Piledriver",  UARCH_PILEDRIVER)
  CHECK_UARCH(arch,  6, 15,  3,  0, NA, "Steamroller", UARCH_STEAMROLLER)
  CHECK_UARCH(arch,  6, 15,  3,  8, NA, "Steamroller", UARCH_STEAMROLLER)
  CHECK_UARCH(arch,  6, 15,  4,  0, NA, "Steamroller", UARCH_STEAMROLLER)
  CHECK_UARCH(arch,  6, 15,  6,  0, NA, "Excavator",   UARCH_EXCAVATOR)
  CHECK_UARCH(arch,  6, 15,  6,  5, NA, "Excavator",   UARCH_EXCAVATOR)
  CHECK_UARCH(arch,  6, 15,  7,  0, NA, "Excavator",   UARCH_EXCAVATOR)
  CHECK_UARCH(arch,  7, 15,  0,  0, NA, "Jaguar",      UARCH_JAGUAR)
  CHECK_UARCH(arch,  7, 15,  3,  0, NA, "Puma 2014",   UARCH_PUMA_2014)
  CHECK_UARCH(arch,  8, 15,  0,  0, NA, "Zen",         UARCH_ZEN)
  CHECK_UARCH(arch,  8, 15,  0,  1, NA, "Zen",         UARCH_ZEN)
  CHECK_UARCH(arch,  8, 15,  0,  8, NA, "Zen+",        UARCH_ZEN_PLUS)
  CHECK_UARCH(arch,  8, 15,  1,  1, NA, "Zen",         UARCH_ZEN)
  CHECK_UARCH(arch,  8, 15,  1,  8, NA, "Zen+",        UARCH_ZEN_PLUS)
  CHECK_UARCH(arch,  8, 15,  3,  1, NA, "Zen 2",       UARCH_ZEN2)
  CHECK_UARCH(arch,  8, 15,  6,  0, NA, "Zen 2",       UARCH_ZEN2)
  CHECK_UARCH(arch,  8, 15,  7,  1, NA, "Zen 2",       UARCH_ZEN2)
  CHECK_UARCH(arch, 10, 15, NA, NA, NA, "Zen 3",       UARCH_ZEN3)
  UARCH_END
    
  return arch;
}

struct uarch* get_uarch_from_cpuid(struct cpu* cpu, uint32_t ef, uint32_t f, uint32_t em, uint32_t m, int s) {  
  if(is_cpu_intel(cpu))
    return get_uarch_from_cpuid_intel(ef, f, em, m, s);
  else if(is_cpu_amd(cpu))
    return get_uarch_from_cpuid_amd(ef, f, em, m, s);
  else {
    printf("Found invalid vendor in get_uarch_from_cpuid\n");
    return NULL;
  }
}

struct uarch* get_uarch(struct cpu* cpu) {
  uint32_t eax = 0x00000001;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;
  
  cpuid(&eax, &ebx, &ecx, &edx);
   
  uint32_t stepping = eax & 0xF;
  uint32_t model = (eax >> 4) & 0xF;
  uint32_t emodel = (eax >> 16) & 0xF;
  uint32_t family = (eax >> 8) & 0xF;
  uint32_t efamily = (eax >> 20) & 0xFF;
  
  return get_uarch_from_cpuid(cpu, efamily, family, emodel, model, (int)stepping);
}

void free_uarch_struct(struct uarch* arch) {    
  free(arch->uarch_str);
  free(arch);
}
