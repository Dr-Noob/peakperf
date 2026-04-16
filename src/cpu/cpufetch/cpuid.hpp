#ifndef __CPUID__
#define __CPUID__

#if defined(__x86_64__) || defined(__i386__)
void cpuid(unsigned int *eax, unsigned int *ebx, unsigned int *ecx, unsigned int *edx);
#endif

#endif
