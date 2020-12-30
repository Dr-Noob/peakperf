CXX=gcc

SANITY_FLAGS=-Wall -Wextra -Werror -fstack-protector-all -pedantic -Wno-unused -Wfloat-equal -Wshadow -Wpointer-arith -Wformat=2
CXXFLAGS_GENERIC=-std=c99 -O2 $(SANITY_FLAGS)
CXXFLAGS_LINK=-lm -fopenmp
CXXFLAGS_HASWELL         = -DTEST_NAME="\"Haswell - 256 bits\""     -DBUILDING_OBJECT           -DAVX_256_10      -march=haswell        $(CXXFLAGS_GENERIC)
CXXFLAGS_KABY_LAKE       = -DTEST_NAME="\"Kaby Lake - 256 bits\""   -DBUILDING_OBJECT           -DAVX_256_8       -march=skylake        $(CXXFLAGS_GENERIC)

ARCH_DIR=Arch
CPUFETCH_DIR=cpufetch
MAIN=main.c getarg.c $(CPUFETCH_DIR)/cpu.c $(CPUFETCH_DIR)/cpuid.c $(CPUFETCH_DIR)/uarch.c $(ARCH_DIR)/Arch.c

HASWELL=$(ARCH_DIR)/256_10.c
HASWELL_HEADERS=$(ARCH_DIR)/256_10.h $(ARCH_DIR)/Arch.h

KABY_LAKE=$(ARCH_DIR)/256_8.c
KABY_LAKE_HEADERS=$(ARCH_DIR)/256_8.h $(ARCH_DIR)/Arch.h

OUTPUT_DIR=output
$(shell mkdir -p $(OUTPUT_DIR))

OUTPUT_HASWELL=$(OUTPUT_DIR)/comp_haswell.o
OUTPUT_KABY_LAKE=$(OUTPUT_DIR)/comp_kaby_lake.o

peakperf: main.c $(OUTPUT_HASWELL) $(OUTPUT_KABY_LAKE)
	$(CXX) $(CXXFLAGS_GENERIC) -mavx $(CXXFLAGS_LINK) $(MAIN) $(OUTPUT_HASWELL) $(OUTPUT_KABY_LAKE) -o $@

$(OUTPUT_HASWELL): Makefile $(HASWELL) $(HASWELL_HEADERS)
	$(CXX) $(CXXFLAGS_HASWELL) $(HASWELL) -c -o $@
	
$(OUTPUT_KABY_LAKE): Makefile $(KABY_LAKE) $(KABY_LAKE_HEADERS)
	$(CXX) $(CXXFLAGS_KABY_LAKE) $(KABY_LAKE) -c -o $@
	
clean:
	@rm $(OUTPUT_DIR)/*
