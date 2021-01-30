CXX=gcc

CUDA_HOME            ?= $(CUDA_PATH)
CUDA_INC_PATH        ?= $(CUDA_HOME)/include
CUDA_INC_PATH_COMMON ?= $(CUDA_HOME)/samples/common/inc

CUDA_BIN_PATH        ?= $(CUDA_HOME)/bin
CUDA_LIB_PATH        ?= $(CUDA_HOME)/lib64

SANITY_FLAGS=-Wall -Wextra -Werror -fstack-protector-all -pedantic -Wno-unused -Wfloat-equal -Wshadow -Wpointer-arith -Wformat=2
CXXFLAGS_GENERIC=-std=c99 -O2 $(SANITY_FLAGS)
CXXFLAGS_LINK=-lm -fopenmp
CXXFLAGS_LINK_CUDA=-lcuda -lcudart -lm
CXXFLAGS_SANDY_BRIDGE    = -DAVX_256_6_NOFMA -march=sandybridge    $(CXXFLAGS_GENERIC)
CXXFLAGS_IVY_BRIDGE      = -DAVX_256_6_NOFMA -march=ivybridge      $(CXXFLAGS_GENERIC)
CXXFLAGS_HASWELL         = -DAVX_256_10      -march=haswell        $(CXXFLAGS_GENERIC)
CXXFLAGS_SKYLAKE_256     = -DAVX_256_8       -march=skylake        $(CXXFLAGS_GENERIC)
CXXFLAGS_SKYLAKE_512     = -DAVX_512_8       -march=skylake-avx512 $(CXXFLAGS_GENERIC)
CXXFLAGS_BROADWELL       = -DAVX_256_8       -march=broadwell      $(CXXFLAGS_GENERIC)
CXXFLAGS_ICE_LAKE        = -DAVX_256_8       -march=icelake-client $(CXXFLAGS_GENERIC)
CXXFLAGS_KNL             = -DAVX_512_12      -march=knl            $(CXXFLAGS_GENERIC)
CXXFLAGS_ZEN             = -DAVX_256_5       -march=znver1         $(CXXFLAGS_GENERIC)
CXXFLAGS_ZEN2            = -DAVX_256_10      -march=znver1         $(CXXFLAGS_GENERIC)

SRC_DIR=src
CPU_DIR=$(SRC_DIR)/cpu
GPU_DIR=$(SRC_DIR)/gpu
ARCH_DIR=$(CPU_DIR)/arch
CPUFETCH_DIR=$(CPU_DIR)/cpufetch

MAIN=$(SRC_DIR)/main.c $(SRC_DIR)/getarg.c
CPU_DEVICE=$(CPU_DIR)/cpu.c $(CPUFETCH_DIR)/cpufetch.c $(CPUFETCH_DIR)/cpuid.c $(CPUFETCH_DIR)/uarch.c $(ARCH_DIR)/arch.c
GPU_DEVICE=$(GPU_DIR)/gpu.c

SANDY_BRIDGE=$(ARCH_DIR)/sandy_bridge.c
SANDY_BRIDGE_HEADERS=$(ARCH_DIR)/sandy_bridge.h $(ARCH_DIR)/arch.h

IVY_BRIDGE=$(ARCH_DIR)/ivy_bridge.c
IVY_BRIDGE_HEADERS=$(ARCH_DIR)/ivy_bridge.h $(ARCH_DIR)/arch.h

HASWELL=$(ARCH_DIR)/haswell.c
HASWELL_HEADERS=$(ARCH_DIR)/haswell.h $(ARCH_DIR)/arch.h

SKYLAKE_256=$(ARCH_DIR)/skylake_256.c
SKYLAKE_256_HEADERS=$(ARCH_DIR)/skylake_256.h $(ARCH_DIR)/arch.h

SKYLAKE_512=$(ARCH_DIR)/skylake_512.c
SKYLAKE_512_HEADERS=$(ARCH_DIR)/skylake_512.h $(ARCH_DIR)/arch.h

BROADWELL=$(ARCH_DIR)/broadwell.c
BROADWELL_HEADERS=$(ARCH_DIR)/broadwell.h $(ARCH_DIR)/arch.h

ICE_LAKE=$(ARCH_DIR)/ice_lake.c
ICE_LAKE_HEADERS=$(ARCH_DIR)/ice_lake.h $(ARCH_DIR)/arch.h

KNL=$(ARCH_DIR)/knl.c
KNL_HEADERS=$(ARCH_DIR)/knl.h $(ARCH_DIR)/arch.h

ZEN=$(ARCH_DIR)/zen.c
ZEN_HEADERS=$(ARCH_DIR)/zen.h $(ARCH_DIR)/arch.h

ZEN2=$(ARCH_DIR)/zen2.c
ZEN2_HEADERS=$(ARCH_DIR)/zen2.h $(ARCH_DIR)/arch.h

OUTPUT_DIR=output
$(shell mkdir -p $(OUTPUT_DIR))

OUT_SANDY_BRIDGE=$(OUTPUT_DIR)/sandy_bridge.o
OUT_IVY_BRIDGE=$(OUTPUT_DIR)/ivy_bridge.o
OUT_HASWELL=$(OUTPUT_DIR)/haswell.o
OUT_SKYLAKE_256=$(OUTPUT_DIR)/skylake_256.o
OUT_SKYLAKE_512=$(OUTPUT_DIR)/skylake_512.o
OUT_BROADWELL=$(OUTPUT_DIR)/broadwell.o
OUT_ICE_LAKE=$(OUTPUT_DIR)/ice_lake.o
OUT_KNL=$(OUTPUT_DIR)/knl.o
OUT_ZEN=$(OUTPUT_DIR)/zen.o
OUT_ZEN2=$(OUTPUT_DIR)/zen2.o
CPU_DEVICE_OUT=$(OUTPUT_DIR)/arch.o $(OUTPUT_DIR)/cpuid.o $(OUTPUT_DIR)/cpu.o $(OUTPUT_DIR)/cpufetch.o $(OUTPUT_DIR)/uarch.o
GPU_DEVICE_OUT=$(OUTPUT_DIR)/gpu.o

ALL_OUTS=$(OUT_SANDY_BRIDGE) $(OUT_IVY_BRIDGE) $(OUT_HASWELL) $(OUT_SKYLAKE_256) $(OUT_SKYLAKE_512) $(OUT_BROADWELL) $(OUT_ICE_LAKE) $(OUT_KNL) $(OUT_ZEN) $(OUT_ZEN2)

peakperf: Makefile $(MAIN) $(ALL_OUTS) $(CPU_DEVICE_OUT) $(GPU_DEVICE_OUT)
	$(CXX) -mavx $(ALL_OUTS) $(CPU_DEVICE_OUT) $(GPU_DEVICE_OUT) -L$(CUDA_LIB_PATH) $(CXXFLAGS_LINK_CUDA) $(CXXFLAGS_LINK) $(MAIN) -o $@

$(OUT_SANDY_BRIDGE): Makefile $(SANDY_BRIDGE) $(SANDY_BRIDGE_HEADERS)
	$(CXX) $(CXXFLAGS_SANDY_BRIDGE) $(SANDY_BRIDGE) -c -o $@

$(OUT_IVY_BRIDGE): Makefile $(IVY_BRIDGE) $(IVY_BRIDGE_HEADERS)
	$(CXX) $(CXXFLAGS_IVY_BRIDGE) $(IVY_BRIDGE) -c -o $@

$(OUT_HASWELL): Makefile $(HASWELL) $(HASWELL_HEADERS)
	$(CXX) $(CXXFLAGS_HASWELL) $(HASWELL) -c -o $@

$(OUT_SKYLAKE_256): Makefile $(SKYLAKE_256) $(SKYLAKE_256_HEADERS)
	$(CXX) $(CXXFLAGS_SKYLAKE_256) $(SKYLAKE_256) -c -o $@

$(OUT_SKYLAKE_512): Makefile $(SKYLAKE_512) $(SKYLAKE_512_HEADERS)
	$(CXX) $(CXXFLAGS_SKYLAKE_512) $(SKYLAKE_512) -c -o $@

$(OUT_BROADWELL): Makefile $(BROADWELL) $(BROADWELL_HEADERS)
	$(CXX) $(CXXFLAGS_BROADWELL) $(BROADWELL) -c -o $@

$(OUT_ICE_LAKE): Makefile $(ICE_LAKE) $(ICE_LAKE_HEADERS)
	$(CXX) $(CXXFLAGS_ICE_LAKE) $(ICE_LAKE) -c -o $@

$(OUT_KNL): Makefile $(KNL) $(KNL_HEADERS)
	$(CXX) $(CXXFLAGS_KNL) $(KNL) -c -o $@

$(OUT_ZEN): Makefile $(ZEN) $(ZEN_HEADERS)
	$(CXX) $(CXXFLAGS_ZEN) $(ZEN) -c -o $@

$(OUT_ZEN2): Makefile $(ZEN2) $(ZEN2_HEADERS)
	$(CXX) $(CXXFLAGS_ZEN2) $(ZEN2) -c -o $@

$(CPU_DEVICE_OUT): Makefile $(CPU_DEVICE)
	$(CXX) $(CXXFLAGS_GENERIC) -mavx $(CPU_DEVICE) $(CXXFLAGS_LINK) -c
	mv *.o output/

$(GPU_DEVICE_OUT): Makefile $(GPU_DEVICE)
	nvcc -I$(CUDA_INC_PATH) -I$(CUDA_INC_PATH_COMMON) $(GPU_DEVICE) -L$(CUDA_LIB_PATH) $(CXXFLAGS_LINK_CUDA) -c -o $@

clean:
	@rm -r peakperf $(OUTPUT_DIR)
