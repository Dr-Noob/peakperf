cmake_minimum_required(VERSION 3.10)

project(peakperf CXX CUDA)
set(PROJECT_NAME "peakperf")
set(PROJECT_VERSION "1.2")
set(SRC_DIR "src")
set(CPU_DIR "${SRC_DIR}/cpu")
set(GPU_DIR "${SRC_DIR}/gpu")
set(CPUFETCH_DIR "${CPU_DIR}/cpufetch")
set(ARCH_CPU_DIR "${CPU_DIR}/arch")
set(ARCH_GPU_DIR "${GPU_DIR}/arch")

set(CMAKE_CUDA_ARCHITECTURES 52)

set(SANITY_FLAGS "-Wall -Wextra -Werror -fstack-protector-all -pedantic -Wno-unused -Wfloat-equal -Wshadow -Wpointer-arith -Wformat=2")
set(CMAKE_C_FLAGS "${SANITY_FLAGS} -std=c99 -O2")
set_source_files_properties(${ARCH_CPU_DIR}/arch.cpp       PROPERTIES COMPILE_FLAGS "-mavx -mavx512f -fopenmp")

# architecture specific build -- cpu
set_source_files_properties(${ARCH_CPU_DIR}/sandy_bridge.cpp PROPERTIES COMPILE_FLAGS "-DAVX_256_6_NOFMA -march=sandybridge")
set_source_files_properties(${ARCH_CPU_DIR}/ivy_bridge.cpp   PROPERTIES COMPILE_FLAGS "-DAVX_256_6_NOFMA -march=ivybridge")
set_source_files_properties(${ARCH_CPU_DIR}/haswell.cpp      PROPERTIES COMPILE_FLAGS "-DAVX_256_10      -march=haswell")
set_source_files_properties(${ARCH_CPU_DIR}/skylake_256.cpp  PROPERTIES COMPILE_FLAGS "-DAVX_256_8       -march=skylake")
set_source_files_properties(${ARCH_CPU_DIR}/skylake_512.cpp  PROPERTIES COMPILE_FLAGS "-DAVX_512_8       -march=skylake-avx512")
set_source_files_properties(${ARCH_CPU_DIR}/broadwell.cpp    PROPERTIES COMPILE_FLAGS "-DAVX_256_8       -march=broadwell")
set_source_files_properties(${ARCH_CPU_DIR}/ice_lake.cpp     PROPERTIES COMPILE_FLAGS "-DAVX_256_8       -march=icelake-client")
set_source_files_properties(${ARCH_CPU_DIR}/knl.cpp          PROPERTIES COMPILE_FLAGS "-DAVX_512_12      -march=knl")
set_source_files_properties(${ARCH_CPU_DIR}/zen.cpp          PROPERTIES COMPILE_FLAGS "-DAVX_256_5       -march=znver1")
set_source_files_properties(${ARCH_CPU_DIR}/zen2.cpp         PROPERTIES COMPILE_FLAGS "-DAVX_256_10      -march=znver1")

# architecture specific build -- gpu
set_source_files_properties(${ARCH_GPU_DIR}/maxwell.cu PROPERTIES COMPILE_FLAGS "-arch=sm_52")

add_library(sandy_bridge SHARED ${ARCH_CPU_DIR}/sandy_bridge.cpp)
add_library(ivy_bridge   SHARED ${ARCH_CPU_DIR}/ivy_bridge.cpp)
add_library(haswell      SHARED ${ARCH_CPU_DIR}/haswell.cpp)
add_library(skylake_256  SHARED ${ARCH_CPU_DIR}/skylake_256.cpp)
add_library(skylake_512  SHARED ${ARCH_CPU_DIR}/skylake_512.cpp)
add_library(broadwell    SHARED ${ARCH_CPU_DIR}/broadwell.cpp)
add_library(ice_lake     SHARED ${ARCH_CPU_DIR}/ice_lake.cpp)
add_library(knl          SHARED ${ARCH_CPU_DIR}/knl.cpp)
add_library(zen          SHARED ${ARCH_CPU_DIR}/zen.cpp)
add_library(zen2         SHARED ${ARCH_CPU_DIR}/zen2.cpp)
add_library(maxwell      SHARED ${ARCH_GPU_DIR}/maxwell.cu)

add_library(cpu_device   SHARED ${CPU_DIR}/cpu.cpp ${CPUFETCH_DIR}/cpufetch.cpp ${CPUFETCH_DIR}/cpuid.cpp ${CPUFETCH_DIR}/uarch.cpp ${ARCH_CPU_DIR}/arch.cpp)
add_library(gpu_device   SHARED ${GPU_DIR}/gpu.cu ${ARCH_GPU_DIR}/kernel.cu)

add_executable(peakperf ${SRC_DIR}/main.cpp ${SRC_DIR}/getarg.cpp)
target_link_libraries (peakperf -lm -fopenmp sandy_bridge ivy_bridge haswell skylake_256 skylake_512 broadwell ice_lake knl zen zen2 maxwell cpu_device gpu_device)
