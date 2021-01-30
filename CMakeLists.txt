cmake_minimum_required(VERSION 3.10)
include(CheckLanguage)

project(peakperf CXX)

set(SRC_DIR "src")
set(CPU_DIR "${SRC_DIR}/cpu")
set(GPU_DIR "${SRC_DIR}/gpu")
set(CPUFETCH_DIR "${CPU_DIR}/cpufetch")
set(ARCH_CPU_DIR "${CPU_DIR}/arch")
set(ARCH_GPU_DIR "${GPU_DIR}/arch")

add_executable(peakperf ${SRC_DIR}/main.cpp ${SRC_DIR}/getarg.cpp)

if(NOT DEFINED ENABLE_CPU_DEVICE)
    set(ENABLE_CPU_DEVICE true)
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  message(STATUS "Building peakperf with CUDA support")
  set(ENABLE_GPU_DEVICE true)
else()
  set(ENABLE_GPU_DEVICE false)
  message(STATUS "No CUDA support")
endif()

set(SANITY_FLAGS "-Wall -Wextra -Werror -fstack-protector-all -pedantic -Wno-unused -Wfloat-equal -Wshadow -Wpointer-arith -Wformat=2")
set(CMAKE_C_FLAGS "${SANITY_FLAGS} -std=c99 -O2")

if(ENABLE_CPU_DEVICE)
    target_compile_definitions(peakperf PUBLIC DEVICE_CPU_ENABLED)

    set_source_files_properties(${ARCH_CPU_DIR}/arch.cpp       PROPERTIES COMPILE_FLAGS "-mavx -mavx512f -fopenmp")
    add_library(cpu_device   STATIC ${CPU_DIR}/cpu.cpp ${CPUFETCH_DIR}/cpufetch.cpp ${CPUFETCH_DIR}/cpuid.cpp ${CPUFETCH_DIR}/uarch.cpp ${ARCH_CPU_DIR}/arch.cpp)
    target_link_libraries(peakperf -lm -fopenmp cpu_device sandy_bridge ivy_bridge haswell skylake_256 skylake_512 broadwell ice_lake knl zen zen2)

    # architecture specific build
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

    add_library(sandy_bridge STATIC ${ARCH_CPU_DIR}/sandy_bridge.cpp)
    add_library(ivy_bridge   STATIC ${ARCH_CPU_DIR}/ivy_bridge.cpp)
    add_library(haswell      STATIC ${ARCH_CPU_DIR}/haswell.cpp)
    add_library(skylake_256  STATIC ${ARCH_CPU_DIR}/skylake_256.cpp)
    add_library(skylake_512  STATIC ${ARCH_CPU_DIR}/skylake_512.cpp)
    add_library(broadwell    STATIC ${ARCH_CPU_DIR}/broadwell.cpp)
    add_library(ice_lake     STATIC ${ARCH_CPU_DIR}/ice_lake.cpp)
    add_library(knl          STATIC ${ARCH_CPU_DIR}/knl.cpp)
    add_library(zen          STATIC ${ARCH_CPU_DIR}/zen.cpp)
    add_library(zen2         STATIC ${ARCH_CPU_DIR}/zen2.cpp)
endif()

if(ENABLE_GPU_DEVICE)
    target_compile_definitions(peakperf PUBLIC DEVICE_GPU_ENABLED)

    set(CMAKE_CUDA_ARCHITECTURES 52)
    add_library(gpu_device   STATIC ${GPU_DIR}/gpu.cu ${ARCH_GPU_DIR}/kernel.cu)
    target_link_libraries(peakperf gpu_device maxwell)

    # architecture specific build
    set_source_files_properties(${ARCH_GPU_DIR}/maxwell.cu PROPERTIES COMPILE_FLAGS "-arch=sm_52")
    add_library(maxwell      STATIC ${ARCH_GPU_DIR}/maxwell.cu)
endif()
