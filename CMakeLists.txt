cmake_minimum_required(VERSION 3.10)
include(CheckLanguage)

project(peakperf CXX)

set(SRC_DIR "src")
set(CPU_DIR "${SRC_DIR}/cpu")
set(GPU_DIR "${SRC_DIR}/gpu")
set(CPUFETCH_DIR "${CPU_DIR}/cpufetch")
set(ARCH_CPU_DIR "${CPU_DIR}/arch")

add_executable(peakperf ${SRC_DIR}/main.cpp ${SRC_DIR}/benchmark.cpp ${SRC_DIR}/getarg.cpp ${SRC_DIR}/global.cpp)

if(NOT DEFINED ENABLE_CPU_DEVICE)
    set(ENABLE_CPU_DEVICE true)
endif()

if(NOT DEFINED ENABLE_GPU_DEVICE OR ENABLE_GPU_DEVICE)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(ENABLE_GPU_DEVICE true)
  else()
    set(ENABLE_GPU_DEVICE false)
  endif()
endif()

set(SANITY_FLAGS "-Wall -Wextra -Werror -fstack-protector-all -pedantic -Wno-unused -Wfloat-equal -Wshadow -Wpointer-arith -Wformat=2")
set(CMAKE_CXX_FLAGS "${SANITY_FLAGS} -O2")

if(ENABLE_CPU_DEVICE)
    target_compile_definitions(peakperf PUBLIC DEVICE_CPU_ENABLED)

    set_source_files_properties(${ARCH_CPU_DIR}/arch.cpp        PROPERTIES COMPILE_FLAGS "-fopenmp")
    set_source_files_properties(${ARCH_CPU_DIR}/arch_sse.cpp    PROPERTIES COMPILE_FLAGS "-msse -fopenmp")
    set_source_files_properties(${ARCH_CPU_DIR}/arch_avx.cpp    PROPERTIES COMPILE_FLAGS "-mavx -fopenmp")
    set_source_files_properties(${ARCH_CPU_DIR}/arch_avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -fopenmp")

    add_library(cpu_device STATIC ${CPUFETCH_DIR}/cpufetch.cpp ${CPUFETCH_DIR}/cpuid.cpp ${CPUFETCH_DIR}/uarch.cpp ${ARCH_CPU_DIR}/arch.cpp ${ARCH_CPU_DIR}/arch_sse.cpp ${ARCH_CPU_DIR}/arch_avx.cpp ${ARCH_CPU_DIR}/arch_avx512.cpp)
    target_link_libraries(peakperf -lm -fopenmp cpu_device 128_6 128_8 256_5 256_6_nofma 256_6 256_8 256_10 512_12 512_8)

    # architecture specific build
    set_source_files_properties(${ARCH_CPU_DIR}/128_6.cpp	PROPERTIES COMPILE_FLAGS "-DSSE_128_6       -msse")
    set_source_files_properties(${ARCH_CPU_DIR}/128_8.cpp       PROPERTIES COMPILE_FLAGS "-DSSE_128_8       -msse")
    set_source_files_properties(${ARCH_CPU_DIR}/256_5.cpp	PROPERTIES COMPILE_FLAGS "-DAVX_256_5       -mavx2 -mfma")
    set_source_files_properties(${ARCH_CPU_DIR}/256_6.cpp	PROPERTIES COMPILE_FLAGS "-DAVX_256_6       -mavx2 -mfma")
    set_source_files_properties(${ARCH_CPU_DIR}/256_6_nofma.cpp PROPERTIES COMPILE_FLAGS "-DAVX_256_6_NOFMA -mavx")
    set_source_files_properties(${ARCH_CPU_DIR}/256_8.cpp       PROPERTIES COMPILE_FLAGS "-DAVX_256_8       -mavx2 -mfma")
    set_source_files_properties(${ARCH_CPU_DIR}/256_10.cpp      PROPERTIES COMPILE_FLAGS "-DAVX_256_10      -mavx2 -mfma")
    set_source_files_properties(${ARCH_CPU_DIR}/512_8.cpp       PROPERTIES COMPILE_FLAGS "-DAVX_512_8       -mavx512f")
    set_source_files_properties(${ARCH_CPU_DIR}/512_12.cpp      PROPERTIES COMPILE_FLAGS "-DAVX_512_12      -mavx512f")

    add_library(128_6       STATIC ${ARCH_CPU_DIR}/128_6.cpp)
    add_library(128_8       STATIC ${ARCH_CPU_DIR}/128_8.cpp)
    add_library(256_5       STATIC ${ARCH_CPU_DIR}/256_5.cpp)
    add_library(256_6       STATIC ${ARCH_CPU_DIR}/256_6.cpp)
    add_library(256_6_nofma STATIC ${ARCH_CPU_DIR}/256_6_nofma.cpp)
    add_library(256_8       STATIC ${ARCH_CPU_DIR}/256_8.cpp)
    add_library(256_10      STATIC ${ARCH_CPU_DIR}/256_10.cpp)
    add_library(512_8       STATIC ${ARCH_CPU_DIR}/512_8.cpp)
    add_library(512_12      STATIC ${ARCH_CPU_DIR}/512_12.cpp)
endif()

if(ENABLE_GPU_DEVICE)
    target_compile_definitions(peakperf PUBLIC DEVICE_GPU_ENABLED)

    # https://en.wikipedia.org/w/index.php?title=CUDA&section=5#GPUs_supported
    # https://raw.githubusercontent.com/PointCloudLibrary/pcl/master/cmake/pcl_find_cuda.cmake
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.0")
      set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86)
    elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "10.0")
      set(CMAKE_CUDA_ARCHITECTURES 30 32 35 37 50 52 53 60 61 62 70 72 75)
    elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "9.0")
      set(CMAKE_CUDA_ARCHITECTURES 30 32 35 37 50 52 53 60 61 62 70 72)
    elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "8.0")
      set(CMAKE_CUDA_ARCHITECTURES 20 21 30 32 35 37 50 52 53 60 61 62)
    endif()

    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
    add_library(gpu_device STATIC ${GPU_DIR}/arch.cu ${GPU_DIR}/kernel.cu)

    target_link_libraries(peakperf gpu_device)
endif()

install(TARGETS peakperf DESTINATION bin)

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
endif()

message(STATUS "----------------------")
message(STATUS "peakperf build report:")
if(ENABLE_CPU_DEVICE)
  message(STATUS "CPU mode: ${BoldGreen}ON${ColourReset}")
else()
  message(STATUS "CPU mode: ${BoldRed}OFF${ColourReset}")
endif()
if(ENABLE_GPU_DEVICE)
  message(STATUS "GPU mode: ${BoldGreen}ON${ColourReset}")
else()
  message(STATUS "GPU mode: ${BoldRed}OFF${ColourReset}")
endif()
message(STATUS "----------------------")

