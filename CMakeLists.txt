cmake_minimum_required(VERSION 3.14)
project(llm_rerank_cpu_simd CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------
# build type & includes
# ----------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

if (MSVC)
  add_compile_options(/O2)
  # Enable OpenMP on MSVC
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    target_compile_options(simd_ops PRIVATE /openmp)
  endif()
else()
  add_compile_options(-O3 -ffast-math -march=native)
endif()

# ----------------------------------------
# SIMD source selection
# ----------------------------------------
set(SIMD_SCALAR_SRC
  src/simd/simd_dispatch.cpp
  src/simd/rmsnorm_scalar.cpp
  src/simd/rope_scalar.cpp
  src/simd/pack_qkv_scalar.cpp
  src/simd/rmsnorm_scalar_novec.cpp
)

set(SIMD_X86_SRC
  src/simd/rmsnorm_avx2.cpp
  src/simd/rope_avx2.cpp
  src/simd/pack_qkv_avx2.cpp
)

set(SIMD_ARM_SRC
  src/simd/rmsnorm_neon.cpp
  src/simd/rope_neon.cpp
  src/simd/pack_qkv_neon.cpp
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|X86_64|amd64|AMD64")
  add_definitions(-DSIMD_X86)
  set(SIMD_SRC ${SIMD_SCALAR_SRC} ${SIMD_X86_SRC})
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  add_definitions(-DSIMD_ARM64)
  add_compile_options(-mcpu=apple-m1)
  set(SIMD_SRC ${SIMD_SCALAR_SRC} ${SIMD_ARM_SRC})
else()
  set(SIMD_SRC ${SIMD_SCALAR_SRC})
endif()

# ----------------------------------------
# PyTorch extension only
# ----------------------------------------
#   export CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")

set(Python3_EXECUTABLE "/opt/miniconda3/bin/python")
set(Python3_ROOT_DIR  "/opt/miniconda3")
find_package(Python3 COMPONENTS Development REQUIRED)

find_package(Torch REQUIRED)

add_library(simd_ops SHARED
  src/ops/simd_ops.cpp
  src/thread_scheduler.cpp
  ${SIMD_SRC}
)

target_link_libraries(simd_ops "${TORCH_LIBRARIES}")
target_include_directories(simd_ops PRIVATE ${TORCH_INCLUDE_DIRS})

if (OpenMP_CXX_FOUND)
  target_link_libraries(simd_ops OpenMP::OpenMP_CXX)
endif()

set_target_properties(simd_ops PROPERTIES
  PREFIX ""
  OUTPUT_NAME "simd_ops"
)
