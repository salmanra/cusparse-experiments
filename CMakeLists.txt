cmake_minimum_required(VERSION 3.18)
project(cusparse_spmm CUDA CXX)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit (provides cuSPARSE)
find_package(CUDAToolkit REQUIRED)

# Create executable
add_executable(spmm_demo
    src/main.cu
)

# Link cuSPARSE library
target_link_libraries(spmm_demo
    CUDA::cusparse
    CUDA::cudart
)

# Include directories
target_include_directories(spmm_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Set CUDA architectures (adjust for your GPU)
set_target_properties(spmm_demo PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

# Blocked-ELL FP16 Tensor Core variant
add_executable(spmm_bell
    src/main_bell.cu
)

target_link_libraries(spmm_bell
    CUDA::cusparse
    CUDA::cudart
)

target_include_directories(spmm_bell PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(spmm_bell PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

# BSR FP16 Tensor Core variant
add_executable(spmm_bsr_fp16
    src/main_bsr_fp16.cu
)

target_link_libraries(spmm_bsr_fp16
    CUDA::cusparse
    CUDA::cudart
)

target_include_directories(spmm_bsr_fp16 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(spmm_bsr_fp16 PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

# cuBLAS GEMM profiler
add_executable(gemm_demo
    src/main_gemm.cu
)

target_link_libraries(gemm_demo
    CUDA::cublas
    CUDA::cudart
)

target_include_directories(gemm_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(gemm_demo PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

# Optional: Enable verbose output during build
set(CMAKE_VERBOSE_MAKEFILE OFF)
