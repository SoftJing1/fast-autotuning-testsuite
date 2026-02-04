cmake_minimum_required(VERSION 3.20)

# Set compiler to clang
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

project(sAutoTuning CXX)

set(CMAKE_CXX_STANDARD 17)
enable_testing()

# Define kernel paths
set(KERNEL_TEMPLATE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel-template")
set(KERNEL_HOST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel-host")

# ============================================================================
# Find required packages
# ============================================================================
find_package(OpenCL REQUIRED)

# Find nlohmann_json - try different methods
find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
    # Try to find it as a header-only library in common locations
    find_path(NLOHMANN_JSON_INCLUDE_DIR
        NAMES nlohmann/json.hpp
        PATHS
            /usr/include
            /usr/local/include
            ${CMAKE_CURRENT_SOURCE_DIR}/extern/json/include
    )
    if(NLOHMANN_JSON_INCLUDE_DIR)
        message(STATUS "Found nlohmann/json.hpp at: ${NLOHMANN_JSON_INCLUDE_DIR}")
        add_library(nlohmann_json INTERFACE)
        target_include_directories(nlohmann_json INTERFACE ${NLOHMANN_JSON_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "nlohmann_json not found. Please install it or set NLOHMANN_JSON_INCLUDE_DIR")
    endif()
endif()

# ============================================================================
# Build main executables with JSON configuration support
# ============================================================================
add_executable(gaussian kernels/kernel-host/gaussian_host.cc)
add_executable(gemm kernels/kernel-host/gemm_host.cc)

target_include_directories(gaussian PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
target_include_directories(gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

# Pass kernel path to compiler
target_compile_definitions(gaussian PRIVATE KERNEL_TEMPLATE_DIR="${KERNEL_TEMPLATE_DIR}")
target_compile_definitions(gemm PRIVATE KERNEL_TEMPLATE_DIR="${KERNEL_TEMPLATE_DIR}")

# Define OpenCL version to suppress pragma warnings
target_compile_definitions(gaussian PRIVATE CL_TARGET_OPENCL_VERSION=300)
target_compile_definitions(gemm PRIVATE CL_TARGET_OPENCL_VERSION=300)

target_link_libraries(gaussian OpenCL::OpenCL nlohmann_json::nlohmann_json)
target_link_libraries(gemm OpenCL::OpenCL nlohmann_json::nlohmann_json)

# ============================================================================
# Testing Configuration
# ============================================================================
# Add basic tests using pytest/Python testing framework
# Run: cmake --build . && ctest --verbose
# Or:  cd build && python ../tests/test_kernels.py

# Add a custom target to run Python tests
add_custom_target(test_python
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kernels.py
    DEPENDS gaussian gemm
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running Python-based kernel tests"
)