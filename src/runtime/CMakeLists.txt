cmake_minimum_required(VERSION 3.16)
project(sbpf_runtime C)

# Runtime library source
set(RUNTIME_SOURCES
    sbpf_runtime.c
)

# Build as static library (for linking into final binary)
add_library(sbpf_runtime_static STATIC ${RUNTIME_SOURCES})
target_include_directories(sbpf_runtime_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(sbpf_runtime_static PROPERTIES
    OUTPUT_NAME sbpf_runtime
    C_STANDARD 11
)

# Build as shared library (for dynamic linking)
add_library(sbpf_runtime_shared SHARED ${RUNTIME_SOURCES})
target_include_directories(sbpf_runtime_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(sbpf_runtime_shared PROPERTIES
    OUTPUT_NAME sbpf_runtime
    C_STANDARD 11
)

# For RISC-V cross-compilation (optional)
# Use: cmake -DCMAKE_TOOLCHAIN_FILE=riscv64-toolchain.cmake ..
option(BUILD_RISCV "Build for RISC-V target" OFF)

if(BUILD_RISCV)
    message(STATUS "Building for RISC-V target")
    # When cross-compiling, you typically only want static lib
    set_target_properties(sbpf_runtime_static PROPERTIES
        OUTPUT_NAME sbpf_runtime_riscv
    )
endif()

# Installation rules
install(TARGETS sbpf_runtime_static sbpf_runtime_shared
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES sbpf_runtime.h DESTINATION include)

# Simple test program
option(BUILD_TESTS "Build test programs" ON)

if(BUILD_TESTS AND NOT BUILD_RISCV)
    add_executable(test_runtime test_runtime.c)
    target_link_libraries(test_runtime sbpf_runtime_static)
endif()
