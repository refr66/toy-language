cmake_minimum_required(VERSION 3.20)
project(SbpfDialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#===----------------------------------------------------------------------===//
# Find LLVM/MLIR
#===----------------------------------------------------------------------===//

# Point this to your LLVM 22 build directory
set(LLVM_DIR "$ENV{HOME}/Documents/Archive/code/MLIR/base/llvm-project/build/lib/cmake/llvm" CACHE PATH "")
set(MLIR_DIR "$ENV{HOME}/Documents/Archive/code/MLIR/base/llvm-project/build/lib/cmake/mlir" CACHE PATH "")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVM: ${LLVM_DIR}")
message(STATUS "Using MLIR: ${MLIR_DIR}")
message(STATUS "MLIR_CMAKE_DIR: ${MLIR_CMAKE_DIR}")
message(STATUS "LLVM_CMAKE_DIR: ${LLVM_CMAKE_DIR}")

# Add MLIR/LLVM cmake module paths BEFORE including modules
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Include MLIR CMake modules
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# Add LLVM definitions
add_definitions(${LLVM_DEFINITIONS})

#===----------------------------------------------------------------------===//
# TableGen Targets
#===----------------------------------------------------------------------===//

# Create output directory for generated files
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include/sbpf)

# Generate Ops
set(LLVM_TARGET_DEFINITIONS include/sbpf/SbpfOps.td)

mlir_tablegen(include/sbpf/SbpfOps.h.inc -gen-op-decls)
mlir_tablegen(include/sbpf/SbpfOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/sbpf/SbpfDialect.h.inc -gen-dialect-decls -dialect=sbpf)
mlir_tablegen(include/sbpf/SbpfDialect.cpp.inc -gen-dialect-defs -dialect=sbpf)

add_public_tablegen_target(SbpfOpsIncGen)

# Generate Passes
set(LLVM_TARGET_DEFINITIONS include/sbpf/SbpfPasses.td)

mlir_tablegen(include/sbpf/SbpfPasses.h.inc -gen-pass-decls -name Sbpf)

add_public_tablegen_target(SbpfPassesIncGen)

#===----------------------------------------------------------------------===//
# Dialect Library
#===----------------------------------------------------------------------===//

add_mlir_dialect_library(MLIRSbpfDialect
    lib/SbpfDialect.cpp
    lib/SbpfOps.cpp
    lib/SbpfPasses.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/sbpf
    
    DEPENDS
    SbpfOpsIncGen
    SbpfPassesIncGen
    
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRControlFlowDialect
    MLIRSCFDialect
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
)

#===----------------------------------------------------------------------===//
# Standalone Tool (sbpf-opt)
#===----------------------------------------------------------------------===//

add_llvm_executable(sbpf-opt
    tools/sbpf-opt.cpp
)

target_link_libraries(sbpf-opt PRIVATE
    MLIRSbpfDialect
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRControlFlowDialect
    MLIRSCFDialect
    MLIRIndexDialect
    MLIRUBDialect
    MLIRFuncTransforms
    MLIRMemRefTransforms
    MLIRArithTransforms
    # LLVM Dialect and Conversion
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
    MLIRConvertToLLVMInterface
    MLIRConvertToLLVMPass
    MLIRFuncToLLVM
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRMemRefToLLVM
    MLIRIndexToLLVM
    MLIRSCFToControlFlow
    MLIRReconcileUnrealizedCasts
)

mlir_check_all_link_libraries(sbpf-opt)

#===----------------------------------------------------------------------===//
# Installation
#===----------------------------------------------------------------------===//

install(TARGETS MLIRSbpfDialect sbpf-opt
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/sbpf
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.td"
)
