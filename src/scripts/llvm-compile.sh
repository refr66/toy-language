#!/bin/bash

# 1. 自动定位路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
LLVM_SRC="$PROJECT_ROOT/third_party/llvm-project"
LLVM_BUILD="$LLVM_SRC/build"

echo "Using The Holy Trinity: Mold + Ccache + Ninja"

mkdir -p "$LLVM_BUILD"

# 2. 配置 CMake
# -DLLVM_USE_LINKER=mold: 使用极速链接器 mold
# -DCMAKE_CXX_COMPILER_LAUNCHER=ccache: 使用编译器缓存
# -DCMAKE_C_COMPILER_LAUNCHER=ccache
cmake -G Ninja \
  -S "$LLVM_SRC/llvm" \
  -B "$LLVM_BUILD" \
  -DLLVM_ENABLE_PROJECTS="mlir;lld" \
  -DLLVM_TARGETS_TO_BUILD="RISCV" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  \
  -DLLVM_USE_LINKER=mold \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_BUILD_TESTS=OFF \
  \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++

# 3. 执行编译
# 使用 -j 指定核心数。如果内存小于 32G，建议留 2 个核心，防止系统卡死
# 例如：ninja -j $(($(nproc) - 2)) -C "$LLVM_BUILD"
ninja -C "$LLVM_BUILD"
