#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${SCRIPT_DIR}/../runtime"
BUILD_DIR_HOST="${RUNTIME_DIR}/build-host"
BUILD_DIR_RISCV="${RUNTIME_DIR}/build-riscv"

echo "=== Building sBPF Runtime Library ==="
echo ""

# Build for host (x86_64) - for testing
echo "[1/2] Building for host (x86_64)..."
mkdir -p "${BUILD_DIR_HOST}"
cd "${BUILD_DIR_HOST}"
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

echo ""
echo "Host build complete:"
echo "  Static library: ${BUILD_DIR_HOST}/libsbpf_runtime.a"
echo "  Shared library: ${BUILD_DIR_HOST}/libsbpf_runtime.so"
echo "  Test binary:    ${BUILD_DIR_HOST}/test_runtime"
echo ""

# Run host tests
echo "Running host tests..."
./test_runtime
echo ""

# Build for RISC-V target
echo "[2/2] Building for RISC-V (riscv64)..."

# Check if RISC-V toolchain is available
if ! command -v riscv64-linux-gnu-gcc &> /dev/null; then
    echo "WARNING: RISC-V toolchain not found (riscv64-linux-gnu-gcc)"
    echo "Skipping RISC-V build. To install:"
    echo "  sudo apt-get install gcc-riscv64-linux-gnu"
    echo ""
    echo "Host build completed successfully. Runtime library ready for testing."
    exit 0
fi

mkdir -p "${BUILD_DIR_RISCV}"
cd "${BUILD_DIR_RISCV}"

echo "Building RISC-V runtime (freestanding mode)..."

# Compile runtime as freestanding object file (no libc dependency)
riscv64-linux-gnu-gcc -c "${RUNTIME_DIR}/sbpf_runtime.c" \
    -o sbpf_runtime.o \
    -I"${RUNTIME_DIR}" \
    -O2 -march=rv64gc -mabi=lp64d \
    -Wall -Wextra

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to compile runtime for RISC-V"
    exit 1
fi

# Create static library
riscv64-linux-gnu-ar rcs libsbpf_runtime_riscv.a sbpf_runtime.o

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create static library"
    exit 1
fi

echo ""
echo "RISC-V build complete:"
echo "  Static library: ${BUILD_DIR_RISCV}/libsbpf_runtime_riscv.a"
echo "  Object file:    ${BUILD_DIR_RISCV}/sbpf_runtime.o"
echo ""

# Summary
echo "=== Build Summary ==="
echo ""
echo "To link with generated RISC-V code:"
echo "  riscv64-linux-gnu-gcc program.o \\"
echo "    -L${BUILD_DIR_RISCV} -lsbpf_runtime_riscv \\"
echo "    -o program"
echo ""
echo "Or use the helper script:"
echo "  ${SCRIPT_DIR}/sbpf-link.sh program.o program"
echo ""
