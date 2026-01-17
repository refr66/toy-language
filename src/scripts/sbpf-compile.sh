#!/bin/bash
#===----------------------------------------------------------------------===//
# sbpf-compile.sh - Complete sBPF to RISC-V compilation pipeline
#
# Usage: ./sbpf-compile.sh input.mlir [output_prefix]
#
# This script performs the full compilation:
#   1. Lower sbpf.load/store to runtime function calls
#   2. Convert to LLVM dialect
#   3. Translate to LLVM IR
#   4. Compile to RISC-V assembly
#   5. Optionally compile to object file
#===----------------------------------------------------------------------===//

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBPF_OPT="${SCRIPT_DIR}/../mlir-dialect/build/sbpf-opt"
LLVM_BUILD="third_party/llvm-project/build"
MLIR_TRANSLATE="${LLVM_BUILD}/mlir-translate"
LLC="${LLVM_BUILD}/llc"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.mlir> [output_prefix]"
    echo ""
    echo "Options:"
    echo "  input.mlir     - Input MLIR file with sbpf dialect"
    echo "  output_prefix  - Output file prefix (default: based on input)"
    echo ""
    echo "Output files:"
    echo "  <prefix>.lowered.mlir  - After lowering sbpf ops"
    echo "  <prefix>.llvm.mlir     - LLVM dialect"
    echo "  <prefix>.ll            - LLVM IR"
    echo "  <prefix>.s             - RISC-V assembly"
    echo "  <prefix>.o             - RISC-V object file"
    exit 1
fi

INPUT="$1"
PREFIX="${2:-${INPUT%.mlir}}"

echo "=== sBPF to RISC-V Compiler ==="
echo "Input: $INPUT"
echo "Output prefix: $PREFIX"
echo ""

# Step 1: Lower sbpf.load/store to runtime function calls
echo "[1/5] Lowering sbpf.load/store..."
"${SBPF_OPT}" "$INPUT" --lower-sbpf-mem -o "${PREFIX}.lowered.mlir"

# Step 2: Apply optimizations and convert to LLVM (all in one pipeline)
echo "[2/5] Optimizing and converting to LLVM dialect..."
"${SBPF_OPT}" "${PREFIX}.lowered.mlir" \
  --mem2reg \
  --canonicalize \
  --cse \
  --convert-to-llvm \
  -o "${PREFIX}.llvm.mlir.tmp"

# Remove ub.poison operations (replace with llvm.mlir.poison or zero)
sed 's/ub\.poison/llvm.mlir.poison/g' "${PREFIX}.llvm.mlir.tmp" > "${PREFIX}.llvm.mlir"
rm -f "${PREFIX}.llvm.mlir.tmp"

# Step 3: Translate to LLVM IR
echo "[3/5] Translating to LLVM IR..."
"${MLIR_TRANSLATE}" --mlir-to-llvmir "${PREFIX}.llvm.mlir" -o "${PREFIX}.ll"

# Step 4: Compile to RISC-V assembly
echo "[4/5] Compiling to RISC-V assembly..."
"${LLC}" -march=riscv64 -mattr=+m "${PREFIX}.ll" -o "${PREFIX}.s"

# Step 5: Compile to object file
echo "[5/5] Compiling to object file..."
"${LLC}" -march=riscv64 -target-abi lp64 -mattr=+m,+a,+c -filetype=obj "${PREFIX}.ll" -o "${PREFIX}.o"

echo ""
echo "=== Compilation complete! ==="
echo "Generated files:"
echo "  ${PREFIX}.lowered.mlir - sbpf.load/store -> runtime calls"
echo "  ${PREFIX}.llvm.mlir    - LLVM dialect (after optimization)"
echo "  ${PREFIX}.ll           - LLVM IR"
echo "  ${PREFIX}.s            - RISC-V assembly"
echo "  ${PREFIX}.o            - RISC-V object file"
echo ""
echo "To link with runtime:"
echo "  riscv64-linux-gnu-gcc ${PREFIX}.o -L<runtime_path> -lsbpf_runtime -o ${PREFIX}"
