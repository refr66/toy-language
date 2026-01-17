#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_LIB_DIR="${SCRIPT_DIR}/../runtime/build-riscv"
RUNTIME_MAIN="${SCRIPT_DIR}/../runtime/sbpf_main_freestanding.c"
RUNTIME_HEADER_DIR="${SCRIPT_DIR}/../runtime"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input.o> <output_binary> [entry_function]"
    echo ""
    echo "Links RISC-V object file with sBPF runtime library and main wrapper"
    echo ""
    echo "Arguments:"
    echo "  entry_function: Optional sBPF entry point function name (default: entrypoint)"
    echo "                  This will be called from the C main() function"
    echo ""
    echo "Example:"
    echo "  $0 program.o program"
    echo "  $0 program.o program function_4d9"
    exit 1
fi

INPUT_OBJ="$1"
OUTPUT_BIN="$2"
ENTRY_FUNCTION="${3:-entrypoint}"  # Default to 'entrypoint' if not specified

# Check if runtime library exists
if [ ! -f "${RUNTIME_LIB_DIR}/libsbpf_runtime_riscv.a" ]; then
    echo "ERROR: Runtime library not found!"
    echo "Expected: ${RUNTIME_LIB_DIR}/libsbpf_runtime_riscv.a"
    echo ""
    echo "Please build the runtime first:"
    echo "  ${SCRIPT_DIR}/build-runtime.sh"
    exit 1
fi

# Check if input exists
if [ ! -f "$INPUT_OBJ" ]; then
    echo "ERROR: Input object file not found: $INPUT_OBJ"
    exit 1
fi

# Check if RISC-V toolchain is available
if ! command -v riscv64-linux-gnu-gcc &> /dev/null; then
    echo "ERROR: RISC-V toolchain not found (riscv64-linux-gnu-gcc)"
    echo "Install with: sudo apt-get install gcc-riscv64-linux-gnu"
    exit 1
fi

echo "=== sBPF RISC-V Linker ==="
echo "Input object:   $INPUT_OBJ"
echo "Entry function: $ENTRY_FUNCTION"
echo "Output binary:  $OUTPUT_BIN"
echo "Runtime lib:    ${RUNTIME_LIB_DIR}/libsbpf_runtime_riscv.a"
echo "Main wrapper:   ${RUNTIME_MAIN}"
echo ""

# Check if we need a wrapper (only if entry function is not 'entrypoint')
if [ "$ENTRY_FUNCTION" = "entrypoint" ]; then
    echo "Entry function is already 'entrypoint', no wrapper needed"
    echo ""
    
    # Link directly without wrapper
    riscv64-linux-gnu-gcc \
        "$INPUT_OBJ" \
        "${RUNTIME_MAIN}" \
        "${RUNTIME_LIB_DIR}/sbpf_runtime.o" \
        -nostdlib \
        -nostartfiles \
        -Wl,--entry=_start \
        -o "$OUTPUT_BIN"
else
    # Create a temporary wrapper that renames the entry function
    TEMP_WRAPPER=$(mktemp /tmp/sbpf_wrapper_XXXXXX.c)
    trap "rm -f $TEMP_WRAPPER" EXIT
    
    cat > "$TEMP_WRAPPER" <<EOF
typedef unsigned long uint64_t;

// Declare the actual sBPF function (mangled name from LLVM)
extern uint64_t $ENTRY_FUNCTION(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

// Create an alias with the expected name 'entrypoint'
uint64_t entrypoint(uint64_t r1, uint64_t r2, uint64_t r3, uint64_t r4, uint64_t r5) {
    return $ENTRY_FUNCTION(r1, r2, r3, r4, r5);
}
EOF
    
    echo "Generated wrapper to call $ENTRY_FUNCTION as entrypoint"
    echo ""
    
    # Link with wrapper
    riscv64-linux-gnu-gcc \
        "$INPUT_OBJ" \
        "$TEMP_WRAPPER" \
        "${RUNTIME_MAIN}" \
        "${RUNTIME_LIB_DIR}/sbpf_runtime.o" \
        -nostdlib \
        -nostartfiles \
        -Wl,--entry=_start \
        -o "$OUTPUT_BIN"
fi

echo ""
echo "=== Link complete ==="
echo "Generated: $OUTPUT_BIN"
echo ""
echo "To run on RISC-V (requires QEMU or hardware):"
echo "  qemu-riscv64 $OUTPUT_BIN"
echo ""
