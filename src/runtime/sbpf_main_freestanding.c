//===- sbpf_main_freestanding.c - Freestanding Entry Point ----------------===//
//
// Freestanding main entry point for sBPF AOT-compiled programs.
// No standard library dependencies - uses raw system calls for output.
//
//===----------------------------------------------------------------------===//

#include "sbpf_runtime.h"

// Simple write syscall for QEMU Linux user mode
static void write_str(const char* s) {
    // Calculate string length
    size_t len = 0;
    while (s[len]) len++;
    
    // Linux syscall: write(fd=1, buf, count)
    register long a0 __asm__("a0") = 1;       // stdout
    register long a1 __asm__("a1") = (long)s;
    register long a2 __asm__("a2") = len;
    register long a7 __asm__("a7") = 64;      // SYS_write
    __asm__ __volatile__(
        "ecall"
        : "+r"(a0)
        : "r"(a1), "r"(a2), "r"(a7)
        : "memory"
    );
}

static void write_hex(uint64_t val) {
    char buf[19];  // "0x" + 16 hex digits + null
    buf[0] = '0';
    buf[1] = 'x';
    for (int i = 15; i >= 0; i--) {
        int nibble = (val >> (i * 4)) & 0xF;
        buf[17 - i] = nibble < 10 ? '0' + nibble : 'a' + nibble - 10;
    }
    buf[18] = '\0';
    write_str(buf);
}

// Static stack allocation for freestanding mode
static uint8_t __sbpf_stack[SBPF_STACK_SIZE] __attribute__((aligned(8)));
static sbpf_runtime_ctx_t __sbpf_default_ctx;

// Weak symbol - will be overridden by actual sBPF function
__attribute__((weak))
uint64_t entrypoint(uint64_t r1, uint64_t r2, uint64_t r3, uint64_t r4, uint64_t r5) {
    (void)r1; (void)r2; (void)r3; (void)r4; (void)r5;
    return 0xDEADBEEF;  // Error marker
}

// Minimal main for freestanding execution
void _start(void) {
    // Initialize runtime context
    __sbpf_default_ctx.stack_base = __sbpf_stack;
    __sbpf_default_ctx.heap_base = NULL;
    __sbpf_default_ctx.heap_size = 0;
    __sbpf_default_ctx.heap_vaddr = 0;
    __sbpf_default_ctx.input_base = NULL;
    __sbpf_default_ctx.input_size = 0;
    __sbpf_default_ctx.input_vaddr = 0;
    __sbpf_default_ctx.abort_handler = NULL;
    
    __sbpf_ctx = &__sbpf_default_ctx;
    
    write_str("=== sBPF AOT Execution Start ===\n");
    write_str("Stack base: ");
    write_hex((uint64_t)__sbpf_stack);
    write_str("\n");
    
    // Initialize stack with some test data at offset 0
    // This allows the program to read from r1 when we pass SBPF_STACK_VADDR_LOW
    uint64_t* stack_data = (uint64_t*)__sbpf_stack;
    stack_data[0] = 0x1234567890ABCDEFULL;  // Test value at [r1+0]
    stack_data[1] = 0xDEADBEEFCAFEBABEULL;  // Test value at [r1+8]
    
    // Call sBPF entry function
    // r1 = valid readable address (bottom of stack)
    // The program expects r1 to point to input data
    uint64_t input_addr = SBPF_STACK_VADDR_BASE - SBPF_STACK_SIZE;  // = SBPF_STACK_VADDR_LOW
    
    write_str("Calling entrypoint with r1=");
    write_hex(input_addr);
    write_str("\n");
    
    uint64_t result = entrypoint(input_addr, 0, 0, 0, 0);
    
    write_str("Execution complete! Result: ");
    write_hex(result);
    write_str("\n=== sBPF AOT Execution End ===\n");
    
    // Exit using Linux syscall directly - exit(result & 0xFF)
    // Load result into a0, syscall number 93 into a7, then ecall
    register long a0 __asm__("a0") = result & 0xFF;
    register long a7 __asm__("a7") = 93;
    __asm__ __volatile__ (
        "ecall"
        : /* no outputs */
        : "r" (a0), "r" (a7)
        : "memory"
    );
    
    // Should never reach here
    __builtin_unreachable();
}
