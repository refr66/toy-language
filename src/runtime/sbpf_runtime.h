//===- sbpf_runtime.h - sBPF Runtime Library Header -----------------------===//
//
// Runtime support for sBPF programs running on RISC-V.
// Provides memory access functions with address validation and translation.
//
//===----------------------------------------------------------------------===//

#ifndef SBPF_RUNTIME_H
#define SBPF_RUNTIME_H

#ifdef __riscv
/* Freestanding mode for RISC-V */
#include "sbpf_types.h"
#else
/* Standard headers for host builds */
#include <stdint.h>
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// sBPF Memory Layout Constants
//===----------------------------------------------------------------------===//

// sBPF uses a virtual address space with these regions:
// - Stack:  0x200000000 - 0x200001000 (4KB, grows down from r10)
// - Heap:   Dynamic, passed via input
// - Input:  Read-only data region

#define SBPF_STACK_VADDR_BASE   0x200000000ULL  // Virtual stack base (top of stack)
#define SBPF_STACK_SIZE         4096            // 4KB stack
#define SBPF_STACK_VADDR_LOW    (SBPF_STACK_VADDR_BASE - SBPF_STACK_SIZE)

//===----------------------------------------------------------------------===//
// Runtime Context
//===----------------------------------------------------------------------===//

/// Runtime context structure - holds the mapping between sBPF virtual 
/// addresses and actual physical addresses.
typedef struct {
    // Stack region
    uint8_t* stack_base;        // Actual stack memory (4KB allocated)
    
    // Heap region (optional)
    uint8_t* heap_base;
    size_t   heap_size;
    uint64_t heap_vaddr;        // sBPF virtual address of heap
    
    // Input data region (read-only)
    const uint8_t* input_base;
    size_t   input_size;
    uint64_t input_vaddr;       // sBPF virtual address of input
    
    // Error handling
    void (*abort_handler)(const char* msg);
} sbpf_runtime_ctx_t;

/// Global runtime context (set before running sBPF code)
extern sbpf_runtime_ctx_t* __sbpf_ctx;

//===----------------------------------------------------------------------===//
// Runtime Initialization
//===----------------------------------------------------------------------===//

/// Initialize the runtime with a given context
void sbpf_runtime_init(sbpf_runtime_ctx_t* ctx);

/// Create a default context with stack only
sbpf_runtime_ctx_t* sbpf_runtime_create_default(void);

/// Free a runtime context
void sbpf_runtime_destroy(sbpf_runtime_ctx_t* ctx);

/// Get the initial stack pointer (r10 value)
uint64_t sbpf_get_stack_pointer(void);

//===----------------------------------------------------------------------===//
// Memory Access Functions (called by generated MLIR code)
//===----------------------------------------------------------------------===//

// Load functions - read from sBPF virtual address
uint8_t  __sbpf_load_8(uint64_t vaddr);
uint16_t __sbpf_load_16(uint64_t vaddr);
uint32_t __sbpf_load_32(uint64_t vaddr);
uint64_t __sbpf_load_64(uint64_t vaddr);

// Store functions - write to sBPF virtual address
void __sbpf_store_8(uint64_t vaddr, uint8_t value);
void __sbpf_store_16(uint64_t vaddr, uint16_t value);
void __sbpf_store_32(uint64_t vaddr, uint32_t value);
void __sbpf_store_64(uint64_t vaddr, uint64_t value);

// Abort function - called on invalid memory access
void __sbpf_abort(void);

//===----------------------------------------------------------------------===//
// Syscall Support (Solana runtime calls)
//===----------------------------------------------------------------------===//

/// Generic syscall handler type
typedef uint64_t (*sbpf_syscall_fn)(uint64_t r1, uint64_t r2, uint64_t r3, 
                                    uint64_t r4, uint64_t r5);

/// Register a syscall handler by name
void sbpf_register_syscall(const char* name, sbpf_syscall_fn handler);

/// Lookup a syscall by name
sbpf_syscall_fn sbpf_lookup_syscall(const char* name);

#ifdef __cplusplus
}
#endif

#endif // SBPF_RUNTIME_H
