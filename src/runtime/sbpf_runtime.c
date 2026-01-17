//===- sbpf_runtime.c - sBPF Runtime Library Implementation ---------------===//
//
// Runtime support for sBPF programs running on RISC-V.
//
//===----------------------------------------------------------------------===//

#include "sbpf_runtime.h"

#ifndef __riscv
/* Only include standard libs on host builds */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#endif

//===----------------------------------------------------------------------===//
// Global Runtime Context
//===----------------------------------------------------------------------===//

sbpf_runtime_ctx_t* __sbpf_ctx = NULL;

//===----------------------------------------------------------------------===//
// Default Abort Handler
//===----------------------------------------------------------------------===//

#ifdef __riscv
/* Freestanding abort - just infinite loop */
static void default_abort_handler(const char* msg) {
    (void)msg;  /* Unused in freestanding mode */
    while(1) { /* Halt */ }
}

void __sbpf_abort(void) {
    while(1) { /* Halt */ }
}
#else
/* Host mode - use stdio */
static void default_abort_handler(const char* msg) {
    fprintf(stderr, "[SBPF ABORT] %s\n", msg);
    abort();
}

void __sbpf_abort(void) {
    if (__sbpf_ctx && __sbpf_ctx->abort_handler) {
        __sbpf_ctx->abort_handler("Memory access violation");
    } else {
        default_abort_handler("Memory access violation");
    }
}
#endif

//===----------------------------------------------------------------------===//
// Runtime Initialization
//===----------------------------------------------------------------------===//

void sbpf_runtime_init(sbpf_runtime_ctx_t* ctx) {
    __sbpf_ctx = ctx;
#ifndef __riscv
    if (ctx && !ctx->abort_handler) {
        ctx->abort_handler = default_abort_handler;
    }
#endif
}

#ifndef __riscv
sbpf_runtime_ctx_t* sbpf_runtime_create_default(void) {
    sbpf_runtime_ctx_t* ctx = (sbpf_runtime_ctx_t*)calloc(1, sizeof(sbpf_runtime_ctx_t));
    if (!ctx) return NULL;
    
    // Allocate stack memory
    ctx->stack_base = (uint8_t*)aligned_alloc(8, SBPF_STACK_SIZE);
    if (!ctx->stack_base) {
        free(ctx);
        return NULL;
    }
    
    // Initialize stack to zero
    memset(ctx->stack_base, 0, SBPF_STACK_SIZE);
    
    // Set default abort handler
    ctx->abort_handler = default_abort_handler;
    
    return ctx;
}

void sbpf_runtime_destroy(sbpf_runtime_ctx_t* ctx) {
    if (!ctx) return;
    
    if (ctx->stack_base) {
        free(ctx->stack_base);
    }
    if (ctx->heap_base) {
        free(ctx->heap_base);
    }
    free(ctx);
    
    if (__sbpf_ctx == ctx) {
        __sbpf_ctx = NULL;
    }
}
#endif

uint64_t sbpf_get_stack_pointer(void) {
    // Return the sBPF virtual stack pointer (top of stack)
    return SBPF_STACK_VADDR_BASE;
}

//===----------------------------------------------------------------------===//
// Address Translation Helper
//===----------------------------------------------------------------------===//

typedef enum {
    REGION_INVALID = 0,
    REGION_STACK,
    REGION_HEAP,
    REGION_INPUT
} memory_region_t;

/// Translate sBPF virtual address to actual pointer
/// Returns NULL if address is invalid
static uint8_t* translate_address(uint64_t vaddr, size_t access_size, 
                                   memory_region_t* region, int write_access) {
    if (!__sbpf_ctx) {
        return NULL;
    }
    
    // Check stack region: [SBPF_STACK_VADDR_LOW, SBPF_STACK_VADDR_BASE)
    if (vaddr >= SBPF_STACK_VADDR_LOW && vaddr < SBPF_STACK_VADDR_BASE) {
        // Check if access is within bounds
        if (vaddr + access_size > SBPF_STACK_VADDR_BASE) {
            return NULL;  // Access crosses stack boundary
        }
        
        // Calculate offset from stack base
        // Stack grows down, so offset from base
        uint64_t offset = vaddr - SBPF_STACK_VADDR_LOW;
        
        if (region) *region = REGION_STACK;
        return __sbpf_ctx->stack_base + offset;
    }
    
    // Check heap region (if configured)
    if (__sbpf_ctx->heap_base && __sbpf_ctx->heap_size > 0) {
        uint64_t heap_end = __sbpf_ctx->heap_vaddr + __sbpf_ctx->heap_size;
        if (vaddr >= __sbpf_ctx->heap_vaddr && vaddr < heap_end) {
            if (vaddr + access_size > heap_end) {
                return NULL;  // Access crosses heap boundary
            }
            
            uint64_t offset = vaddr - __sbpf_ctx->heap_vaddr;
            if (region) *region = REGION_HEAP;
            return __sbpf_ctx->heap_base + offset;
        }
    }
    
    // Check input region (if configured) - read only!
    if (__sbpf_ctx->input_base && __sbpf_ctx->input_size > 0) {
        uint64_t input_end = __sbpf_ctx->input_vaddr + __sbpf_ctx->input_size;
        if (vaddr >= __sbpf_ctx->input_vaddr && vaddr < input_end) {
            if (write_access) {
                return NULL;  // Cannot write to input region
            }
            if (vaddr + access_size > input_end) {
                return NULL;  // Access crosses input boundary
            }
            
            uint64_t offset = vaddr - __sbpf_ctx->input_vaddr;
            if (region) *region = REGION_INPUT;
            return (uint8_t*)__sbpf_ctx->input_base + offset;
        }
    }
    
    // Invalid address
    if (region) *region = REGION_INVALID;
    return NULL;
}

//===----------------------------------------------------------------------===//
// Memory Load Functions
//===----------------------------------------------------------------------===//

uint8_t __sbpf_load_8(uint64_t vaddr) {
    uint8_t* ptr = translate_address(vaddr, 1, NULL, 0);
    if (!ptr) {
        __sbpf_abort();
        return 0;
    }
    return *ptr;
}

uint16_t __sbpf_load_16(uint64_t vaddr) {
    if (vaddr & 0x1) {
        __sbpf_abort();
        return 0;
    }
    
    uint8_t* ptr = translate_address(vaddr, 2, NULL, 0);
    if (!ptr) {
        __sbpf_abort();
        return 0;
    }
    return *(uint16_t*)ptr;
}

uint32_t __sbpf_load_32(uint64_t vaddr) {
    if (vaddr & 0x3) {
        __sbpf_abort();
        return 0;
    }
    
    uint8_t* ptr = translate_address(vaddr, 4, NULL, 0);
    if (!ptr) {
        __sbpf_abort();
        return 0;
    }
    return *(uint32_t*)ptr;
}

uint64_t __sbpf_load_64(uint64_t vaddr) {
    if (vaddr & 0x7) {
        __sbpf_abort();
        return 0;
    }
    
    uint8_t* ptr = translate_address(vaddr, 8, NULL, 0);
    if (!ptr) {
        __sbpf_abort();
        return 0;
    }
    return *(uint64_t*)ptr;
}

//===----------------------------------------------------------------------===//
// Memory Store Functions
//===----------------------------------------------------------------------===//

void __sbpf_store_8(uint64_t vaddr, uint8_t value) {
    uint8_t* ptr = translate_address(vaddr, 1, NULL, 1);
    if (!ptr) {
        __sbpf_abort();
        return;
    }
    *ptr = value;
}

void __sbpf_store_16(uint64_t vaddr, uint16_t value) {
    if (vaddr & 0x1) {
        __sbpf_abort();
        return;
    }
    
    uint8_t* ptr = translate_address(vaddr, 2, NULL, 1);
    if (!ptr) {
        __sbpf_abort();
        return;
    }
    *(uint16_t*)ptr = value;
}

void __sbpf_store_32(uint64_t vaddr, uint32_t value) {
    if (vaddr & 0x3) {
        __sbpf_abort();
        return;
    }
    
    uint8_t* ptr = translate_address(vaddr, 4, NULL, 1);
    if (!ptr) {
        __sbpf_abort();
        return;
    }
    *(uint32_t*)ptr = value;
}

void __sbpf_store_64(uint64_t vaddr, uint64_t value) {
    if (vaddr & 0x7) {
        __sbpf_abort();
        return;
    }
    
    uint8_t* ptr = translate_address(vaddr, 8, NULL, 1);
    if (!ptr) {
        __sbpf_abort();
        return;
    }
    *(uint64_t*)ptr = value;
}

#ifndef __riscv
//===----------------------------------------------------------------------===//
// Syscall Support (Simple Hash Table) - Host only
//===----------------------------------------------------------------------===//

#define MAX_SYSCALLS 64

typedef struct {
    const char* name;
    sbpf_syscall_fn handler;
} syscall_entry_t;

static syscall_entry_t syscall_table[MAX_SYSCALLS];
static int syscall_count = 0;

void sbpf_register_syscall(const char* name, sbpf_syscall_fn handler) {
    if (syscall_count >= MAX_SYSCALLS) {
        fprintf(stderr, "[SBPF] Warning: syscall table full\n");
        return;
    }
    
    // Check for duplicate
    for (int i = 0; i < syscall_count; i++) {
        if (strcmp(syscall_table[i].name, name) == 0) {
            syscall_table[i].handler = handler;
            return;
        }
    }
    
    // Add new entry
    syscall_table[syscall_count].name = name;
    syscall_table[syscall_count].handler = handler;
    syscall_count++;
}

sbpf_syscall_fn sbpf_lookup_syscall(const char* name) {
    for (int i = 0; i < syscall_count; i++) {
        if (strcmp(syscall_table[i].name, name) == 0) {
            return syscall_table[i].handler;
        }
    }
    return NULL;
}

//===----------------------------------------------------------------------===//
// Built-in Syscalls (Solana-compatible stubs)
//===----------------------------------------------------------------------===//

// sol_log - Print a message (for debugging)
static uint64_t syscall_sol_log(uint64_t msg_ptr, uint64_t msg_len, 
                                 uint64_t r3, uint64_t r4, uint64_t r5) {
    (void)r3; (void)r4; (void)r5;
    
    // Translate message pointer
    uint8_t* ptr = translate_address(msg_ptr, msg_len, NULL, 0);
    if (ptr && msg_len > 0) {
        printf("[SOL_LOG] %.*s\n", (int)msg_len, (char*)ptr);
    }
    return 0;
}

// sol_log_64 - Print a 64-bit value
static uint64_t syscall_sol_log_64(uint64_t arg1, uint64_t arg2, uint64_t arg3,
                                    uint64_t arg4, uint64_t arg5) {
    printf("[SOL_LOG_64] %llu %llu %llu %llu %llu\n",
           (unsigned long long)arg1, (unsigned long long)arg2,
           (unsigned long long)arg3, (unsigned long long)arg4,
           (unsigned long long)arg5);
    return 0;
}

// Register built-in syscalls
__attribute__((constructor))
static void register_builtin_syscalls(void) {
    sbpf_register_syscall("sol_log", syscall_sol_log);
    sbpf_register_syscall("sol_log_64", syscall_sol_log_64);
}

#endif /* __riscv */
