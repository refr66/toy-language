//===- test_runtime.c - Test sBPF Runtime Library -------------------------===//
//
// Simple test to verify the runtime library works correctly.
//
//===----------------------------------------------------------------------===//

#include "sbpf_runtime.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) \
    do { \
        test_count++; \
        printf("Test %d: %s ... ", test_count, name); \
    } while(0)

#define PASS() \
    do { \
        pass_count++; \
        printf("PASS\n"); \
    } while(0)

#define FAIL(msg) \
    do { \
        printf("FAIL: %s\n", msg); \
    } while(0)

// Test basic stack operations
void test_stack_memory(void) {
    TEST("Stack store/load 64-bit");
    
    // Get stack pointer (top of stack)
    uint64_t sp = sbpf_get_stack_pointer();
    
    // Store at SP - 8 (first slot below stack top)
    uint64_t addr = sp - 8;
    __sbpf_store_64(addr, 0xDEADBEEFCAFEBABEULL);
    
    // Load it back
    uint64_t val = __sbpf_load_64(addr);
    if (val == 0xDEADBEEFCAFEBABEULL) {
        PASS();
    } else {
        FAIL("Value mismatch");
    }
    
    // Test different sizes
    TEST("Stack store/load 32-bit");
    addr = sp - 16;
    __sbpf_store_32(addr, 0x12345678);
    if (__sbpf_load_32(addr) == 0x12345678) {
        PASS();
    } else {
        FAIL("Value mismatch");
    }
    
    TEST("Stack store/load 16-bit");
    addr = sp - 20;
    __sbpf_store_16(addr, 0xABCD);
    if (__sbpf_load_16(addr) == 0xABCD) {
        PASS();
    } else {
        FAIL("Value mismatch");
    }
    
    TEST("Stack store/load 8-bit");
    addr = sp - 24;
    __sbpf_store_8(addr, 0x42);
    if (__sbpf_load_8(addr) == 0x42) {
        PASS();
    } else {
        FAIL("Value mismatch");
    }
}

// Test stack boundary
void test_stack_boundary(void) {
    TEST("Stack boundary - valid access at bottom");
    
    uint64_t sp = sbpf_get_stack_pointer();
    uint64_t bottom = sp - SBPF_STACK_SIZE + 8;  // Just inside boundary
    
    __sbpf_store_64(bottom, 0x1234);
    if (__sbpf_load_64(bottom) == 0x1234) {
        PASS();
    } else {
        FAIL("Could not access bottom of stack");
    }
}

// Track if abort was called
static int abort_called = 0;
static void test_abort_handler(const char* msg) {
    printf("(abort called: %s) ", msg);
    abort_called = 1;
    // Don't actually abort for testing
}

// Test invalid access (should trigger abort)
void test_invalid_access(void) {
    // Set custom abort handler
    __sbpf_ctx->abort_handler = test_abort_handler;
    
    TEST("Invalid address detection");
    abort_called = 0;
    
    // Try to access an invalid address (way outside any region)
    __sbpf_load_64(0x12345678);
    
    if (abort_called) {
        PASS();
    } else {
        FAIL("Abort not called for invalid address");
    }
    
    // Restore default handler
    __sbpf_ctx->abort_handler = NULL;
}

// Test multiple stores and loads (simulating a simple program)
void test_simple_program(void) {
    TEST("Simple program simulation");
    
    uint64_t sp = sbpf_get_stack_pointer();
    
    // Simulate: int a = 10; int b = 20; int c = a + b;
    // Store a at SP - 8
    __sbpf_store_64(sp - 8, 10);
    
    // Store b at SP - 16
    __sbpf_store_64(sp - 16, 20);
    
    // Load a and b, compute c
    uint64_t a = __sbpf_load_64(sp - 8);
    uint64_t b = __sbpf_load_64(sp - 16);
    uint64_t c = a + b;
    
    // Store c at SP - 24
    __sbpf_store_64(sp - 24, c);
    
    // Verify
    if (__sbpf_load_64(sp - 24) == 30) {
        PASS();
    } else {
        FAIL("Computation incorrect");
    }
}

// Test heap region (if configured)
void test_heap_region(void) {
    TEST("Heap region access");
    
    // Allocate some heap memory
    uint8_t heap[1024];
    memset(heap, 0, sizeof(heap));
    
    // Configure heap in context
    __sbpf_ctx->heap_base = heap;
    __sbpf_ctx->heap_size = sizeof(heap);
    __sbpf_ctx->heap_vaddr = 0x300000000ULL;  // Example heap vaddr
    
    // Store and load in heap
    uint64_t heap_addr = 0x300000000ULL + 256;  // Offset into heap
    __sbpf_store_64(heap_addr, 0xCAFEBABE);
    
    if (__sbpf_load_64(heap_addr) == 0xCAFEBABE) {
        PASS();
    } else {
        FAIL("Heap access failed");
    }
    
    // Clean up
    __sbpf_ctx->heap_base = NULL;
    __sbpf_ctx->heap_size = 0;
}

int main(void) {
    printf("========================================\n");
    printf("sBPF Runtime Library Tests\n");
    printf("========================================\n\n");
    
    // Initialize runtime
    sbpf_runtime_ctx_t* ctx = sbpf_runtime_create_default();
    if (!ctx) {
        printf("FATAL: Could not create runtime context\n");
        return 1;
    }
    sbpf_runtime_init(ctx);
    
    printf("Stack virtual address range: 0x%llx - 0x%llx\n", 
           (unsigned long long)SBPF_STACK_VADDR_LOW,
           (unsigned long long)SBPF_STACK_VADDR_BASE);
    printf("Stack pointer (r10): 0x%llx\n\n", 
           (unsigned long long)sbpf_get_stack_pointer());
    
    // Run tests
    test_stack_memory();
    test_stack_boundary();
    test_invalid_access();
    test_simple_program();
    test_heap_region();
    
    // Summary
    printf("\n========================================\n");
    printf("Results: %d/%d tests passed\n", pass_count, test_count);
    printf("========================================\n");
    
    // Cleanup
    sbpf_runtime_destroy(ctx);
    
    return (pass_count == test_count) ? 0 : 1;
}
