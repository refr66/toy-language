//===- SbpfPasses.td - SBPF Passes definition --------------*- tablegen -*-===//
//
// Defines transformation passes for the SBPF dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SBPF_PASSES
#define SBPF_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// LowerSbpfMem Pass
//===----------------------------------------------------------------------===//

def LowerSbpfMem : Pass<"lower-sbpf-mem", "mlir::ModuleOp"> {
  let summary = "Lower sbpf.load/store to checked memory operations";
  let description = [{
    This pass lowers `sbpf.load` and `sbpf.store` operations to a combination
    of address validation and actual memory access.
    
    sBPF memory operations cannot be directly translated to RISC-V ld/sd because:
    1. Security: Must validate that addresses are within legal bounds
    2. Mapping: sBPF virtual addresses (e.g., 0x200000000 for stack) must be
       mapped to actual runtime addresses
    
    The pass implements two strategies:
    - **Static path (constant address)**: If the address offset is known at 
      compile time, we can statically verify bounds and emit direct access.
    - **Dynamic path (runtime address)**: Insert runtime bounds checking and
      address translation before the actual memory access.
    
    Memory regions in sBPF:
    - Stack:  0x200000000 - 0x200001000 (4KB, grows down from r10)
    - Heap:   Runtime allocated, passed via r1
    - Input:  Read-only data region
    
    Example transformation:
    ```mlir
    // Before:
    %val = sbpf.load %addr : i64
    
    // After (dynamic path):
    %region = sbpf.mem_region %addr               // Determine memory region
    %real_addr = sbpf.translate_addr %addr, %region  // Map to real address
    %val = memref.load %real_addr[0] : memref<i64>
    ```
  }];
  
  let constructor = "mlir::sbpf::createLowerSbpfMemPass()";
  
  let options = [
    Option<"stackBase", "stack-base", "uint64_t", "0x200000000",
           "sBPF stack base virtual address">,
    Option<"stackSize", "stack-size", "uint64_t", "4096",
           "sBPF stack size in bytes">,
    Option<"emitBoundsCheck", "emit-bounds-check", "bool", "true",
           "Emit runtime bounds checking code">
  ];
  
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::cf::ControlFlowDialect"
  ];
}

//===----------------------------------------------------------------------===//
// LowerSbpfToStandard Pass  
//===----------------------------------------------------------------------===//

def LowerSbpfToStandard : Pass<"lower-sbpf-to-standard", "mlir::ModuleOp"> {
  let summary = "Lower SBPF dialect operations to standard MLIR dialects";
  let description = [{
    Lowers SBPF-specific operations to standard MLIR dialects (arith, cf, func).
    
    This includes:
    - sbpf.add64/sub64/... -> arith.addi/subi/...
    - sbpf.mov64 -> (eliminated, value forwarding)
    - sbpf.exit -> func.return
    - sbpf.syscall -> func.call @__sbpf_syscall_*
  }];
  
  let constructor = "mlir::sbpf::createLowerSbpfToStandardPass()";
  
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::cf::ControlFlowDialect"
  ];
}

#endif // SBPF_PASSES
