//===- SbpfDialect.td - SBPF Dialect definition -----------*- tablegen -*-===//
//
// Defines the SBPF (Solana BPF) dialect for MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef SBPF_DIALECT
#define SBPF_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// SBPF Dialect Definition
//===----------------------------------------------------------------------===//

def Sbpf_Dialect : Dialect {
  let name = "sbpf";
  let summary = "Solana BPF (sBPF) dialect for MLIR";
  let description = [{
    The SBPF dialect represents Solana's variant of eBPF instructions.
    It is designed for:
    - Lifting sBPF bytecode to MLIR
    - Optimizing sBPF programs
    - Lowering to RISC-V or other targets
    
    Key features:
    - 11 64-bit registers (r0-r10)
    - Stack-based memory model
    - Syscall interface for Solana runtime
  }];
  
  let cppNamespace = "::mlir::sbpf";
  
  // Use default attribute/type storage
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  
  // Disable properties storage for compatibility
  let usePropertiesForAttributes = 0;
}

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

// Base class for all SBPF operations
class Sbpf_Op<string mnemonic, list<Trait> traits = []> :
    Op<Sbpf_Dialect, mnemonic, traits>;

// Pure operations (no side effects) - use NoMemoryEffect trait
class Sbpf_PureOp<string mnemonic, list<Trait> traits = []> :
    Sbpf_Op<mnemonic, !listconcat(traits, [NoMemoryEffect])>;

#endif // SBPF_DIALECT
