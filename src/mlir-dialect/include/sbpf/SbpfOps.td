//===- SbpfOps.td - SBPF Operations definition ------------*- tablegen -*-===//
//
// Defines all SBPF operations for MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef SBPF_OPS
#define SBPF_OPS

include "sbpf/SbpfDialect.td"

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

def Sbpf_LoadOp : Sbpf_Op<"load"> {
  let summary = "Load from memory";
  let description = [{
    Load a value from memory at the given address.
    
    Syntax:
    ```
    %result = sbpf.load %addr : i64 -> i8/i16/i32/i64
    ```
    
    The address is computed as base + offset in sBPF semantics.
  }];
  
  let arguments = (ins 
    I64:$address,
    DefaultValuedAttr<I64Attr, "0">:$offset
  );
  let results = (outs AnyInteger:$result);
  
  let assemblyFormat = [{
    $address (`,` $offset^)? `:` type($result) attr-dict
  }];
}

def Sbpf_StoreOp : Sbpf_Op<"store"> {
  let summary = "Store to memory";
  let description = [{
    Store a value to memory at the given address.
    
    Syntax:
    ```
    sbpf.store %addr, %value : i64, i8/i16/i32/i64
    ```
  }];
  
  let arguments = (ins
    I64:$address,
    AnyInteger:$value,
    DefaultValuedAttr<I64Attr, "0">:$offset
  );
  
  let assemblyFormat = [{
    $address `,` $value (`,` $offset^)? `:` type($value) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ALU Operations (64-bit)
//===----------------------------------------------------------------------===//

def Sbpf_Add64Op : Sbpf_PureOp<"add64"> {
  let summary = "64-bit addition";
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Sbpf_Sub64Op : Sbpf_PureOp<"sub64"> {
  let summary = "64-bit subtraction";
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Sbpf_Mul64Op : Sbpf_PureOp<"mul64"> {
  let summary = "64-bit multiplication";
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Sbpf_Div64Op : Sbpf_PureOp<"div64"> {
  let summary = "64-bit unsigned division";
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Sbpf_Mod64Op : Sbpf_PureOp<"mod64"> {
  let summary = "64-bit unsigned modulo";
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Sbpf_Or64Op : Sbpf_PureOp<"or64"> {
  let summary = "64-bit bitwise OR";
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Sbpf_And64Op : Sbpf_PureOp<"and64"> {
  let summary = "64-bit bitwise AND";
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Sbpf_Xor64Op : Sbpf_PureOp<"xor64"> {
  let summary = "64-bit bitwise XOR";
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Sbpf_Lsh64Op : Sbpf_PureOp<"lsh64"> {
  let summary = "64-bit left shift";
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Sbpf_Rsh64Op : Sbpf_PureOp<"rsh64"> {
  let summary = "64-bit logical right shift";
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Sbpf_Arsh64Op : Sbpf_PureOp<"arsh64"> {
  let summary = "64-bit arithmetic right shift";
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Sbpf_Neg64Op : Sbpf_PureOp<"neg64"> {
  let summary = "64-bit negation";
  let arguments = (ins I64:$operand);
  let results = (outs I64:$result);
  let assemblyFormat = "$operand attr-dict";
}

def Sbpf_Mov64Op : Sbpf_PureOp<"mov64"> {
  let summary = "64-bit move";
  let arguments = (ins I64:$src);
  let results = (outs I64:$result);
  let assemblyFormat = "$src attr-dict";
}

//===----------------------------------------------------------------------===//
// ALU Operations (32-bit) - Result zero-extended to 64 bits
//===----------------------------------------------------------------------===//

def Sbpf_Add32Op : Sbpf_PureOp<"add32"> {
  let summary = "32-bit addition (zero-extended)";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I64:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Sbpf_Sub32Op : Sbpf_PureOp<"sub32"> {
  let summary = "32-bit subtraction (zero-extended)";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I64:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

// ... (other 32-bit ops follow same pattern)

//===----------------------------------------------------------------------===//
// Endianness Operations
//===----------------------------------------------------------------------===//

def Sbpf_Le16Op : Sbpf_PureOp<"le16"> {
  let summary = "Convert to little-endian 16-bit";
  let arguments = (ins I64:$src);
  let results = (outs I64:$result);
  let assemblyFormat = "$src attr-dict";
}

def Sbpf_Le32Op : Sbpf_PureOp<"le32"> {
  let summary = "Convert to little-endian 32-bit";
  let arguments = (ins I64:$src);
  let results = (outs I64:$result);
  let assemblyFormat = "$src attr-dict";
}

def Sbpf_Le64Op : Sbpf_PureOp<"le64"> {
  let summary = "Convert to little-endian 64-bit";
  let arguments = (ins I64:$src);
  let results = (outs I64:$result);
  let assemblyFormat = "$src attr-dict";
}

def Sbpf_Be16Op : Sbpf_PureOp<"be16"> {
  let summary = "Convert to big-endian 16-bit";
  let arguments = (ins I64:$src);
  let results = (outs I64:$result);
  let assemblyFormat = "$src attr-dict";
}

def Sbpf_Be32Op : Sbpf_PureOp<"be32"> {
  let summary = "Convert to big-endian 32-bit";
  let arguments = (ins I64:$src);
  let results = (outs I64:$result);
  let assemblyFormat = "$src attr-dict";
}

def Sbpf_Be64Op : Sbpf_PureOp<"be64"> {
  let summary = "Convert to big-endian 64-bit";
  let arguments = (ins I64:$src);
  let results = (outs I64:$result);
  let assemblyFormat = "$src attr-dict";
}

//===----------------------------------------------------------------------===//
// Syscall Operations
//===----------------------------------------------------------------------===//

def Sbpf_SyscallOp : Sbpf_Op<"syscall"> {
  let summary = "Call a Solana syscall (helper function)";
  let description = [{
    Calls a Solana runtime syscall identified by name or hash.
    
    Arguments are passed in r1-r5, result returned in r0.
    
    Syntax:
    ```
    %r0 = sbpf.syscall @sol_log(%r1, %r2, %r3, %r4, %r5) : (i64, i64, i64, i64, i64) -> i64
    ```
  }];
  
  let arguments = (ins
    OptionalAttr<StrAttr>:$callee,
    Variadic<I64>:$args
  );
  let results = (outs I64:$result);
  
  let assemblyFormat = [{
    ($callee^)? `(` $args `)` `:` functional-type($args, $result) attr-dict
  }];
}

def Sbpf_SyscallHashOp : Sbpf_Op<"syscall_hash"> {
  let summary = "Call a syscall by hash (unresolved)";
  let description = [{
    Calls a syscall identified by its Murmur3 hash.
    Used when symbol resolution is not available.
    
    Syntax:
    ```
    %r0 = sbpf.syscall_hash 0x12345678(%r1, %r2, %r3, %r4, %r5)
    ```
  }];
  
  let arguments = (ins
    OptionalAttr<I64Attr>:$hash,
    Variadic<I64>:$args
  );
  let results = (outs I64:$result);
  
  let assemblyFormat = [{
    ($hash^)? `(` $args `)` `:` functional-type($args, $result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Call Operations
//===----------------------------------------------------------------------===//

def Sbpf_CallOp : Sbpf_Op<"call"> {
  let summary = "Call an internal function";
  let description = [{
    Calls another function within the same sBPF program.
    
    Syntax:
    ```
    %r0 = sbpf.call @function_name(%r1, %r2, %r3, %r4, %r5) : (i64, ...) -> i64
    ```
  }];
  
  let arguments = (ins
    OptionalAttr<StrAttr>:$callee,
    Variadic<I64>:$args
  );
  let results = (outs I64:$result);
  
  let assemblyFormat = [{
    ($callee^)? `(` $args `)` `:` functional-type($args, $result) attr-dict
  }];
}

def Sbpf_CallIndirectOp : Sbpf_Op<"call_indirect"> {
  let summary = "Indirect function call via register";
  let description = [{
    Calls a function at the address stored in a register.
    Used for function pointers in sBPF.
    
    Syntax:
    ```
    %r0 = sbpf.call_indirect %target(%r1, %r2, %r3, %r4, %r5)
    ```
  }];
  
  let arguments = (ins
    I64:$target,
    Variadic<I64>:$args
  );
  let results = (outs I64:$result);
  
  let assemblyFormat = [{
    $target `(` $args `)` `:` functional-type($args, $result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Exit Operation
//===----------------------------------------------------------------------===//

def Sbpf_ExitOp : Sbpf_Op<"exit", [Terminator]> {
  let summary = "Exit from function, return r0";
  let description = [{
    Returns from the current function with the value in r0.
    This is the terminator for sBPF functions.
    
    Syntax:
    ```
    sbpf.exit %r0 : i64
    ```
  }];
  
  let arguments = (ins I64:$return_value);
  
  let assemblyFormat = "$return_value `:` type($return_value) attr-dict";
}

#endif // SBPF_OPS
