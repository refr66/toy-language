//===- SbpfOps.h - SBPF Operations header ---------------------------------===//

#ifndef MLIR_DIALECT_SBPF_SBPFOPS_H
#define MLIR_DIALECT_SBPF_SBPFOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

// Include TableGen-generated op declarations
#define GET_OP_CLASSES
#include "sbpf/SbpfOps.h.inc"

#endif // MLIR_DIALECT_SBPF_SBPFOPS_H
