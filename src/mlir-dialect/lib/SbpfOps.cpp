//===- SbpfOps.cpp - SBPF Operations implementation -----------------------===//

#include "sbpf/SbpfOps.h"
#include "sbpf/SbpfDialect.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::sbpf;

//===----------------------------------------------------------------------===//
// TableGen'd op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "sbpf/SbpfOps.cpp.inc"
