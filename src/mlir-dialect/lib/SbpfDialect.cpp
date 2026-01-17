//===- SbpfDialect.cpp - SBPF Dialect implementation ----------------------===//

#include "sbpf/SbpfDialect.h"
#include "sbpf/SbpfOps.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::sbpf;

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

#include "sbpf/SbpfDialect.cpp.inc"

void SbpfDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "sbpf/SbpfOps.cpp.inc"
  >();
}

//===----------------------------------------------------------------------===//
// Attribute/Type parsing and printing
//===----------------------------------------------------------------------===//

::mlir::Attribute SbpfDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                               ::mlir::Type type) const {
  // No custom attributes in sbpf dialect
  parser.emitError(parser.getCurrentLocation(), "unknown sbpf attribute");
  return {};
}

void SbpfDialect::printAttribute(::mlir::Attribute attr,
                                  ::mlir::DialectAsmPrinter &printer) const {
  // No custom attributes in sbpf dialect
}

::mlir::Type SbpfDialect::parseType(::mlir::DialectAsmParser &parser) const {
  // No custom types in sbpf dialect
  parser.emitError(parser.getCurrentLocation(), "unknown sbpf type");
  return {};
}

void SbpfDialect::printType(::mlir::Type type,
                             ::mlir::DialectAsmPrinter &printer) const {
  // No custom types in sbpf dialect
}
