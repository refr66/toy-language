//===- SbpfPasses.h - SBPF Passes declaration ------------------*- C++ -*-===//
//
// Declares transformation passes for the SBPF dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SBPF_PASSES_H
#define SBPF_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace sbpf {

//===----------------------------------------------------------------------===//
// Pass Creation Functions
//===----------------------------------------------------------------------===//

/// Creates a pass that lowers sbpf.load/store to checked memory operations.
/// This handles address validation and virtual-to-physical address mapping.
std::unique_ptr<Pass> createLowerSbpfMemPass();

/// Creates a pass that lowers SBPF dialect ops to standard MLIR dialects.
std::unique_ptr<Pass> createLowerSbpfToStandardPass();

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "sbpf/SbpfPasses.h.inc"

} // namespace sbpf
} // namespace mlir

#endif // SBPF_PASSES_H
