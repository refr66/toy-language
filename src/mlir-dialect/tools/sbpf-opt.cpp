//===- sbpf-opt.cpp - SBPF optimizer driver -------------------------------===//
//
// Main entry point for sbpf-opt, which runs MLIR passes on SBPF dialect.
//
// Usage:
//   sbpf-opt input.mlir                       # Parse and print
//   sbpf-opt input.mlir --mem2reg             # Run mem2reg optimization
//   sbpf-opt input.mlir --lower-sbpf-mem      # Lower sbpf.load/store
//   sbpf-opt input.mlir --canonicalize        # Run canonicalization
//   sbpf-opt input.mlir --cse                 # Common subexpression elimination
//   sbpf-opt input.mlir --convert-func-to-llvm    # Convert func to LLVM
//   sbpf-opt input.mlir --convert-arith-to-llvm   # Convert arith to LLVM
//   sbpf-opt input.mlir --convert-cf-to-llvm      # Convert cf to LLVM
//
//===----------------------------------------------------------------------===//

#include "sbpf/SbpfDialect.h"
#include "sbpf/SbpfOps.h"
#include "sbpf/SbpfPasses.h"

// Standard Dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"  // For ub.poison from mem2reg

// LLVM Dialect
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

// Conversion Passes and Interfaces (LLVM 22 style)
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"

// Core MLIR
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"

int main(int argc, char **argv) {
  // Register all MLIR transformation passes (includes mem2reg, cse, canonicalize, etc.)
  mlir::registerTransformsPasses();
  mlir::func::registerFuncPasses();
  mlir::memref::registerMemRefPasses();
  mlir::arith::registerArithPasses();
  
  // Register conversion pass
  mlir::registerConvertToLLVMPass();
  
  // Register SBPF-specific passes
  mlir::sbpf::registerSbpfPasses();
  
  // Register standard dialects we use
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::ub::UBDialect>();  // Required for mem2reg (ub.poison)
  
  // Register LLVM dialect (target for lowering)
  registry.insert<mlir::LLVM::LLVMDialect>();
  
  // Register conversion interfaces for --convert-to-llvm pass
  mlir::registerConvertToLLVMDependentDialectLoading(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  
  // Register our SBPF dialect
  registry.insert<mlir::sbpf::SbpfDialect>();
  
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SBPF optimizer driver\n", registry));
}
