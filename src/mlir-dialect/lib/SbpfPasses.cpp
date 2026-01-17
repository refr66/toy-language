//===- SbpfPasses.cpp - SBPF Passes implementation ------------------------===//
//
// Implements transformation passes for the SBPF dialect.
//
//===----------------------------------------------------------------------===//

#include "sbpf/SbpfPasses.h"
#include "sbpf/SbpfDialect.h"
#include "sbpf/SbpfOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::sbpf;

//===----------------------------------------------------------------------===//
// Pass Definitions (generated from TableGen)
//===----------------------------------------------------------------------===//

namespace mlir {
namespace sbpf {
// First declare the Options structs
#define GEN_PASS_DECL_LOWERSBPFMEM
#define GEN_PASS_DECL_LOWERSBPFTOSTANDARD
#include "sbpf/SbpfPasses.h.inc"

// Then define the pass base classes
#define GEN_PASS_DEF_LOWERSBPFMEM
#define GEN_PASS_DEF_LOWERSBPFTOSTANDARD
#include "sbpf/SbpfPasses.h.inc"
} // namespace sbpf
} // namespace mlir

//===----------------------------------------------------------------------===//
// Helper: sBPF Memory Region Constants
//===----------------------------------------------------------------------===//

namespace {
// sBPF virtual memory layout
constexpr uint64_t SBPF_STACK_BASE = 0x200000000ULL;  // Stack base address
constexpr uint64_t SBPF_STACK_SIZE = 4096;            // 4KB stack

/// Check if an address is statically known to be in the stack region
bool isStaticStackAddress(Value addr, int64_t offset) {
  // Try to get constant address
  if (auto constOp = addr.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      uint64_t baseAddr = intAttr.getInt();
      uint64_t effectiveAddr = baseAddr + offset;
      return (effectiveAddr >= SBPF_STACK_BASE - SBPF_STACK_SIZE && 
              effectiveAddr < SBPF_STACK_BASE);
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// LoadOp Lowering Pattern
//===----------------------------------------------------------------------===//

struct LoadOpLowering : public OpRewritePattern<LoadOp> {
  using OpRewritePattern<LoadOp>::OpRewritePattern;
  
  uint64_t stackBase;
  uint64_t stackSize;
  bool emitBoundsCheck;
  
  LoadOpLowering(MLIRContext *ctx, uint64_t stackBase, uint64_t stackSize, 
                 bool emitBoundsCheck)
      : OpRewritePattern<LoadOp>(ctx), stackBase(stackBase), 
        stackSize(stackSize), emitBoundsCheck(emitBoundsCheck) {}
  
  LogicalResult matchAndRewrite(LoadOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value addr = op.getAddress();
    int64_t offset = op.getOffset();
    Type resultType = op.getResult().getType();
    
    // Get bit width for the load
    unsigned bitWidth = resultType.getIntOrFloatBitWidth();
    
    // Calculate effective address: addr + offset
    Value effectiveAddr = addr;
    if (offset != 0) {
      Value offsetVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(offset));
      effectiveAddr = rewriter.create<arith::AddIOp>(loc, addr, offsetVal);
    }
    
    // Strategy selection based on whether address is constant
    if (isStaticStackAddress(addr, offset)) {
      // ===== Static Path (Scheme B) =====
      // Address is known at compile time, we can do static verification
      // and emit direct memory access without runtime checks
      
      // Translate sBPF virtual address to offset from runtime stack pointer
      // Real address = runtime_stack_ptr + (sbpf_addr - SBPF_STACK_BASE)
      Value stackBaseVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(stackBase));
      Value addrOffset = rewriter.create<arith::SubIOp>(
          loc, effectiveAddr, stackBaseVal);
      
      // Create memref type and load
      // Note: In a full implementation, we'd use a runtime-provided stack pointer
      // For now, we emit a call to a runtime function
      auto funcOp = rewriter.create<func::CallOp>(
          loc, "__sbpf_load_" + std::to_string(bitWidth),
          TypeRange{resultType}, ValueRange{effectiveAddr});
      
      rewriter.replaceOp(op, funcOp.getResults());
      
    } else {
      // ===== Dynamic Path (Scheme C) =====
      // Address is computed at runtime - delegate to runtime function
      // Runtime function handles bounds checking and address translation
      auto funcOp = rewriter.create<func::CallOp>(
          loc, "__sbpf_load_" + std::to_string(bitWidth),
          TypeRange{resultType}, ValueRange{effectiveAddr});
      rewriter.replaceOp(op, funcOp.getResults());
    }
    
    return success();
  }
};

//===----------------------------------------------------------------------===//
// StoreOp Lowering Pattern
//===----------------------------------------------------------------------===//

struct StoreOpLowering : public OpRewritePattern<StoreOp> {
  using OpRewritePattern<StoreOp>::OpRewritePattern;
  
  uint64_t stackBase;
  uint64_t stackSize;
  bool emitBoundsCheck;
  
  StoreOpLowering(MLIRContext *ctx, uint64_t stackBase, uint64_t stackSize,
                  bool emitBoundsCheck)
      : OpRewritePattern<StoreOp>(ctx), stackBase(stackBase),
        stackSize(stackSize), emitBoundsCheck(emitBoundsCheck) {}
  
  LogicalResult matchAndRewrite(StoreOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value addr = op.getAddress();
    Value value = op.getValue();
    int64_t offset = op.getOffset();
    Type valueType = value.getType();
    
    unsigned bitWidth = valueType.getIntOrFloatBitWidth();
    
    // Calculate effective address
    Value effectiveAddr = addr;
    if (offset != 0) {
      Value offsetVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(offset));
      effectiveAddr = rewriter.create<arith::AddIOp>(loc, addr, offsetVal);
    }
    
    if (isStaticStackAddress(addr, offset)) {
      // ===== Static Path =====
      rewriter.create<func::CallOp>(
          loc, "__sbpf_store_" + std::to_string(bitWidth),
          TypeRange{}, ValueRange{effectiveAddr, value});
      rewriter.eraseOp(op);
      
    } else {
      // ===== Dynamic Path =====
      // Delegate bounds checking to runtime function
      rewriter.create<func::CallOp>(
          loc, "__sbpf_store_" + std::to_string(bitWidth),
          TypeRange{}, ValueRange{effectiveAddr, value});
      rewriter.eraseOp(op);
    }
    
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LowerSbpfMem Pass Implementation
//===----------------------------------------------------------------------===//

struct LowerSbpfMemPass : public sbpf::impl::LowerSbpfMemBase<LowerSbpfMemPass> {
  
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    
    // Declare runtime helper functions if not already present
    declareRuntimeFunctions(module);
    
    // Set up patterns
    RewritePatternSet patterns(ctx);
    patterns.add<LoadOpLowering>(ctx, stackBase, stackSize, emitBoundsCheck);
    patterns.add<StoreOpLowering>(ctx, stackBase, stackSize, emitBoundsCheck);
    
    // Apply patterns using greedy rewrite
    (void)applyPatternsGreedily(module, std::move(patterns));
  }
  
private:
  /// Declare external runtime helper functions
  void declareRuntimeFunctions(ModuleOp module) {
    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToStart(module.getBody());
    Location loc = module.getLoc();
    
    auto i64Type = builder.getI64Type();
    auto i32Type = builder.getI32Type();
    auto i16Type = builder.getI16Type();
    auto i8Type = builder.getI8Type();
    
    // Load functions: i64 __sbpf_load_N(i64 addr)
    auto declareLoadFunc = [&](StringRef name, Type resultType) {
      if (!module.lookupSymbol(name)) {
        auto funcType = builder.getFunctionType({i64Type}, {resultType});
        auto funcOp = func::FuncOp::create(loc, name, funcType);
        funcOp.setPrivate();
        builder.insert(funcOp);
      }
    };
    
    declareLoadFunc("__sbpf_load_8", i8Type);
    declareLoadFunc("__sbpf_load_16", i16Type);
    declareLoadFunc("__sbpf_load_32", i32Type);
    declareLoadFunc("__sbpf_load_64", i64Type);
    
    // Store functions: void __sbpf_store_N(i64 addr, iN value)
    auto declareStoreFunc = [&](StringRef name, Type valueType) {
      if (!module.lookupSymbol(name)) {
        auto funcType = builder.getFunctionType({i64Type, valueType}, {});
        auto funcOp = func::FuncOp::create(loc, name, funcType);
        funcOp.setPrivate();
        builder.insert(funcOp);
      }
    };
    
    declareStoreFunc("__sbpf_store_8", i8Type);
    declareStoreFunc("__sbpf_store_16", i16Type);
    declareStoreFunc("__sbpf_store_32", i32Type);
    declareStoreFunc("__sbpf_store_64", i64Type);
    
    // Abort function: void __sbpf_abort()
    if (!module.lookupSymbol("__sbpf_abort")) {
      auto funcType = builder.getFunctionType({}, {});
      auto abortFunc = func::FuncOp::create(loc, "__sbpf_abort", funcType);
      abortFunc.setPrivate();
      builder.insert(abortFunc);
    }
  }
};

//===----------------------------------------------------------------------===//
// LowerSbpfToStandard Pass Implementation (placeholder)
//===----------------------------------------------------------------------===//

struct LowerSbpfToStandardPass 
    : public sbpf::impl::LowerSbpfToStandardBase<LowerSbpfToStandardPass> {
  
  void runOnOperation() override {
    // TODO: Implement lowering of ALU ops, syscalls, etc.
    // For now, this is a placeholder
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Creation Functions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace sbpf {

std::unique_ptr<Pass> createLowerSbpfMemPass() {
  return std::make_unique<LowerSbpfMemPass>();
}

std::unique_ptr<Pass> createLowerSbpfToStandardPass() {
  return std::make_unique<LowerSbpfToStandardPass>();
}

} // namespace sbpf
} // namespace mlir
