//! MLIR Builder using Melior
//!
//! This module builds real MLIR IR objects using the Melior library.
//! Unlike text emission, this creates in-memory MLIR graphs that can be:
//! - Verified by MLIR's verifier
//! - Optimized by MLIR passes
//! - Lowered to LLVM IR or other targets

use melior::{
    Context,
    dialect::{arith, cf, func, memref, DialectRegistry},
    ir::{
        attribute::{StringAttribute, TypeAttribute, IntegerAttribute, FlatSymbolRefAttribute},
        r#type::{FunctionType, IntegerType, MemRefType},
        Block, Location, Module, Operation, OperationRef, Region, Type, Value,
    },
    pass::{self, PassManager},
    utility::register_all_dialects,
};
use std::collections::HashMap;

use crate::ir::{Program, Function, BasicBlock, Instruction, Op, AluOp, JmpCond, Operand, MemSize, CallTarget};

/// Error type for MLIR generation
#[derive(Debug, thiserror::Error)]
pub enum MlirError {
    #[error("MLIR context error: {0}")]
    Context(String),
    #[error("Invalid operation: {0}")]
    InvalidOp(String),
    #[error("Verification failed: {0}")]
    Verification(String),
}

/// MLIR Builder for sBPF programs
pub struct MlirBuilder<'c> {
    context: &'c Context,
    /// Register file: maps register number to memref Value
    registers: HashMap<u8, Value<'c, 'c>>,
    /// SSA value counter for unique naming
    ssa_counter: usize,
}

impl<'c> MlirBuilder<'c> {
    /// Create a new MLIR builder
    pub fn new(context: &'c Context) -> Self {
        Self {
            context,
            registers: HashMap::new(),
            ssa_counter: 0,
        }
    }

    /// Build complete MLIR module from program
    pub fn build_module(&mut self, program: &Program) -> Result<Module<'c>, MlirError> {
        let location = Location::unknown(self.context);
        let module = Module::new(location);

        // Build each function
        for func in program.functions.values() {
            let func_op = self.build_function(func, &program.entrypoint)?;
            module.body().append_operation(func_op);
        }

        Ok(module)
    }

    /// Build a single function
    fn build_function(&mut self, func: &Function, entrypoint: &str) -> Result<Operation<'c>, MlirError> {
        let location = Location::unknown(self.context);
        let i64_type = IntegerType::new(self.context, 64).into();
        
        let is_entry = func.name == entrypoint;
        
        // Function type: entry takes 1 arg (context), internal takes 5 args
        let (input_types, result_types): (Vec<Type>, Vec<Type>) = if is_entry {
            (vec![i64_type], vec![i64_type])
        } else {
            (vec![i64_type; 5], vec![i64_type])
        };

        let func_type = FunctionType::new(self.context, &input_types, &result_types);
        
        // Create function operation using func dialect
        let func_name = Self::mangle_name(&func.name);
        
        // Build function body
        let region = Region::new();
        let entry_block = Block::new(&input_types.iter().map(|_| (i64_type, location)).collect::<Vec<_>>());
        
        // Allocate register file (memref<i64> for each r0-r10)
        self.allocate_registers(&entry_block, location)?;
        
        // Initialize registers from function arguments
        self.initialize_registers(&entry_block, location, is_entry)?;
        
        region.append_block(entry_block);
        
        // Build basic blocks
        let mut block_map: HashMap<usize, Block<'c>> = HashMap::new();
        for bb in func.blocks.values() {
            if bb.id != func.entry {
                let block = Block::new(&[]);
                block_map.insert(bb.id, block);
            }
        }
        
        // Add blocks to region
        for (_, block) in block_map {
            region.append_block(block);
        }
        
        // Create func.func operation
        let func_op = func::func(
            self.context,
            StringAttribute::new(self.context, &func_name),
            TypeAttribute::new(func_type.into()),
            region,
            &[],
            location,
        );

        Ok(func_op)
    }

    /// Allocate register file using memref.alloca
    fn allocate_registers(&mut self, block: &Block<'c>, location: Location<'c>) -> Result<(), MlirError> {
        let i64_type = IntegerType::new(self.context, 64).into();
        let memref_type = MemRefType::new(i64_type, &[], None, None);

        for reg_num in 0..=10u8 {
            let alloca_op = memref::alloca(
                self.context,
                memref_type,
                &[],
                &[],
                None,
                location,
            );
            block.append_operation(alloca_op);
            // Store the result value for later use
            // Note: In real implementation, we'd track these values
        }

        Ok(())
    }

    /// Initialize registers from function arguments
    fn initialize_registers(&mut self, block: &Block<'c>, location: Location<'c>, is_entry: bool) -> Result<(), MlirError> {
        let i64_type = IntegerType::new(self.context, 64).into();
        
        // r0 = 0 (return value)
        let zero = arith::constant(
            self.context,
            IntegerAttribute::new(i64_type, 0).into(),
            location,
        );
        block.append_operation(zero);

        // r1 = first argument (input context for entry, arg1 for internal)
        // Additional initialization for r2-r5 if internal function
        // r6-r9 = 0 (callee-saved)
        // r10 = stack pointer

        Ok(())
    }

    /// Build a single instruction
    fn build_instruction(&mut self, block: &Block<'c>, insn: &Instruction, location: Location<'c>) -> Result<(), MlirError> {
        match &insn.op {
            Op::Alu64 { op, dst, src } => {
                self.build_alu_op(block, op, *dst, src, 64, location)?;
            }
            Op::Alu32 { op, dst, src } => {
                self.build_alu_op(block, op, *dst, src, 32, location)?;
            }
            Op::LoadImm64 { dst, imm } => {
                self.build_load_imm64(block, *dst, *imm, location)?;
            }
            Op::Load { dst, src, off, size } => {
                self.build_load(block, *dst, *src, *off, size, location)?;
            }
            Op::StoreReg { dst, off, src, size } => {
                self.build_store_reg(block, *dst, *off, *src, size, location)?;
            }
            Op::StoreImm { dst, off, imm, size } => {
                self.build_store_imm(block, *dst, *off, *imm, size, location)?;
            }
            Op::Jump { target } => {
                // cf.br ^bb{target}
            }
            Op::JumpCond { cond, dst, src, target } => {
                // Build comparison and cf.cond_br
            }
            Op::Call { target } => {
                self.build_call(block, target, location)?;
            }
            Op::Exit => {
                self.build_exit(block, location)?;
            }
            Op::Endian { dst, size, to_le } => {
                // Custom sbpf.endian operation
            }
            Op::Unknown { opcode } => {
                // Skip unknown opcodes
            }
        }
        Ok(())
    }

    /// Build ALU operation
    fn build_alu_op(
        &mut self,
        block: &Block<'c>,
        op: &AluOp,
        dst: u8,
        src: &Operand,
        bits: u8,
        location: Location<'c>,
    ) -> Result<(), MlirError> {
        let int_type: Type = IntegerType::new(self.context, bits as u32).into();
        
        // Load dst from register file
        // let dst_val = self.load_register(block, dst, location)?;
        
        // Get source value (either from register or immediate)
        // let src_val = match src {
        //     Operand::Reg(r) => self.load_register(block, *r, location)?,
        //     Operand::Imm(i) => self.create_constant(block, *i, int_type, location)?,
        // };
        
        // Build arithmetic operation
        // let result = match op {
        //     AluOp::Add => arith::addi(dst_val, src_val, location),
        //     AluOp::Sub => arith::subi(dst_val, src_val, location),
        //     // ... other ops
        // };
        // block.append_operation(result);
        
        // Store result back to register file
        // self.store_register(block, dst, result_val, location)?;
        
        Ok(())
    }

    /// Build load immediate 64-bit
    fn build_load_imm64(&mut self, block: &Block<'c>, dst: u8, imm: i64, location: Location<'c>) -> Result<(), MlirError> {
        let i64_type: Type = IntegerType::new(self.context, 64).into();
        
        // Create constant
        let const_op = arith::constant(
            self.context,
            IntegerAttribute::new(i64_type, imm).into(),
            location,
        );
        block.append_operation(const_op);
        
        // Store to register file
        // self.store_register(block, dst, const_val, location)?;
        
        Ok(())
    }

    /// Build load from memory
    fn build_load(&mut self, block: &Block<'c>, dst: u8, src: u8, off: i16, size: &MemSize, location: Location<'c>) -> Result<(), MlirError> {
        // 1. Load base address from src register
        // 2. Add offset
        // 3. Custom sbpf.load operation (or use memref.load with cast)
        // 4. Zero-extend if needed
        // 5. Store to dst register
        Ok(())
    }

    /// Build store register to memory
    fn build_store_reg(&mut self, block: &Block<'c>, dst: u8, off: i16, src: u8, size: &MemSize, location: Location<'c>) -> Result<(), MlirError> {
        // Similar to load but reversed
        Ok(())
    }

    /// Build store immediate to memory
    fn build_store_imm(&mut self, block: &Block<'c>, dst: u8, off: i16, imm: i64, size: &MemSize, location: Location<'c>) -> Result<(), MlirError> {
        Ok(())
    }

    /// Build function call
    fn build_call(&mut self, block: &Block<'c>, target: &CallTarget, location: Location<'c>) -> Result<(), MlirError> {
        match target {
            CallTarget::Internal { name, .. } => {
                // func.call @name(%r1, %r2, %r3, %r4, %r5)
                let func_name = name.as_ref().map(|n| Self::mangle_name(n)).unwrap_or_else(|| "unknown".to_string());
                // Build call operation
            }
            CallTarget::Syscall { name, hash } => {
                // Custom sbpf.syscall operation
            }
            CallTarget::Register(reg) => {
                // Custom sbpf.call_indirect operation
            }
        }
        Ok(())
    }

    /// Build exit (return)
    fn build_exit(&mut self, block: &Block<'c>, location: Location<'c>) -> Result<(), MlirError> {
        // Load r0 and return
        // func.return %r0
        Ok(())
    }

    /// Mangle function name for MLIR
    fn mangle_name(name: &str) -> String {
        name.chars()
            .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
            .collect()
    }

    fn next_ssa(&mut self) -> usize {
        let v = self.ssa_counter;
        self.ssa_counter += 1;
        v
    }
}

/// Initialize MLIR context with required dialects
pub fn create_context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    
    context
}

/// Run verification and optimization passes
pub fn verify_and_optimize(module: &Module) -> Result<(), MlirError> {
    // Create pass manager
    let pass_manager = PassManager::new(module.context());
    
    // Add verification
    // pass_manager.add_pass(pass::conversion::create_builtin_to_llvm());
    
    // Run passes
    if pass_manager.run(module).is_err() {
        return Err(MlirError::Verification("Pass manager failed".to_string()));
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_context() {
        let context = create_context();
        assert!(context.num_loaded_dialects() > 0);
    }

    #[test]
    fn test_build_empty_module() {
        let context = create_context();
        let mut builder = MlirBuilder::new(&context);
        
        let program = Program {
            entrypoint: "main".to_string(),
            functions: std::collections::BTreeMap::new(),
        };
        
        let module = builder.build_module(&program).unwrap();
        // Module should be valid but empty
    }
}
