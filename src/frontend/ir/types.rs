//! Intermediate Representation for sBPF programs
//!
//! This module defines a structured IR that can be used for:
//! - Pretty printing
//! - MLIR translation
//! - Static analysis

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;

use crate::ebpf::RawInsn;

/// Format memory operand with proper offset display
fn format_mem_operand(base: Reg, off: i16) -> String {
    if off == 0 {
        format!("[{}]", base)
    } else if off > 0 {
        format!("[{}+{}]", base, off)
    } else {
        format!("[{}{}]", base, off) // off is negative, will display as [r10-256]
    }
}

// ============================================================================
// High-level instruction representation
// ============================================================================

/// Register identifier (r0-r10)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(pub u8);

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0)
    }
}

/// Operand: either a register or an immediate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    Reg(Reg),
    Imm(i64),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Reg(r) => write!(f, "{}", r),
            Operand::Imm(i) => write!(f, "{}", i),
        }
    }
}

/// Memory size for load/store operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemSize {
    Byte,       // 1 byte
    Half,       // 2 bytes
    Word,       // 4 bytes
    DoubleWord, // 8 bytes
}

impl fmt::Display for MemSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemSize::Byte => write!(f, "b"),
            MemSize::Half => write!(f, "h"),
            MemSize::Word => write!(f, "w"),
            MemSize::DoubleWord => write!(f, "dw"),
        }
    }
}

/// ALU operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AluOp {
    Add,
    Sub,
    Mul,
    Div,
    Or,
    And,
    Lsh,
    Rsh,
    Neg,
    Mod,
    Xor,
    Mov,
    Arsh,
}

impl fmt::Display for AluOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AluOp::Add => write!(f, "add"),
            AluOp::Sub => write!(f, "sub"),
            AluOp::Mul => write!(f, "mul"),
            AluOp::Div => write!(f, "div"),
            AluOp::Or => write!(f, "or"),
            AluOp::And => write!(f, "and"),
            AluOp::Lsh => write!(f, "lsh"),
            AluOp::Rsh => write!(f, "rsh"),
            AluOp::Neg => write!(f, "neg"),
            AluOp::Mod => write!(f, "mod"),
            AluOp::Xor => write!(f, "xor"),
            AluOp::Mov => write!(f, "mov"),
            AluOp::Arsh => write!(f, "arsh"),
        }
    }
}

/// Jump condition type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JmpCond {
    Eq,    // ==
    Gt,    // > (unsigned)
    Ge,    // >= (unsigned)
    Lt,    // < (unsigned)
    Le,    // <= (unsigned)
    Set,   // & (bitwise test)
    Ne,    // !=
    Sgt,   // > (signed)
    Sge,   // >= (signed)
    Slt,   // < (signed)
    Sle,   // <= (signed)
}

impl fmt::Display for JmpCond {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JmpCond::Eq => write!(f, "eq"),
            JmpCond::Gt => write!(f, "gt"),
            JmpCond::Ge => write!(f, "ge"),
            JmpCond::Lt => write!(f, "lt"),
            JmpCond::Le => write!(f, "le"),
            JmpCond::Set => write!(f, "set"),
            JmpCond::Ne => write!(f, "ne"),
            JmpCond::Sgt => write!(f, "sgt"),
            JmpCond::Sge => write!(f, "sge"),
            JmpCond::Slt => write!(f, "slt"),
            JmpCond::Sle => write!(f, "sle"),
        }
    }
}

/// Call target type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallTarget {
    /// Internal function call (relative call to another function in the program)
    Internal { pc: usize, name: Option<String> },
    /// Syscall (external helper function)
    Syscall { hash: u32, name: Option<String> },
    /// Register-indirect call
    Register(Reg),
}

impl fmt::Display for CallTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CallTarget::Internal { name: Some(name), .. } => write!(f, "{}", name),
            CallTarget::Internal { pc, .. } => write!(f, "function_{}", pc),
            CallTarget::Syscall { name: Some(name), .. } => write!(f, "{}", name),
            CallTarget::Syscall { hash, .. } => write!(f, "syscall_0x{:08x}", hash),
            CallTarget::Register(r) => write!(f, "{}", r),
        }
    }
}

/// High-level sBPF instruction
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SbpfInsn {
    /// Load immediate 64-bit value: dst = imm64
    LoadImm64 { dst: Reg, imm: i64 },

    /// Load from memory: dst = *(size*)(src + off)
    Load { dst: Reg, src: Reg, off: i16, size: MemSize },

    /// Store immediate to memory: *(size*)(dst + off) = imm
    StoreImm { dst: Reg, off: i16, imm: i64, size: MemSize },

    /// Store register to memory: *(size*)(dst + off) = src
    StoreReg { dst: Reg, off: i16, src: Reg, size: MemSize },

    /// 32-bit ALU operation: dst = dst op src/imm (result zero-extended)
    Alu32 { op: AluOp, dst: Reg, src: Operand },

    /// 64-bit ALU operation: dst = dst op src/imm
    Alu64 { op: AluOp, dst: Reg, src: Operand },

    /// Endianness conversion
    Endian { dst: Reg, size: u8, to_le: bool },

    /// Unconditional jump: goto target
    Jump { target: usize },

    /// Conditional jump: if (dst cond src/imm) goto target
    JumpCond { cond: JmpCond, dst: Reg, src: Operand, target: usize },

    /// Function call
    Call { target: CallTarget },

    /// Return from function
    Exit,

    /// Unknown/invalid instruction
    Unknown { raw: RawInsn },
}

impl SbpfInsn {
    /// Get the static CU (Compute Unit) cost of this instruction
    /// This is a simplified model; actual costs may vary
    pub fn cu_cost(&self) -> u64 {
        match self {
            // Memory operations are typically more expensive
            SbpfInsn::Load { .. } => 1,
            SbpfInsn::StoreImm { .. } => 1,
            SbpfInsn::StoreReg { .. } => 1,
            SbpfInsn::LoadImm64 { .. } => 1,
            
            // ALU operations
            SbpfInsn::Alu32 { op, .. } | SbpfInsn::Alu64 { op, .. } => {
                match op {
                    AluOp::Div | AluOp::Mod => 1, // Division is expensive
                    _ => 1,
                }
            }
            
            // Control flow
            SbpfInsn::Jump { .. } => 1,
            SbpfInsn::JumpCond { .. } => 1,
            SbpfInsn::Call { .. } => 1,
            SbpfInsn::Exit => 1,
            
            // Others
            SbpfInsn::Endian { .. } => 1,
            SbpfInsn::Unknown { .. } => 1,
        }
    }

    /// Get the PC (instruction pointer) from the original raw instruction if available
    pub fn is_terminator(&self) -> bool {
        matches!(self, 
            SbpfInsn::Jump { .. } | 
            SbpfInsn::JumpCond { .. } | 
            SbpfInsn::Call { .. } | 
            SbpfInsn::Exit
        )
    }

    /// Check if this instruction is a control flow instruction
    pub fn is_control_flow(&self) -> bool {
        matches!(self,
            SbpfInsn::Jump { .. } |
            SbpfInsn::JumpCond { .. } |
            SbpfInsn::Call { .. } |
            SbpfInsn::Exit
        )
    }

    /// Get jump targets (if any)
    pub fn jump_targets(&self) -> Vec<usize> {
        match self {
            SbpfInsn::Jump { target } => vec![*target],
            SbpfInsn::JumpCond { target, .. } => vec![*target],
            SbpfInsn::Call { target: CallTarget::Internal { pc, .. } } => vec![*pc],
            _ => vec![],
        }
    }
}

impl fmt::Display for SbpfInsn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SbpfInsn::LoadImm64 { dst, imm } => {
                write!(f, "lddw {}, 0x{:x}", dst, imm)
            }
            SbpfInsn::Load { dst, src, off, size } => {
                write!(f, "ldx{} {}, {}", size, dst, format_mem_operand(*src, *off))
            }
            SbpfInsn::StoreImm { dst, off, imm, size } => {
                write!(f, "st{} {}, {}", size, format_mem_operand(*dst, *off), imm)
            }
            SbpfInsn::StoreReg { dst, off, src, size } => {
                write!(f, "stx{} {}, {}", size, format_mem_operand(*dst, *off), src)
            }
            SbpfInsn::Alu32 { op, dst, src } => {
                write!(f, "{}32 {}, {}", op, dst, src)
            }
            SbpfInsn::Alu64 { op, dst, src } => {
                write!(f, "{}64 {}, {}", op, dst, src)
            }
            SbpfInsn::Endian { dst, size, to_le } => {
                let endian = if *to_le { "le" } else { "be" };
                write!(f, "{}{} {}", endian, size, dst)
            }
            SbpfInsn::Jump { target } => {
                write!(f, "ja Block {}", target)
            }
            SbpfInsn::JumpCond { cond, dst, src, target } => {
                write!(f, "j{} {}, {}, Block {}", cond, dst, src, target)
            }
            SbpfInsn::Call { target } => {
                write!(f, "call {}", target)
            }
            SbpfInsn::Exit => {
                write!(f, "exit")
            }
            SbpfInsn::Unknown { raw } => {
                write!(f, "unknown 0x{:02x}", raw.opc)
            }
        }
    }
}

// ============================================================================
// Basic Block
// ============================================================================

/// A basic block: a sequence of instructions with single entry and exit
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Block ID (usually the starting PC)
    pub id: usize,
    /// Human-readable label
    pub label: String,
    /// Instructions in this block
    pub instructions: Vec<(usize, SbpfInsn)>, // (pc, insn)
    /// Predecessor blocks (can jump to this block)
    pub predecessors: Vec<usize>,
    /// Successor blocks (this block can jump to)
    pub successors: Vec<usize>,
}

impl BasicBlock {
    /// Create a new basic block
    pub fn new(id: usize) -> Self {
        Self {
            id,
            label: format!("block_{}", id),
            instructions: Vec::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
        }
    }

    /// Get instruction count
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    /// Get total static CU cost of this block
    pub fn cu_cost(&self) -> u64 {
        self.instructions.iter().map(|(_, insn)| insn.cu_cost()).sum()
    }

    /// Get the PC range of this block
    pub fn pc_range(&self) -> Option<Range<usize>> {
        if self.instructions.is_empty() {
            None
        } else {
            let start = self.instructions.first().unwrap().0;
            let end = self.instructions.last().unwrap().0 + 1;
            Some(start..end)
        }
    }

    /// Get the terminator instruction
    pub fn terminator(&self) -> Option<&SbpfInsn> {
        self.instructions.last().map(|(_, insn)| insn)
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}:", self.label)?;
        for (pc, insn) in &self.instructions {
            writeln!(f, "    {:04x}: {}", pc, insn)?;
        }
        Ok(())
    }
}

// ============================================================================
// Function
// ============================================================================

/// A function in the sBPF program
#[derive(Debug, Clone)]
pub struct Function {
    /// Function name
    pub name: String,
    /// Entry block ID
    pub entry: usize,
    /// All basic blocks in this function (block_id -> BasicBlock)
    pub blocks: BTreeMap<usize, BasicBlock>,
}

impl Function {
    /// Create a new function
    pub fn new(name: String, entry: usize) -> Self {
        Self {
            name,
            entry,
            blocks: BTreeMap::new(),
        }
    }

    /// Add a basic block
    pub fn add_block(&mut self, block: BasicBlock) {
        self.blocks.insert(block.id, block);
    }

    /// Get a basic block by ID
    pub fn get_block(&self, id: usize) -> Option<&BasicBlock> {
        self.blocks.get(&id)
    }

    /// Get a mutable basic block by ID
    pub fn get_block_mut(&mut self, id: usize) -> Option<&mut BasicBlock> {
        self.blocks.get_mut(&id)
    }

    /// Get total instruction count
    pub fn instruction_count(&self) -> usize {
        self.blocks.values().map(|b| b.instruction_count()).sum()
    }

    /// Get total static CU cost
    pub fn cu_cost(&self) -> u64 {
        self.blocks.values().map(|b| b.cu_cost()).sum()
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Function: {}", self.name)?;
        for block in self.blocks.values() {
            write!(f, "{}", block)?;
        }
        Ok(())
    }
}

// ============================================================================
// Program
// ============================================================================

/// A complete sBPF program
#[derive(Debug, Clone)]
pub struct SbpfProgram {
    /// Entry point function name
    pub entrypoint: String,
    /// All functions in the program
    pub functions: BTreeMap<String, Function>,
    /// Syscall registry (hash -> name)
    pub syscalls: BTreeMap<u32, String>,
}

impl SbpfProgram {
    /// Create a new empty program
    pub fn new() -> Self {
        Self {
            entrypoint: String::new(),
            functions: BTreeMap::new(),
            syscalls: BTreeMap::new(),
        }
    }

    /// Add a function
    pub fn add_function(&mut self, func: Function) {
        self.functions.insert(func.name.clone(), func);
    }

    /// Get the entrypoint function
    pub fn get_entrypoint(&self) -> Option<&Function> {
        self.functions.get(&self.entrypoint)
    }

    /// Register a syscall
    pub fn register_syscall(&mut self, hash: u32, name: String) {
        self.syscalls.insert(hash, name);
    }
}

impl Default for SbpfProgram {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SbpfProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== sBPF Program ===")?;
        writeln!(f, "Entrypoint: {}", self.entrypoint)?;
        writeln!(f)?;
        for func in self.functions.values() {
            writeln!(f, "{}", func)?;
        }
        Ok(())
    }
}
