//! Serializable IR types for cross-module communication
//!
//! These types mirror the frontend IR but are serializable via serde

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Serializable sBPF Program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Program {
    pub entrypoint: String,
    pub functions: BTreeMap<String, Function>,
}

/// Serializable Function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub entry: usize,
    pub blocks: BTreeMap<usize, BasicBlock>,
}

/// Serializable Basic Block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicBlock {
    pub id: usize,
    pub label: String,
    pub instructions: Vec<Instruction>,
    pub successors: Vec<usize>,
}

/// Serializable Instruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instruction {
    pub pc: usize,
    pub op: Op,
}

/// Serializable Operation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Op {
    // Load/Store
    LoadImm64 { dst: u8, imm: i64 },
    Load { dst: u8, src: u8, off: i16, size: MemSize },
    StoreImm { dst: u8, off: i16, imm: i64, size: MemSize },
    StoreReg { dst: u8, off: i16, src: u8, size: MemSize },
    
    // ALU
    Alu64 { op: AluOp, dst: u8, src: Operand },
    Alu32 { op: AluOp, dst: u8, src: Operand },
    
    // Endian
    Endian { dst: u8, size: u8, to_le: bool },
    
    // Control Flow
    Jump { target: usize },
    JumpCond { cond: JmpCond, dst: u8, src: Operand, target: usize },
    Call { target: CallTarget },
    Exit,
    
    // Unknown
    Unknown { opcode: u8 },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MemSize {
    Byte,
    Half,
    Word,
    DoubleWord,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AluOp {
    Add, Sub, Mul, Div, Or, And, Lsh, Rsh, Neg, Mod, Xor, Mov, Arsh,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum JmpCond {
    Eq, Ne, Gt, Ge, Lt, Le, Set, Sgt, Sge, Slt, Sle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operand {
    Reg(u8),
    Imm(i64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CallTarget {
    Internal { pc: usize, name: Option<String> },
    Syscall { hash: u32, name: Option<String> },
    Register(u8),
}
