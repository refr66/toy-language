//! JSON export for IR types
//!
//! Provides serialization for frontend IR to communicate with mlir-backend

use serde::{Serialize, Deserialize};
use std::collections::BTreeMap;
use crate::ir::{SbpfProgram, Function, BasicBlock, SbpfInsn, Operand, CallTarget};

/// Serializable Program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonProgram {
    pub entrypoint: String,
    pub functions: BTreeMap<String, JsonFunction>,
}

/// Serializable Function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonFunction {
    pub name: String,
    pub entry: usize,
    pub blocks: BTreeMap<usize, JsonBlock>,
}

/// Serializable Basic Block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonBlock {
    pub id: usize,
    pub label: String,
    pub instructions: Vec<JsonInstruction>,
    pub successors: Vec<usize>,
}

/// Serializable Instruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonInstruction {
    pub pc: usize,
    pub op: JsonOp,
}

/// Serializable Operation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum JsonOp {
    LoadImm64 { dst: u8, imm: i64 },
    Load { dst: u8, src: u8, off: i16, size: String },
    StoreImm { dst: u8, off: i16, imm: i64, size: String },
    StoreReg { dst: u8, off: i16, src: u8, size: String },
    Alu64 { op: String, dst: u8, src: JsonOperand },
    Alu32 { op: String, dst: u8, src: JsonOperand },
    Endian { dst: u8, size: u8, to_le: bool },
    Jump { target: usize },
    JumpCond { cond: String, dst: u8, src: JsonOperand, target: usize },
    Call { target: JsonCallTarget },
    Exit,
    Unknown { opcode: u8 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum JsonOperand {
    Reg { reg: u8 },
    Imm { value: i64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum JsonCallTarget {
    Internal { pc: usize, name: Option<String> },
    Syscall { hash: u32, name: Option<String> },
    Register { reg: u8 },
}

// ============================================================================
// Conversion from IR types
// ============================================================================

impl From<&SbpfProgram> for JsonProgram {
    fn from(program: &SbpfProgram) -> Self {
        JsonProgram {
            entrypoint: program.entrypoint.clone(),
            functions: program.functions.iter()
                .map(|(k, v)| (k.clone(), JsonFunction::from(v)))
                .collect(),
        }
    }
}

impl From<&Function> for JsonFunction {
    fn from(func: &Function) -> Self {
        JsonFunction {
            name: func.name.clone(),
            entry: func.entry,
            blocks: func.blocks.iter()
                .map(|(k, v)| (*k, JsonBlock::from(v)))
                .collect(),
        }
    }
}

impl From<&BasicBlock> for JsonBlock {
    fn from(block: &BasicBlock) -> Self {
        JsonBlock {
            id: block.id,
            label: block.label.clone(),
            instructions: block.instructions.iter()
                .map(|(pc, insn)| JsonInstruction {
                    pc: *pc,
                    op: JsonOp::from(insn),
                })
                .collect(),
            successors: block.successors.clone(),
        }
    }
}

impl From<&SbpfInsn> for JsonOp {
    fn from(insn: &SbpfInsn) -> Self {
        match insn {
            SbpfInsn::LoadImm64 { dst, imm } => JsonOp::LoadImm64 { dst: dst.0, imm: *imm },
            SbpfInsn::Load { dst, src, off, size } => JsonOp::Load {
                dst: dst.0,
                src: src.0,
                off: *off,
                size: format!("{:?}", size),
            },
            SbpfInsn::StoreImm { dst, off, imm, size } => JsonOp::StoreImm {
                dst: dst.0,
                off: *off,
                imm: *imm,
                size: format!("{:?}", size),
            },
            SbpfInsn::StoreReg { dst, off, src, size } => JsonOp::StoreReg {
                dst: dst.0,
                off: *off,
                src: src.0,
                size: format!("{:?}", size),
            },
            SbpfInsn::Alu64 { op, dst, src } => JsonOp::Alu64 {
                op: format!("{:?}", op),
                dst: dst.0,
                src: JsonOperand::from(src),
            },
            SbpfInsn::Alu32 { op, dst, src } => JsonOp::Alu32 {
                op: format!("{:?}", op),
                dst: dst.0,
                src: JsonOperand::from(src),
            },
            SbpfInsn::Endian { dst, size, to_le } => JsonOp::Endian {
                dst: dst.0,
                size: *size,
                to_le: *to_le,
            },
            SbpfInsn::Jump { target } => JsonOp::Jump { target: *target },
            SbpfInsn::JumpCond { cond, dst, src, target } => JsonOp::JumpCond {
                cond: format!("{:?}", cond),
                dst: dst.0,
                src: JsonOperand::from(src),
                target: *target,
            },
            SbpfInsn::Call { target } => JsonOp::Call {
                target: JsonCallTarget::from(target),
            },
            SbpfInsn::Exit => JsonOp::Exit,
            SbpfInsn::Unknown { raw } => JsonOp::Unknown { opcode: raw.opc },
        }
    }
}

impl From<&Operand> for JsonOperand {
    fn from(op: &Operand) -> Self {
        match op {
            Operand::Reg(r) => JsonOperand::Reg { reg: r.0 },
            Operand::Imm(i) => JsonOperand::Imm { value: *i },
        }
    }
}

impl From<&CallTarget> for JsonCallTarget {
    fn from(target: &CallTarget) -> Self {
        match target {
            CallTarget::Internal { pc, name } => JsonCallTarget::Internal {
                pc: *pc,
                name: name.clone(),
            },
            CallTarget::Syscall { hash, name } => JsonCallTarget::Syscall {
                hash: *hash,
                name: name.clone(),
            },
            CallTarget::Register(r) => JsonCallTarget::Register { reg: r.0 },
        }
    }
}

// ============================================================================
// Export functions
// ============================================================================

/// Export program to JSON string
pub fn to_json(program: &SbpfProgram) -> Result<String, serde_json::Error> {
    let json_program = JsonProgram::from(program);
    serde_json::to_string_pretty(&json_program)
}

/// Export program to JSON string (compact)
pub fn to_json_compact(program: &SbpfProgram) -> Result<String, serde_json::Error> {
    let json_program = JsonProgram::from(program);
    serde_json::to_string(&json_program)
}
