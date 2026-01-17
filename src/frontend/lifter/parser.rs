//! Bytecode parsing and IR lifting
//!
//! This module handles:
//! - Instruction decoding with LD_DW merging
//! - Basic block splitting
//! - Conversion to high-level IR

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::ebpf::{self, RawInsn, INSN_SIZE};
use crate::ir::*;
use super::elf::{ElfLoader, ElfInfo, ElfError, Relocation};

/// Lifter error types
#[derive(Debug, thiserror::Error)]
pub enum LifterError {
    #[error("ELF error: {0}")]
    ElfError(#[from] ElfError),
    
    #[error("Invalid instruction at PC {0}")]
    InvalidInstruction(usize),
    
    #[error("Invalid jump target at PC {pc}: target {target}")]
    InvalidJumpTarget { pc: usize, target: usize },
}

/// The Lifter: converts ELF bytecode to IR
pub struct Lifter<'a> {
    /// Raw ELF bytes
    #[allow(dead_code)]
    elf_bytes: &'a [u8],
    /// Text section bytes
    text_bytes: &'a [u8],
    /// ELF info (vaddr, relocations, functions)
    elf_info: ElfInfo,
    /// Entrypoint PC (instruction index)
    entrypoint_pc: usize,
}

impl<'a> Lifter<'a> {
    /// Create a new Lifter from ELF bytes
    pub fn new(elf_bytes: &'a [u8]) -> Result<Self, LifterError> {
        let elf_info = ElfLoader::load(elf_bytes)?;
        
        // Get text section bytes
        let text_bytes = &elf_bytes[elf_info.text_offset..elf_info.text_offset + elf_info.text_size];
        
        // Calculate entrypoint PC
        let entrypoint_pc = if elf_info.entry_vaddr >= elf_info.text_vaddr {
            ((elf_info.entry_vaddr - elf_info.text_vaddr) / INSN_SIZE as u64) as usize
        } else {
            0
        };
        
        Ok(Self {
            elf_bytes,
            text_bytes,
            elf_info,
            entrypoint_pc,
        })
    }
    
    /// Get the entrypoint PC
    pub fn entrypoint_pc(&self) -> usize {
        self.entrypoint_pc
    }
    
    /// Get text section bytes
    pub fn text_bytes(&self) -> &[u8] {
        self.text_bytes
    }
    
    /// Get function symbols
    pub fn functions(&self) -> &BTreeMap<u64, String> {
        &self.elf_info.functions
    }
    
    /// Get relocations
    pub fn relocations(&self) -> &HashMap<usize, Relocation> {
        &self.elf_info.relocations
    }

    /// Decode all instructions with LD_DW merging
    pub fn decode_instructions(&self) -> Result<Vec<(usize, RawInsn)>, LifterError> {
        let mut instructions = Vec::new();
        let mut pc = 0;
        let total_insns = self.text_bytes.len() / INSN_SIZE;
        
        while pc < total_insns {
            let mut insn = RawInsn::decode(self.text_bytes, pc);
            
            // Handle LD_DW_IMM (16 bytes = 2 slots)
            if insn.is_lddw() {
                if pc + 1 >= total_insns {
                    return Err(LifterError::InvalidInstruction(pc));
                }
                insn.merge_lddw(self.text_bytes);
                instructions.push((pc, insn));
                pc += 2; // Skip the second slot
            } else {
                instructions.push((pc, insn));
                pc += 1;
            }
        }
        
        Ok(instructions)
    }

    /// Find all basic block leaders (starting PCs)
    fn find_leaders(&self, instructions: &[(usize, RawInsn)]) -> BTreeSet<usize> {
        let mut leaders = BTreeSet::new();
        
        // First instruction is always a leader
        if let Some((pc, _)) = instructions.first() {
            leaders.insert(*pc);
        }
        
        // Entrypoint is a leader
        leaders.insert(self.entrypoint_pc);
        
        // Function symbols are leaders
        for &vaddr in self.elf_info.functions.keys() {
            if vaddr >= self.elf_info.text_vaddr {
                let pc = ((vaddr - self.elf_info.text_vaddr) / INSN_SIZE as u64) as usize;
                leaders.insert(pc);
            }
        }
        
        // Find jump targets and instructions after jumps/calls
        for (pc, insn) in instructions {
            if insn.is_jump() {
                let target = insn.jump_target();
                leaders.insert(target);
                
                if insn.is_cond_jump() {
                    leaders.insert(pc + 1);
                }
                if insn.is_uncond_jump() {
                    leaders.insert(pc + 1);
                }
            } else if insn.is_call() {
                leaders.insert(pc + 1);
            } else if insn.is_exit() {
                leaders.insert(pc + 1);
            }
        }
        
        leaders
    }

    /// Convert raw instruction to high-level IR
    fn lift_instruction(&self, raw: &RawInsn) -> SbpfInsn {
        let dst = Reg(raw.dst);
        let src_reg = Reg(raw.src);
        
        match raw.opc {
            // Load immediate 64-bit
            ebpf::LD_DW_IMM => SbpfInsn::LoadImm64 { dst, imm: raw.imm },
            
            // Load from memory
            ebpf::LD_B_REG => SbpfInsn::Load { dst, src: src_reg, off: raw.off, size: MemSize::Byte },
            ebpf::LD_H_REG => SbpfInsn::Load { dst, src: src_reg, off: raw.off, size: MemSize::Half },
            ebpf::LD_W_REG => SbpfInsn::Load { dst, src: src_reg, off: raw.off, size: MemSize::Word },
            ebpf::LD_DW_REG => SbpfInsn::Load { dst, src: src_reg, off: raw.off, size: MemSize::DoubleWord },
            
            // Store immediate
            ebpf::ST_B_IMM => SbpfInsn::StoreImm { dst, off: raw.off, imm: raw.imm, size: MemSize::Byte },
            ebpf::ST_H_IMM => SbpfInsn::StoreImm { dst, off: raw.off, imm: raw.imm, size: MemSize::Half },
            ebpf::ST_W_IMM => SbpfInsn::StoreImm { dst, off: raw.off, imm: raw.imm, size: MemSize::Word },
            ebpf::ST_DW_IMM => SbpfInsn::StoreImm { dst, off: raw.off, imm: raw.imm, size: MemSize::DoubleWord },
            
            // Store register
            ebpf::ST_B_REG => SbpfInsn::StoreReg { dst, off: raw.off, src: src_reg, size: MemSize::Byte },
            ebpf::ST_H_REG => SbpfInsn::StoreReg { dst, off: raw.off, src: src_reg, size: MemSize::Half },
            ebpf::ST_W_REG => SbpfInsn::StoreReg { dst, off: raw.off, src: src_reg, size: MemSize::Word },
            ebpf::ST_DW_REG => SbpfInsn::StoreReg { dst, off: raw.off, src: src_reg, size: MemSize::DoubleWord },
            
            // 32-bit ALU
            ebpf::ADD32_IMM => SbpfInsn::Alu32 { op: AluOp::Add, dst, src: Operand::Imm(raw.imm) },
            ebpf::ADD32_REG => SbpfInsn::Alu32 { op: AluOp::Add, dst, src: Operand::Reg(src_reg) },
            ebpf::SUB32_IMM => SbpfInsn::Alu32 { op: AluOp::Sub, dst, src: Operand::Imm(raw.imm) },
            ebpf::SUB32_REG => SbpfInsn::Alu32 { op: AluOp::Sub, dst, src: Operand::Reg(src_reg) },
            ebpf::MUL32_IMM => SbpfInsn::Alu32 { op: AluOp::Mul, dst, src: Operand::Imm(raw.imm) },
            ebpf::MUL32_REG => SbpfInsn::Alu32 { op: AluOp::Mul, dst, src: Operand::Reg(src_reg) },
            ebpf::DIV32_IMM => SbpfInsn::Alu32 { op: AluOp::Div, dst, src: Operand::Imm(raw.imm) },
            ebpf::DIV32_REG => SbpfInsn::Alu32 { op: AluOp::Div, dst, src: Operand::Reg(src_reg) },
            ebpf::OR32_IMM => SbpfInsn::Alu32 { op: AluOp::Or, dst, src: Operand::Imm(raw.imm) },
            ebpf::OR32_REG => SbpfInsn::Alu32 { op: AluOp::Or, dst, src: Operand::Reg(src_reg) },
            ebpf::AND32_IMM => SbpfInsn::Alu32 { op: AluOp::And, dst, src: Operand::Imm(raw.imm) },
            ebpf::AND32_REG => SbpfInsn::Alu32 { op: AluOp::And, dst, src: Operand::Reg(src_reg) },
            ebpf::LSH32_IMM => SbpfInsn::Alu32 { op: AluOp::Lsh, dst, src: Operand::Imm(raw.imm) },
            ebpf::LSH32_REG => SbpfInsn::Alu32 { op: AluOp::Lsh, dst, src: Operand::Reg(src_reg) },
            ebpf::RSH32_IMM => SbpfInsn::Alu32 { op: AluOp::Rsh, dst, src: Operand::Imm(raw.imm) },
            ebpf::RSH32_REG => SbpfInsn::Alu32 { op: AluOp::Rsh, dst, src: Operand::Reg(src_reg) },
            ebpf::NEG32 => SbpfInsn::Alu32 { op: AluOp::Neg, dst, src: Operand::Imm(0) },
            ebpf::MOD32_IMM => SbpfInsn::Alu32 { op: AluOp::Mod, dst, src: Operand::Imm(raw.imm) },
            ebpf::MOD32_REG => SbpfInsn::Alu32 { op: AluOp::Mod, dst, src: Operand::Reg(src_reg) },
            ebpf::XOR32_IMM => SbpfInsn::Alu32 { op: AluOp::Xor, dst, src: Operand::Imm(raw.imm) },
            ebpf::XOR32_REG => SbpfInsn::Alu32 { op: AluOp::Xor, dst, src: Operand::Reg(src_reg) },
            ebpf::MOV32_IMM => SbpfInsn::Alu32 { op: AluOp::Mov, dst, src: Operand::Imm(raw.imm) },
            ebpf::MOV32_REG => SbpfInsn::Alu32 { op: AluOp::Mov, dst, src: Operand::Reg(src_reg) },
            ebpf::ARSH32_IMM => SbpfInsn::Alu32 { op: AluOp::Arsh, dst, src: Operand::Imm(raw.imm) },
            ebpf::ARSH32_REG => SbpfInsn::Alu32 { op: AluOp::Arsh, dst, src: Operand::Reg(src_reg) },
            
            // 64-bit ALU
            ebpf::ADD64_IMM => SbpfInsn::Alu64 { op: AluOp::Add, dst, src: Operand::Imm(raw.imm) },
            ebpf::ADD64_REG => SbpfInsn::Alu64 { op: AluOp::Add, dst, src: Operand::Reg(src_reg) },
            ebpf::SUB64_IMM => SbpfInsn::Alu64 { op: AluOp::Sub, dst, src: Operand::Imm(raw.imm) },
            ebpf::SUB64_REG => SbpfInsn::Alu64 { op: AluOp::Sub, dst, src: Operand::Reg(src_reg) },
            ebpf::MUL64_IMM => SbpfInsn::Alu64 { op: AluOp::Mul, dst, src: Operand::Imm(raw.imm) },
            ebpf::MUL64_REG => SbpfInsn::Alu64 { op: AluOp::Mul, dst, src: Operand::Reg(src_reg) },
            ebpf::DIV64_IMM => SbpfInsn::Alu64 { op: AluOp::Div, dst, src: Operand::Imm(raw.imm) },
            ebpf::DIV64_REG => SbpfInsn::Alu64 { op: AluOp::Div, dst, src: Operand::Reg(src_reg) },
            ebpf::OR64_IMM => SbpfInsn::Alu64 { op: AluOp::Or, dst, src: Operand::Imm(raw.imm) },
            ebpf::OR64_REG => SbpfInsn::Alu64 { op: AluOp::Or, dst, src: Operand::Reg(src_reg) },
            ebpf::AND64_IMM => SbpfInsn::Alu64 { op: AluOp::And, dst, src: Operand::Imm(raw.imm) },
            ebpf::AND64_REG => SbpfInsn::Alu64 { op: AluOp::And, dst, src: Operand::Reg(src_reg) },
            ebpf::LSH64_IMM => SbpfInsn::Alu64 { op: AluOp::Lsh, dst, src: Operand::Imm(raw.imm) },
            ebpf::LSH64_REG => SbpfInsn::Alu64 { op: AluOp::Lsh, dst, src: Operand::Reg(src_reg) },
            ebpf::RSH64_IMM => SbpfInsn::Alu64 { op: AluOp::Rsh, dst, src: Operand::Imm(raw.imm) },
            ebpf::RSH64_REG => SbpfInsn::Alu64 { op: AluOp::Rsh, dst, src: Operand::Reg(src_reg) },
            ebpf::NEG64 => SbpfInsn::Alu64 { op: AluOp::Neg, dst, src: Operand::Imm(0) },
            ebpf::MOD64_IMM => SbpfInsn::Alu64 { op: AluOp::Mod, dst, src: Operand::Imm(raw.imm) },
            ebpf::MOD64_REG => SbpfInsn::Alu64 { op: AluOp::Mod, dst, src: Operand::Reg(src_reg) },
            ebpf::XOR64_IMM => SbpfInsn::Alu64 { op: AluOp::Xor, dst, src: Operand::Imm(raw.imm) },
            ebpf::XOR64_REG => SbpfInsn::Alu64 { op: AluOp::Xor, dst, src: Operand::Reg(src_reg) },
            ebpf::MOV64_IMM => SbpfInsn::Alu64 { op: AluOp::Mov, dst, src: Operand::Imm(raw.imm) },
            ebpf::MOV64_REG => SbpfInsn::Alu64 { op: AluOp::Mov, dst, src: Operand::Reg(src_reg) },
            ebpf::ARSH64_IMM => SbpfInsn::Alu64 { op: AluOp::Arsh, dst, src: Operand::Imm(raw.imm) },
            ebpf::ARSH64_REG => SbpfInsn::Alu64 { op: AluOp::Arsh, dst, src: Operand::Reg(src_reg) },
            
            // Endianness
            ebpf::LE => SbpfInsn::Endian { dst, size: raw.imm as u8, to_le: true },
            ebpf::BE => SbpfInsn::Endian { dst, size: raw.imm as u8, to_le: false },
            
            // Jumps
            ebpf::JA => SbpfInsn::Jump { target: raw.jump_target() },
            
            // Conditional jumps
            ebpf::JEQ64_IMM => SbpfInsn::JumpCond { cond: JmpCond::Eq, dst, src: Operand::Imm(raw.imm), target: raw.jump_target() },
            ebpf::JEQ64_REG => SbpfInsn::JumpCond { cond: JmpCond::Eq, dst, src: Operand::Reg(src_reg), target: raw.jump_target() },
            ebpf::JGT64_IMM => SbpfInsn::JumpCond { cond: JmpCond::Gt, dst, src: Operand::Imm(raw.imm), target: raw.jump_target() },
            ebpf::JGT64_REG => SbpfInsn::JumpCond { cond: JmpCond::Gt, dst, src: Operand::Reg(src_reg), target: raw.jump_target() },
            ebpf::JGE64_IMM => SbpfInsn::JumpCond { cond: JmpCond::Ge, dst, src: Operand::Imm(raw.imm), target: raw.jump_target() },
            ebpf::JGE64_REG => SbpfInsn::JumpCond { cond: JmpCond::Ge, dst, src: Operand::Reg(src_reg), target: raw.jump_target() },
            ebpf::JLT64_IMM => SbpfInsn::JumpCond { cond: JmpCond::Lt, dst, src: Operand::Imm(raw.imm), target: raw.jump_target() },
            ebpf::JLT64_REG => SbpfInsn::JumpCond { cond: JmpCond::Lt, dst, src: Operand::Reg(src_reg), target: raw.jump_target() },
            ebpf::JLE64_IMM => SbpfInsn::JumpCond { cond: JmpCond::Le, dst, src: Operand::Imm(raw.imm), target: raw.jump_target() },
            ebpf::JLE64_REG => SbpfInsn::JumpCond { cond: JmpCond::Le, dst, src: Operand::Reg(src_reg), target: raw.jump_target() },
            ebpf::JSET64_IMM => SbpfInsn::JumpCond { cond: JmpCond::Set, dst, src: Operand::Imm(raw.imm), target: raw.jump_target() },
            ebpf::JSET64_REG => SbpfInsn::JumpCond { cond: JmpCond::Set, dst, src: Operand::Reg(src_reg), target: raw.jump_target() },
            ebpf::JNE64_IMM => SbpfInsn::JumpCond { cond: JmpCond::Ne, dst, src: Operand::Imm(raw.imm), target: raw.jump_target() },
            ebpf::JNE64_REG => SbpfInsn::JumpCond { cond: JmpCond::Ne, dst, src: Operand::Reg(src_reg), target: raw.jump_target() },
            ebpf::JSGT64_IMM => SbpfInsn::JumpCond { cond: JmpCond::Sgt, dst, src: Operand::Imm(raw.imm), target: raw.jump_target() },
            ebpf::JSGT64_REG => SbpfInsn::JumpCond { cond: JmpCond::Sgt, dst, src: Operand::Reg(src_reg), target: raw.jump_target() },
            ebpf::JSGE64_IMM => SbpfInsn::JumpCond { cond: JmpCond::Sge, dst, src: Operand::Imm(raw.imm), target: raw.jump_target() },
            ebpf::JSGE64_REG => SbpfInsn::JumpCond { cond: JmpCond::Sge, dst, src: Operand::Reg(src_reg), target: raw.jump_target() },
            ebpf::JSLT64_IMM => SbpfInsn::JumpCond { cond: JmpCond::Slt, dst, src: Operand::Imm(raw.imm), target: raw.jump_target() },
            ebpf::JSLT64_REG => SbpfInsn::JumpCond { cond: JmpCond::Slt, dst, src: Operand::Reg(src_reg), target: raw.jump_target() },
            ebpf::JSLE64_IMM => SbpfInsn::JumpCond { cond: JmpCond::Sle, dst, src: Operand::Imm(raw.imm), target: raw.jump_target() },
            ebpf::JSLE64_REG => SbpfInsn::JumpCond { cond: JmpCond::Sle, dst, src: Operand::Reg(src_reg), target: raw.jump_target() },
            
            // Call
            ebpf::CALL_IMM => {
                let byte_offset = raw.ptr * INSN_SIZE;
                let target = if let Some(reloc) = self.elf_info.relocations.get(&byte_offset) {
                    CallTarget::Syscall { 
                        hash: raw.imm as u32, 
                        name: reloc.symbol_name.clone() 
                    }
                } else {
                    let target_pc = (raw.ptr as i64 + raw.imm + 1) as usize;
                    let target_vaddr = self.elf_info.text_vaddr + (target_pc as u64) * INSN_SIZE as u64;
                    let name = self.elf_info.functions.get(&target_vaddr).cloned();
                    CallTarget::Internal { pc: target_pc, name }
                };
                SbpfInsn::Call { target }
            }
            ebpf::CALL_REG => SbpfInsn::Call { target: CallTarget::Register(dst) },
            
            // Exit
            ebpf::EXIT => SbpfInsn::Exit,
            
            // Unknown
            _ => SbpfInsn::Unknown { raw: raw.clone() },
        }
    }

    /// Lift the entire ELF to a SbpfProgram
    pub fn lift(&self) -> Result<SbpfProgram, LifterError> {
        let instructions = self.decode_instructions()?;
        let leaders = self.find_leaders(&instructions);
        
        // Create basic blocks
        let mut blocks: BTreeMap<usize, BasicBlock> = BTreeMap::new();
        let leader_vec: Vec<usize> = leaders.iter().cloned().collect();
        
        for (i, &leader) in leader_vec.iter().enumerate() {
            let next_leader = leader_vec.get(i + 1).cloned();
            let mut block = BasicBlock::new(leader);
            
            // Set label if this is a known function
            let vaddr = self.elf_info.text_vaddr + (leader as u64) * INSN_SIZE as u64;
            if let Some(name) = self.elf_info.functions.get(&vaddr) {
                block.label = name.clone();
            }
            
            // Add instructions to block
            for (pc, raw_insn) in &instructions {
                if *pc < leader {
                    continue;
                }
                if let Some(next) = next_leader {
                    if *pc >= next {
                        break;
                    }
                }
                
                let insn = self.lift_instruction(raw_insn);
                block.instructions.push((*pc, insn));
                
                if raw_insn.is_terminator() {
                    break;
                }
            }
            
            blocks.insert(leader, block);
        }
        
        // Link predecessors and successors
        let block_ids: Vec<usize> = blocks.keys().cloned().collect();
        for &block_id in &block_ids {
            let block = blocks.get(&block_id).unwrap();
            if let Some((_, last_insn)) = block.instructions.last() {
                let successors: Vec<usize> = match last_insn {
                    SbpfInsn::Jump { target } => vec![*target],
                    SbpfInsn::JumpCond { target, .. } => {
                        let last_pc = block.instructions.last().unwrap().0;
                        let fallthrough = last_pc + 1;
                        if blocks.contains_key(target) && blocks.contains_key(&fallthrough) {
                            vec![fallthrough, *target]
                        } else if blocks.contains_key(target) {
                            vec![*target]
                        } else if blocks.contains_key(&fallthrough) {
                            vec![fallthrough]
                        } else {
                            vec![]
                        }
                    }
                    SbpfInsn::Call { target: CallTarget::Internal { .. } } => {
                        let last_pc = block.instructions.last().unwrap().0;
                        let return_point = last_pc + 1;
                        if blocks.contains_key(&return_point) {
                            vec![return_point]
                        } else {
                            vec![]
                        }
                    }
                    SbpfInsn::Exit => vec![],
                    _ => {
                        let last_pc = block.instructions.last().unwrap().0;
                        let next_pc = last_pc + 1;
                        if blocks.contains_key(&next_pc) {
                            vec![next_pc]
                        } else {
                            vec![]
                        }
                    }
                };
                
                let block = blocks.get_mut(&block_id).unwrap();
                block.successors = successors.clone();
                
                for &succ in &successors {
                    if let Some(succ_block) = blocks.get_mut(&succ) {
                        if !succ_block.predecessors.contains(&block_id) {
                            succ_block.predecessors.push(block_id);
                        }
                    }
                }
            }
        }
        
        // Build the program with MULTIPLE FUNCTIONS
        let mut program = SbpfProgram::new();
        
        // Get all function entry points from symbols, sorted by PC
        let mut func_entries: Vec<(usize, String)> = self.elf_info.functions
            .iter()
            .filter_map(|(&vaddr, name)| {
                if vaddr >= self.elf_info.text_vaddr {
                    let pc = ((vaddr - self.elf_info.text_vaddr) / INSN_SIZE as u64) as usize;
                    Some((pc, name.clone()))
                } else {
                    None
                }
            })
            .collect();
        func_entries.sort_by_key(|(pc, _)| *pc);
        
        // If no symbols, use entrypoint as the only function
        if func_entries.is_empty() {
            func_entries.push((self.entrypoint_pc, "entrypoint".to_string()));
        }
        
        // Calculate function boundaries: each function ends where the next begins (or at code end)
        let total_insns = self.text_bytes.len() / INSN_SIZE;
        let mut func_ranges: Vec<(usize, usize, String)> = Vec::new(); // (start_pc, end_pc_exclusive, name)
        
        for (i, (start_pc, name)) in func_entries.iter().enumerate() {
            let end_pc = if i + 1 < func_entries.len() {
                func_entries[i + 1].0
            } else {
                total_insns
            };
            func_ranges.push((*start_pc, end_pc, name.clone()));
        }
        
        // Assign blocks to functions
        for (start_pc, end_pc, func_name) in &func_ranges {
            let mut func = Function::new(func_name.clone(), *start_pc);
            
            // Filter blocks that belong to this function
            for (&block_id, block) in &blocks {
                // A block belongs to a function if its start PC is within the function's range
                if block_id >= *start_pc && block_id < *end_pc && !block.instructions.is_empty() {
                    func.add_block(block.clone());
                }
            }
            
            // Only add function if it has blocks
            if !func.blocks.is_empty() {
                program.add_function(func);
            }
        }
        
        // Set entrypoint
        let entry_vaddr = self.elf_info.text_vaddr + (self.entrypoint_pc as u64) * INSN_SIZE as u64;
        program.entrypoint = self.elf_info.functions
            .get(&entry_vaddr)
            .cloned()
            .unwrap_or_else(|| "entrypoint".to_string());
        
        Ok(program)
    }
}
