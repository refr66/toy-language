//! sBPF instruction definitions and constants
//!
//! This module defines all sBPF opcodes and the instruction structure.

use byteorder::{ByteOrder, LittleEndian};
use std::fmt;

/// Size of an sBPF instruction in bytes (8 bytes per slot)
pub const INSN_SIZE: usize = 8;

// ============================================================================
// Instruction classes (3 bits)
// ============================================================================
pub const BPF_LD: u8 = 0x00;
pub const BPF_LDX: u8 = 0x01;
pub const BPF_ST: u8 = 0x02;
pub const BPF_STX: u8 = 0x03;
pub const BPF_ALU32_LOAD: u8 = 0x04;
pub const BPF_JMP64: u8 = 0x05;
pub const BPF_JMP32: u8 = 0x06;
pub const BPF_PQR: u8 = 0x06;
pub const BPF_ALU64_STORE: u8 = 0x07;

// ============================================================================
// Size modifiers
// ============================================================================
pub const BPF_W: u8 = 0x00;   // word (4 bytes)
pub const BPF_H: u8 = 0x08;   // half-word (2 bytes)
pub const BPF_B: u8 = 0x10;   // byte (1 byte)
pub const BPF_DW: u8 = 0x18;  // double word (8 bytes)

// ============================================================================
// Mode modifiers
// ============================================================================
pub const BPF_IMM: u8 = 0x00;
pub const BPF_ABS: u8 = 0x20;
pub const BPF_IND: u8 = 0x40;
pub const BPF_MEM: u8 = 0x60;

// ============================================================================
// Source modifiers
// ============================================================================
pub const BPF_K: u8 = 0x00;   // immediate
pub const BPF_X: u8 = 0x08;   // register

// ============================================================================
// ALU operation codes
// ============================================================================
pub const BPF_ADD: u8 = 0x00;
pub const BPF_SUB: u8 = 0x10;
pub const BPF_MUL: u8 = 0x20;
pub const BPF_DIV: u8 = 0x30;
pub const BPF_OR: u8 = 0x40;
pub const BPF_AND: u8 = 0x50;
pub const BPF_LSH: u8 = 0x60;
pub const BPF_RSH: u8 = 0x70;
pub const BPF_NEG: u8 = 0x80;
pub const BPF_MOD: u8 = 0x90;
pub const BPF_XOR: u8 = 0xa0;
pub const BPF_MOV: u8 = 0xb0;
pub const BPF_ARSH: u8 = 0xc0;
pub const BPF_END: u8 = 0xd0;

// ============================================================================
// Jump operation codes
// ============================================================================
pub const BPF_JA: u8 = 0x00;
pub const BPF_JEQ: u8 = 0x10;
pub const BPF_JGT: u8 = 0x20;
pub const BPF_JGE: u8 = 0x30;
pub const BPF_JSET: u8 = 0x40;
pub const BPF_JNE: u8 = 0x50;
pub const BPF_JSGT: u8 = 0x60;
pub const BPF_JSGE: u8 = 0x70;
pub const BPF_CALL: u8 = 0x80;
pub const BPF_EXIT: u8 = 0x90;
pub const BPF_JLT: u8 = 0xa0;
pub const BPF_JLE: u8 = 0xb0;
pub const BPF_JSLT: u8 = 0xc0;
pub const BPF_JSLE: u8 = 0xd0;

// ============================================================================
// Composed opcodes
// ============================================================================

// Load double-word immediate (16 bytes total - 2 slots)
pub const LD_DW_IMM: u8 = BPF_LD | BPF_IMM | BPF_DW;  // 0x18

// Load from memory
pub const LD_B_REG: u8 = BPF_LDX | BPF_MEM | BPF_B;
pub const LD_H_REG: u8 = BPF_LDX | BPF_MEM | BPF_H;
pub const LD_W_REG: u8 = BPF_LDX | BPF_MEM | BPF_W;
pub const LD_DW_REG: u8 = BPF_LDX | BPF_MEM | BPF_DW;

// Store immediate
pub const ST_B_IMM: u8 = BPF_ST | BPF_MEM | BPF_B;
pub const ST_H_IMM: u8 = BPF_ST | BPF_MEM | BPF_H;
pub const ST_W_IMM: u8 = BPF_ST | BPF_MEM | BPF_W;
pub const ST_DW_IMM: u8 = BPF_ST | BPF_MEM | BPF_DW;

// Store from register
pub const ST_B_REG: u8 = BPF_STX | BPF_MEM | BPF_B;
pub const ST_H_REG: u8 = BPF_STX | BPF_MEM | BPF_H;
pub const ST_W_REG: u8 = BPF_STX | BPF_MEM | BPF_W;
pub const ST_DW_REG: u8 = BPF_STX | BPF_MEM | BPF_DW;

// 32-bit ALU
pub const ADD32_IMM: u8 = BPF_ALU32_LOAD | BPF_K | BPF_ADD;
pub const ADD32_REG: u8 = BPF_ALU32_LOAD | BPF_X | BPF_ADD;
pub const SUB32_IMM: u8 = BPF_ALU32_LOAD | BPF_K | BPF_SUB;
pub const SUB32_REG: u8 = BPF_ALU32_LOAD | BPF_X | BPF_SUB;
pub const MUL32_IMM: u8 = BPF_ALU32_LOAD | BPF_K | BPF_MUL;
pub const MUL32_REG: u8 = BPF_ALU32_LOAD | BPF_X | BPF_MUL;
pub const DIV32_IMM: u8 = BPF_ALU32_LOAD | BPF_K | BPF_DIV;
pub const DIV32_REG: u8 = BPF_ALU32_LOAD | BPF_X | BPF_DIV;
pub const OR32_IMM: u8 = BPF_ALU32_LOAD | BPF_K | BPF_OR;
pub const OR32_REG: u8 = BPF_ALU32_LOAD | BPF_X | BPF_OR;
pub const AND32_IMM: u8 = BPF_ALU32_LOAD | BPF_K | BPF_AND;
pub const AND32_REG: u8 = BPF_ALU32_LOAD | BPF_X | BPF_AND;
pub const LSH32_IMM: u8 = BPF_ALU32_LOAD | BPF_K | BPF_LSH;
pub const LSH32_REG: u8 = BPF_ALU32_LOAD | BPF_X | BPF_LSH;
pub const RSH32_IMM: u8 = BPF_ALU32_LOAD | BPF_K | BPF_RSH;
pub const RSH32_REG: u8 = BPF_ALU32_LOAD | BPF_X | BPF_RSH;
pub const NEG32: u8 = BPF_ALU32_LOAD | BPF_NEG;
pub const MOD32_IMM: u8 = BPF_ALU32_LOAD | BPF_K | BPF_MOD;
pub const MOD32_REG: u8 = BPF_ALU32_LOAD | BPF_X | BPF_MOD;
pub const XOR32_IMM: u8 = BPF_ALU32_LOAD | BPF_K | BPF_XOR;
pub const XOR32_REG: u8 = BPF_ALU32_LOAD | BPF_X | BPF_XOR;
pub const MOV32_IMM: u8 = BPF_ALU32_LOAD | BPF_K | BPF_MOV;
pub const MOV32_REG: u8 = BPF_ALU32_LOAD | BPF_X | BPF_MOV;
pub const ARSH32_IMM: u8 = BPF_ALU32_LOAD | BPF_K | BPF_ARSH;
pub const ARSH32_REG: u8 = BPF_ALU32_LOAD | BPF_X | BPF_ARSH;

// 64-bit ALU
pub const ADD64_IMM: u8 = BPF_ALU64_STORE | BPF_K | BPF_ADD;
pub const ADD64_REG: u8 = BPF_ALU64_STORE | BPF_X | BPF_ADD;
pub const SUB64_IMM: u8 = BPF_ALU64_STORE | BPF_K | BPF_SUB;
pub const SUB64_REG: u8 = BPF_ALU64_STORE | BPF_X | BPF_SUB;
pub const MUL64_IMM: u8 = BPF_ALU64_STORE | BPF_K | BPF_MUL;
pub const MUL64_REG: u8 = BPF_ALU64_STORE | BPF_X | BPF_MUL;
pub const DIV64_IMM: u8 = BPF_ALU64_STORE | BPF_K | BPF_DIV;
pub const DIV64_REG: u8 = BPF_ALU64_STORE | BPF_X | BPF_DIV;
pub const OR64_IMM: u8 = BPF_ALU64_STORE | BPF_K | BPF_OR;
pub const OR64_REG: u8 = BPF_ALU64_STORE | BPF_X | BPF_OR;
pub const AND64_IMM: u8 = BPF_ALU64_STORE | BPF_K | BPF_AND;
pub const AND64_REG: u8 = BPF_ALU64_STORE | BPF_X | BPF_AND;
pub const LSH64_IMM: u8 = BPF_ALU64_STORE | BPF_K | BPF_LSH;
pub const LSH64_REG: u8 = BPF_ALU64_STORE | BPF_X | BPF_LSH;
pub const RSH64_IMM: u8 = BPF_ALU64_STORE | BPF_K | BPF_RSH;
pub const RSH64_REG: u8 = BPF_ALU64_STORE | BPF_X | BPF_RSH;
pub const NEG64: u8 = BPF_ALU64_STORE | BPF_NEG;
pub const MOD64_IMM: u8 = BPF_ALU64_STORE | BPF_K | BPF_MOD;
pub const MOD64_REG: u8 = BPF_ALU64_STORE | BPF_X | BPF_MOD;
pub const XOR64_IMM: u8 = BPF_ALU64_STORE | BPF_K | BPF_XOR;
pub const XOR64_REG: u8 = BPF_ALU64_STORE | BPF_X | BPF_XOR;
pub const MOV64_IMM: u8 = BPF_ALU64_STORE | BPF_K | BPF_MOV;
pub const MOV64_REG: u8 = BPF_ALU64_STORE | BPF_X | BPF_MOV;
pub const ARSH64_IMM: u8 = BPF_ALU64_STORE | BPF_K | BPF_ARSH;
pub const ARSH64_REG: u8 = BPF_ALU64_STORE | BPF_X | BPF_ARSH;

// Endianness
pub const LE: u8 = BPF_ALU32_LOAD | BPF_K | BPF_END;
pub const BE: u8 = BPF_ALU32_LOAD | BPF_X | BPF_END;

// 64-bit jumps
pub const JA: u8 = BPF_JMP64 | BPF_JA;
pub const JEQ64_IMM: u8 = BPF_JMP64 | BPF_K | BPF_JEQ;
pub const JEQ64_REG: u8 = BPF_JMP64 | BPF_X | BPF_JEQ;
pub const JGT64_IMM: u8 = BPF_JMP64 | BPF_K | BPF_JGT;
pub const JGT64_REG: u8 = BPF_JMP64 | BPF_X | BPF_JGT;
pub const JGE64_IMM: u8 = BPF_JMP64 | BPF_K | BPF_JGE;
pub const JGE64_REG: u8 = BPF_JMP64 | BPF_X | BPF_JGE;
pub const JLT64_IMM: u8 = BPF_JMP64 | BPF_K | BPF_JLT;
pub const JLT64_REG: u8 = BPF_JMP64 | BPF_X | BPF_JLT;
pub const JLE64_IMM: u8 = BPF_JMP64 | BPF_K | BPF_JLE;
pub const JLE64_REG: u8 = BPF_JMP64 | BPF_X | BPF_JLE;
pub const JSET64_IMM: u8 = BPF_JMP64 | BPF_K | BPF_JSET;
pub const JSET64_REG: u8 = BPF_JMP64 | BPF_X | BPF_JSET;
pub const JNE64_IMM: u8 = BPF_JMP64 | BPF_K | BPF_JNE;
pub const JNE64_REG: u8 = BPF_JMP64 | BPF_X | BPF_JNE;
pub const JSGT64_IMM: u8 = BPF_JMP64 | BPF_K | BPF_JSGT;
pub const JSGT64_REG: u8 = BPF_JMP64 | BPF_X | BPF_JSGT;
pub const JSGE64_IMM: u8 = BPF_JMP64 | BPF_K | BPF_JSGE;
pub const JSGE64_REG: u8 = BPF_JMP64 | BPF_X | BPF_JSGE;
pub const JSLT64_IMM: u8 = BPF_JMP64 | BPF_K | BPF_JSLT;
pub const JSLT64_REG: u8 = BPF_JMP64 | BPF_X | BPF_JSLT;
pub const JSLE64_IMM: u8 = BPF_JMP64 | BPF_K | BPF_JSLE;
pub const JSLE64_REG: u8 = BPF_JMP64 | BPF_X | BPF_JSLE;

// Call and exit
pub const CALL_IMM: u8 = BPF_JMP64 | BPF_CALL;
pub const CALL_REG: u8 = BPF_JMP64 | BPF_X | BPF_CALL;
pub const EXIT: u8 = BPF_JMP64 | BPF_EXIT;

// ============================================================================
// Instruction structure
// ============================================================================

/// Raw sBPF instruction (8 bytes)
#[derive(Clone, Default, PartialEq, Eq)]
pub struct RawInsn {
    /// Instruction index (PC)
    pub ptr: usize,
    /// Operation code
    pub opc: u8,
    /// Destination register (0-10)
    pub dst: u8,
    /// Source register (0-10)
    pub src: u8,
    /// Offset (signed 16-bit)
    pub off: i16,
    /// Immediate value (32-bit, sign-extended to 64-bit for LD_DW)
    pub imm: i64,
}

impl fmt::Debug for RawInsn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "RawInsn {{ ptr: 0x{:04x}, opc: 0x{:02x}, dst: r{}, src: r{}, off: {}, imm: 0x{:x} }}",
            self.ptr, self.opc, self.dst, self.src, self.off, self.imm
        )
    }
}

impl RawInsn {
    /// Decode a single instruction from the bytecode at the given PC
    pub fn decode(prog: &[u8], pc: usize) -> Self {
        let base = pc * INSN_SIZE;
        RawInsn {
            ptr: pc,
            opc: prog[base],
            dst: prog[base + 1] & 0x0f,
            src: (prog[base + 1] & 0xf0) >> 4,
            off: LittleEndian::read_i16(&prog[base + 2..]),
            imm: LittleEndian::read_i32(&prog[base + 4..]) as i64,
        }
    }

    /// Merge the second slot of an LD_DW_IMM instruction
    pub fn merge_lddw(&mut self, prog: &[u8]) {
        let next_base = (self.ptr + 1) * INSN_SIZE;
        let high = LittleEndian::read_i32(&prog[next_base + 4..]) as u32;
        self.imm = ((self.imm as u64 & 0xffffffff) | ((high as u64) << 32)) as i64;
    }

    /// Check if this is an LD_DW_IMM instruction (16 bytes)
    pub fn is_lddw(&self) -> bool {
        self.opc == LD_DW_IMM
    }

    /// Check if this is a jump instruction
    pub fn is_jump(&self) -> bool {
        matches!(
            self.opc,
            JA | JEQ64_IMM | JEQ64_REG | JGT64_IMM | JGT64_REG | JGE64_IMM | JGE64_REG
            | JLT64_IMM | JLT64_REG | JLE64_IMM | JLE64_REG | JSET64_IMM | JSET64_REG
            | JNE64_IMM | JNE64_REG | JSGT64_IMM | JSGT64_REG | JSGE64_IMM | JSGE64_REG
            | JSLT64_IMM | JSLT64_REG | JSLE64_IMM | JSLE64_REG
        )
    }

    /// Check if this is a conditional jump
    pub fn is_cond_jump(&self) -> bool {
        self.is_jump() && self.opc != JA
    }

    /// Check if this is an unconditional jump
    pub fn is_uncond_jump(&self) -> bool {
        self.opc == JA
    }

    /// Check if this is a call instruction
    pub fn is_call(&self) -> bool {
        self.opc == CALL_IMM || self.opc == CALL_REG
    }

    /// Check if this is an exit instruction
    pub fn is_exit(&self) -> bool {
        self.opc == EXIT
    }

    /// Check if this is a terminator (jump, call, or exit)
    pub fn is_terminator(&self) -> bool {
        self.is_jump() || self.is_call() || self.is_exit()
    }

    /// Calculate the jump target PC
    pub fn jump_target(&self) -> usize {
        ((self.ptr as isize) + (self.off as isize) + 1) as usize
    }

    /// Get a human-readable mnemonic for the opcode
    pub fn mnemonic(&self) -> &'static str {
        match self.opc {
            LD_DW_IMM => "lddw",
            LD_B_REG => "ldxb",
            LD_H_REG => "ldxh",
            LD_W_REG => "ldxw",
            LD_DW_REG => "ldxdw",
            ST_B_IMM => "stb",
            ST_H_IMM => "sth",
            ST_W_IMM => "stw",
            ST_DW_IMM => "stdw",
            ST_B_REG => "stxb",
            ST_H_REG => "stxh",
            ST_W_REG => "stxw",
            ST_DW_REG => "stxdw",
            ADD32_IMM | ADD32_REG => "add32",
            ADD64_IMM | ADD64_REG => "add64",
            SUB32_IMM | SUB32_REG => "sub32",
            SUB64_IMM | SUB64_REG => "sub64",
            MUL32_IMM | MUL32_REG => "mul32",
            MUL64_IMM | MUL64_REG => "mul64",
            DIV32_IMM | DIV32_REG => "div32",
            DIV64_IMM | DIV64_REG => "div64",
            OR32_IMM | OR32_REG => "or32",
            OR64_IMM | OR64_REG => "or64",
            AND32_IMM | AND32_REG => "and32",
            AND64_IMM | AND64_REG => "and64",
            LSH32_IMM | LSH32_REG => "lsh32",
            LSH64_IMM | LSH64_REG => "lsh64",
            RSH32_IMM | RSH32_REG => "rsh32",
            RSH64_IMM | RSH64_REG => "rsh64",
            NEG32 => "neg32",
            NEG64 => "neg64",
            MOD32_IMM | MOD32_REG => "mod32",
            MOD64_IMM | MOD64_REG => "mod64",
            XOR32_IMM | XOR32_REG => "xor32",
            XOR64_IMM | XOR64_REG => "xor64",
            MOV32_IMM | MOV32_REG => "mov32",
            MOV64_IMM | MOV64_REG => "mov64",
            ARSH32_IMM | ARSH32_REG => "arsh32",
            ARSH64_IMM | ARSH64_REG => "arsh64",
            LE => "le",
            BE => "be",
            JA => "ja",
            JEQ64_IMM | JEQ64_REG => "jeq",
            JGT64_IMM | JGT64_REG => "jgt",
            JGE64_IMM | JGE64_REG => "jge",
            JLT64_IMM | JLT64_REG => "jlt",
            JLE64_IMM | JLE64_REG => "jle",
            JSET64_IMM | JSET64_REG => "jset",
            JNE64_IMM | JNE64_REG => "jne",
            JSGT64_IMM | JSGT64_REG => "jsgt",
            JSGE64_IMM | JSGE64_REG => "jsge",
            JSLT64_IMM | JSLT64_REG => "jslt",
            JSLE64_IMM | JSLE64_REG => "jsle",
            CALL_IMM => "call",
            CALL_REG => "callx",
            EXIT => "exit",
            _ => "unknown",
        }
    }
}

/// Format memory operand with proper offset display
fn format_raw_mem(base: u8, off: i16) -> String {
    if off == 0 {
        format!("[r{}]", base)
    } else if off > 0 {
        format!("[r{}+{}]", base, off)
    } else {
        format!("[r{}{}]", base, off)
    }
}

impl fmt::Display for RawInsn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mnemonic = self.mnemonic();
        match self.opc {
            LD_DW_IMM => write!(f, "{} r{}, 0x{:x}", mnemonic, self.dst, self.imm),
            LD_B_REG | LD_H_REG | LD_W_REG | LD_DW_REG => {
                write!(f, "{} r{}, {}", mnemonic, self.dst, format_raw_mem(self.src, self.off))
            }
            ST_B_IMM | ST_H_IMM | ST_W_IMM | ST_DW_IMM => {
                write!(f, "{} {}, {}", mnemonic, format_raw_mem(self.dst, self.off), self.imm)
            }
            ST_B_REG | ST_H_REG | ST_W_REG | ST_DW_REG => {
                write!(f, "{} {}, r{}", mnemonic, format_raw_mem(self.dst, self.off), self.src)
            }
            JA => write!(f, "{} +{}", mnemonic, self.off),
            EXIT => write!(f, "{}", mnemonic),
            CALL_IMM => write!(f, "{} {}", mnemonic, self.imm),
            CALL_REG => write!(f, "{} r{}", mnemonic, self.dst),
            _ if self.is_cond_jump() => {
                if self.opc & BPF_X != 0 {
                    write!(f, "{} r{}, r{}, +{}", mnemonic, self.dst, self.src, self.off)
                } else {
                    write!(f, "{} r{}, {}, +{}", mnemonic, self.dst, self.imm, self.off)
                }
            }
            _ => {
                if self.opc & BPF_X != 0 {
                    write!(f, "{} r{}, r{}", mnemonic, self.dst, self.src)
                } else {
                    write!(f, "{} r{}, {}", mnemonic, self.dst, self.imm)
                }
            }
        }
    }
}
