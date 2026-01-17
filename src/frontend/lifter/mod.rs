//! Lifter module: Binary to IR transformation
//!
//! This module handles:
//! - ELF parsing and relocation processing
//! - Bytecode scanning and instruction decoding
//! - LD_DW merging (16-byte instructions)
//! - Basic block splitting

pub mod elf;
pub mod parser;

pub use elf::ElfLoader;
pub use parser::Lifter;
