//! Intermediate Representation module
//!
//! This module defines the IR data structures:
//! - SbpfInsn: High-level instruction representation
//! - BasicBlock: A sequence of instructions with single entry/exit
//! - Function: A collection of basic blocks
//! - SbpfProgram: The complete program

pub mod types;

pub use types::*;
