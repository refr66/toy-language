//! sbpf-frontend: sBPF ELF parser and lifter
//!
//! This crate provides tools for parsing sBPF (Solana BPF) ELF files
//! and converting them to a structured IR suitable for MLIR translation.
//!
//! ## Modules
//!
//! - `lifter`: ELF parsing and IR lifting
//! - `ir`: Intermediate representation types
//! - `emitter`: Output generation (MLIR, DOT)
//! - `verify`: Verification and coverage statistics
//! - `ebpf`: sBPF instruction definitions
//! - `elf_parser`: Low-level ELF parsing

pub mod elf_parser;
pub mod ebpf;
pub mod lifter;
pub mod ir;
pub mod emitter;
pub mod verify;

// Re-exports for convenience
pub use lifter::Lifter;
pub use ir::{SbpfProgram, Function, BasicBlock, SbpfInsn};
pub use emitter::{MlirEmitter, DotEmitter};
pub use verify::{CoverageStats, DiffTester};

/// Arithmetic overflow error
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("arithmetic overflow")]
pub struct ArithmeticOverflow;

/// Trait for checked arithmetic operations
pub trait ErrCheckedArithmetic: Sized {
    fn err_checked_add(self, rhs: Self) -> Result<Self, ArithmeticOverflow>;
    fn err_checked_sub(self, rhs: Self) -> Result<Self, ArithmeticOverflow>;
    fn err_checked_mul(self, rhs: Self) -> Result<Self, ArithmeticOverflow>;
}

macro_rules! impl_err_checked_arithmetic {
    ($($t:ty),*) => {
        $(
            impl ErrCheckedArithmetic for $t {
                fn err_checked_add(self, rhs: Self) -> Result<Self, ArithmeticOverflow> {
                    self.checked_add(rhs).ok_or(ArithmeticOverflow)
                }
                fn err_checked_sub(self, rhs: Self) -> Result<Self, ArithmeticOverflow> {
                    self.checked_sub(rhs).ok_or(ArithmeticOverflow)
                }
                fn err_checked_mul(self, rhs: Self) -> Result<Self, ArithmeticOverflow> {
                    self.checked_mul(rhs).ok_or(ArithmeticOverflow)
                }
            }
        )*
    };
}

impl_err_checked_arithmetic!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);
