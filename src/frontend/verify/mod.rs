//! Verification module
//!
//! This module provides:
//! - Instruction coverage statistics
//! - Differential testing with solana_rbpf
//! - CFG validation

pub mod coverage;
pub mod diff;

pub use coverage::CoverageStats;
pub use diff::DiffTester;
