//! MLIR Backend for sBPF
//!
//! This crate provides real MLIR IR generation using Melior (MLIR Rust bindings).
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐    JSON    ┌─────────────────┐    MLIR    ┌─────────────┐
//! │  sbpf-frontend  │ ────────> │   sbpf-mlir     │ ────────> │  mlir-opt   │
//! │  (Rust Lifter)  │           │  (Melior API)   │           │  (Passes)   │
//! └─────────────────┘           └─────────────────┘           └─────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sbpf_mlir::{create_context, MlirBuilder};
//!
//! let context = create_context();
//! let mut builder = MlirBuilder::new(&context);
//! let module = builder.build_module(&program)?;
//! println!("{}", module.as_operation());
//! ```

pub mod ir;
pub mod builder;

pub use ir::*;
pub use builder::{MlirBuilder, MlirError, create_context, verify_and_optimize};
