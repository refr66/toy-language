//! Emitter module: IR output generation
//!
//! This module handles:
//! - MLIR text generation
//! - DOT (Graphviz) CFG generation
//! - JSON export for mlir-backend
//! - Other output formats

pub mod mlir;
pub mod dot;
pub mod json;

pub use mlir::MlirEmitter;
pub use dot::DotEmitter;
pub use json::{to_json, to_json_compact};
