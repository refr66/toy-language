//! MLIR Generator CLI
//!
//! Reads JSON IR from frontend and generates MLIR using Melior

use clap::Parser;
use sbpf_mlir::{Program, MlirBuilder, create_context};
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "sbpf-mlir-gen")]
#[command(about = "Generate MLIR from sBPF IR JSON")]
struct Args {
    /// Input JSON file from sbpf-frontend
    #[arg(short, long)]
    input: PathBuf,

    /// Output MLIR file (stdout if not specified)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Run verification passes
    #[arg(long)]
    verify: bool,

    /// Run optimization passes
    #[arg(long)]
    optimize: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Read JSON input
    let json_content = fs::read_to_string(&args.input)?;
    let program: Program = serde_json::from_str(&json_content)?;

    println!("// Loaded program with {} functions", program.functions.len());
    println!("// Entrypoint: {}", program.entrypoint);

    // Create MLIR context
    let context = create_context();
    let mut builder = MlirBuilder::new(&context);

    // Build MLIR module
    let module = builder.build_module(&program)?;

    // Optionally verify
    if args.verify {
        sbpf_mlir::verify_and_optimize(&module)?;
        println!("// Verification passed");
    }

    // Output MLIR
    let mlir_output = format!("{}", module.as_operation());
    
    if let Some(output_path) = args.output {
        fs::write(&output_path, &mlir_output)?;
        println!("// Written to {:?}", output_path);
    } else {
        println!("{}", mlir_output);
    }

    Ok(())
}
