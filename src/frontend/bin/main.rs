//! sbpf-lift: Command-line tool for lifting sBPF ELF to IR
//!
//! Usage: sbpf-lift [OPTIONS] <elf_file>
//!
//! Options:
//!   --mlir      Output MLIR representation
//!   --dot       Output DOT (Graphviz) CFG
//!   --json      Output JSON for mlir-backend
//!   --coverage  Show instruction coverage statistics
//!   --validate  Validate CFG correctness

use std::env;
use std::fs;
use std::process;

use sbpf_frontend::{Lifter, MlirEmitter, DotEmitter, CoverageStats, DiffTester};
use sbpf_frontend::emitter::to_json;

fn print_usage(program: &str) {
    eprintln!("Usage: {} [OPTIONS] <elf_file>", program);
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --mlir      Output MLIR text representation");
    eprintln!("  --dot       Output DOT (Graphviz) CFG");
    eprintln!("  --json      Output JSON for mlir-backend (Melior)");
    eprintln!("  --coverage  Show instruction coverage statistics");
    eprintln!("  --validate  Validate CFG correctness");
    eprintln!();
    eprintln!("Example:");
    eprintln!("  {} relative_call_sbpfv0.so", program);
    eprintln!("  {} --mlir --dot program.so", program);
    eprintln!("  {} --json program.so > program.json", program);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_usage(&args[0]);
        process::exit(1);
    }
    
    // Parse options
    let mut emit_mlir = false;
    let mut emit_dot = false;
    let mut emit_json = false;
    let mut show_coverage = false;
    let mut validate_cfg = false;
    let mut elf_path: Option<&str> = None;
    
    for arg in &args[1..] {
        match arg.as_str() {
            "--mlir" => emit_mlir = true,
            "--dot" => emit_dot = true,
            "--json" => emit_json = true,
            "--coverage" => show_coverage = true,
            "--validate" => validate_cfg = true,
            "--help" | "-h" => {
                print_usage(&args[0]);
                process::exit(0);
            }
            _ if !arg.starts_with("--") => {
                elf_path = Some(arg);
            }
            _ => {
                eprintln!("Unknown option: {}", arg);
                print_usage(&args[0]);
                process::exit(1);
            }
        }
    }
    
    let elf_path = match elf_path {
        Some(p) => p,
        None => {
            eprintln!("Error: No ELF file specified");
            print_usage(&args[0]);
            process::exit(1);
        }
    };
    
    // Read ELF file
    let elf_bytes = match fs::read(elf_path) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", elf_path, e);
            process::exit(1);
        }
    };
    
    // Create lifter
    let lifter = match Lifter::new(&elf_bytes) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Error parsing ELF: {}", e);
            process::exit(1);
        }
    };
    
    // Check if we're in "pipe-friendly" mode (only output MLIR or JSON)
    let pipe_mode = (emit_mlir || emit_json) && !emit_dot && !show_coverage && !validate_cfg;
    
    if !pipe_mode {
        println!("=== sBPF ELF Analysis ===");
        println!("File: {}", elf_path);
        println!("Text section size: {} bytes", lifter.text_bytes().len());
        println!("Entrypoint PC: {}", lifter.entrypoint_pc());
        println!();
        
        // Show coverage statistics
        if show_coverage {
            let coverage = CoverageStats::new();
            println!("{}", coverage.summary());
            println!();
        }
        
        // Decode and print raw instructions
        println!("=== Raw Instructions (with LD_DW merged) ===");
        match lifter.decode_instructions() {
            Ok(instructions) => {
                for (pc, insn) in &instructions {
                    println!("{:04x}: {}", pc, insn);
                }
            }
            Err(e) => {
                eprintln!("Error decoding instructions: {}", e);
                process::exit(1);
            }
        }
        println!();
    }
    
    // Lift to IR
    if !pipe_mode {
        println!("=== Lifted IR ===");
    }
    match lifter.lift() {
        Ok(program) => {
            if !pipe_mode {
                println!("{}", program);
                
                // Static Cost Analysis
                println!("=== Static Cost Analysis ===");
                for func in program.functions.values() {
                    println!("Function: {}", func.name);
                    for block in func.blocks.values() {
                        println!("  {}: {} instructions, Static CU Cost: {}",
                            block.label,
                            block.instruction_count(),
                            block.cu_cost()
                        );
                    }
                    println!("  Total: {} instructions, Static CU Cost: {}",
                        func.instruction_count(),
                        func.cu_cost()
                    );
                    println!();
                }
                
                // CFG Validation
                if validate_cfg {
                    println!("=== CFG Validation ===");
                    for func in program.functions.values() {
                        println!("{}", DiffTester::format_validation(func));
                    }
                    println!();
                }
            }
            
            // MLIR output
            if emit_mlir {
                if !pipe_mode {
                    println!("=== MLIR Output ===");
                }
                let mlir = MlirEmitter::emit_program(&program);
                print!("{}", mlir);
            }
            
            // DOT output
            if emit_dot {
                println!("=== DOT (Graphviz) Output ===");
                let dot = DotEmitter::emit_program(&program);
                println!("{}", dot);
                
                // Also save to file
                let dot_path = format!("{}.dot", elf_path);
                if fs::write(&dot_path, &dot).is_ok() {
                    println!("// DOT file saved to: {}", dot_path);
                    println!("// Generate image with: dot -Tpng {} -o {}.png", dot_path, elf_path);
                }
            }
            
            // JSON output (for mlir-backend)
            if emit_json {
                match to_json(&program) {
                    Ok(json) => print!("{}", json),
                    Err(e) => eprintln!("Error generating JSON: {}", e),
                }
            }
        }
        Err(e) => {
            eprintln!("Error lifting to IR: {}", e);
            process::exit(1);
        }
    }
}
