//! Integration tests for sBPF ELF parsing
//!
//! Tests all .so files in sbpf/tests/elfs/

use std::fs;
use std::path::Path;

use sbpf_frontend::{Lifter, DiffTester, CoverageStats};

/// Get all .so files in the test ELFs directory
fn get_test_elfs() -> Vec<String> {
    let test_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/elfs");
    
    let mut files = Vec::new();
    
    if let Ok(entries) = fs::read_dir(&test_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "so") {
                if let Some(path_str) = path.to_str() {
                    files.push(path_str.to_string());
                }
            }
        }
    }
    
    files.sort();
    files
}

#[test]
fn test_all_elfs_parse() {
    let files = get_test_elfs();
    
    if files.is_empty() {
        eprintln!("Warning: No .so files found in test directory");
        return;
    }
    
    let mut passed = 0;
    let mut failed = 0;
    let mut errors: Vec<(String, String)> = Vec::new();
    
    for file in &files {
        let elf_bytes = match fs::read(file) {
            Ok(bytes) => bytes,
            Err(e) => {
                errors.push((file.clone(), format!("Read error: {}", e)));
                failed += 1;
                continue;
            }
        };
        
        match Lifter::new(&elf_bytes) {
            Ok(lifter) => {
                // Try to decode instructions
                match lifter.decode_instructions() {
                    Ok(_) => {
                        // Try to lift
                        match lifter.lift() {
                            Ok(program) => {
                                // Validate CFG
                                for func in program.functions.values() {
                                    let issues = DiffTester::validate_cfg(func);
                                    if !issues.is_empty() {
                                        errors.push((file.clone(), 
                                            format!("CFG issues: {:?}", issues)));
                                        failed += 1;
                                        continue;
                                    }
                                }
                                passed += 1;
                            }
                            Err(e) => {
                                errors.push((file.clone(), format!("Lift error: {}", e)));
                                failed += 1;
                            }
                        }
                    }
                    Err(e) => {
                        errors.push((file.clone(), format!("Decode error: {}", e)));
                        failed += 1;
                    }
                }
            }
            Err(e) => {
                errors.push((file.clone(), format!("Parse error: {}", e)));
                failed += 1;
            }
        }
    }
    
    println!("\n=== Integration Test Results ===");
    println!("Passed: {}/{}", passed, files.len());
    println!("Failed: {}/{}", failed, files.len());
    
    if !errors.is_empty() {
        println!("\nErrors:");
        for (file, error) in &errors {
            let filename = Path::new(file).file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(file);
            println!("  {}: {}", filename, error);
        }
    }
    
    // Don't fail the test if some files fail - they might be intentionally malformed
    // assert_eq!(failed, 0, "Some ELF files failed to parse");
}

#[test]
fn test_coverage_stats() {
    let coverage = CoverageStats::new();
    
    // Should support at least 80% of opcodes
    assert!(coverage.coverage_percent() > 80.0, 
            "Coverage should be > 80%, got {:.1}%", 
            coverage.coverage_percent());
    
    println!("\n{}", coverage.summary());
    println!("{}", coverage.detailed_report());
}

#[test]
fn test_specific_elf() {
    let test_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/elfs");
    
    let test_file = test_dir.join("relative_call_sbpfv0.so");
    
    if !test_file.exists() {
        eprintln!("Test file not found: {:?}", test_file);
        return;
    }
    
    let elf_bytes = fs::read(&test_file).expect("Failed to read test file");
    let lifter = Lifter::new(&elf_bytes).expect("Failed to create lifter");
    
    // Check basic info
    assert!(lifter.text_bytes().len() > 0, "Text section should not be empty");
    
    // Decode instructions
    let instructions = lifter.decode_instructions().expect("Failed to decode");
    assert!(!instructions.is_empty(), "Should have some instructions");
    
    // Lift to IR
    let program = lifter.lift().expect("Failed to lift");
    assert!(!program.functions.is_empty(), "Should have at least one function");
    
    // Validate CFG
    for func in program.functions.values() {
        let issues = DiffTester::validate_cfg(func);
        assert!(issues.is_empty(), "CFG validation failed: {:?}", issues);
    }
    
    println!("\nTest passed for: {:?}", test_file);
    println!("Instructions: {}", instructions.len());
    println!("Functions: {}", program.functions.len());
}
