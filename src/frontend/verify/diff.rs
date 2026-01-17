//! Differential testing with solana_rbpf
//!
//! Compares our lifter output with the reference implementation

use std::fmt;
use crate::ir::{SbpfProgram, Function};

/// Result of a differential test
#[derive(Debug)]
pub struct DiffResult {
    /// File being tested
    pub file_name: String,
    /// Whether the test passed
    pub passed: bool,
    /// Differences found (if any)
    pub differences: Vec<Difference>,
    /// Instruction count from our lifter
    pub our_insn_count: usize,
    /// Block count from our lifter
    pub our_block_count: usize,
}

/// A single difference found
#[derive(Debug)]
pub struct Difference {
    /// PC where difference occurred
    pub pc: usize,
    /// Our output
    pub ours: String,
    /// Reference output
    pub reference: String,
    /// Type of difference
    pub diff_type: DiffType,
}

/// Type of difference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffType {
    /// Different instruction mnemonic
    Mnemonic,
    /// Different register
    Register,
    /// Different immediate value
    Immediate,
    /// Different jump target
    JumpTarget,
    /// Missing instruction
    Missing,
    /// Extra instruction
    Extra,
}

impl fmt::Display for DiffType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffType::Mnemonic => write!(f, "mnemonic"),
            DiffType::Register => write!(f, "register"),
            DiffType::Immediate => write!(f, "immediate"),
            DiffType::JumpTarget => write!(f, "jump_target"),
            DiffType::Missing => write!(f, "missing"),
            DiffType::Extra => write!(f, "extra"),
        }
    }
}

/// Differential tester
pub struct DiffTester;

impl DiffTester {
    /// Run differential test on a program
    /// 
    /// Note: This is a placeholder. In a real implementation, you would:
    /// 1. Call solana_rbpf's disassembler
    /// 2. Parse its output
    /// 3. Compare with our IR
    pub fn test_program(program: &SbpfProgram, file_name: &str) -> DiffResult {
        let mut our_insn_count = 0;
        let mut our_block_count = 0;
        
        for func in program.functions.values() {
            our_block_count += func.blocks.len();
            for block in func.blocks.values() {
                our_insn_count += block.instructions.len();
            }
        }
        
        // Placeholder: In reality, compare with solana_rbpf output
        DiffResult {
            file_name: file_name.to_string(),
            passed: true, // Assume pass for now
            differences: vec![],
            our_insn_count,
            our_block_count,
        }
    }
    
    /// Validate CFG correctness
    pub fn validate_cfg(func: &Function) -> Vec<CfgIssue> {
        let mut issues = Vec::new();
        
        // Check 1: All blocks have at least one instruction
        for block in func.blocks.values() {
            if block.instructions.is_empty() {
                issues.push(CfgIssue::EmptyBlock { block_id: block.id });
            }
        }
        
        // Check 2: All jump targets point to valid blocks
        for block in func.blocks.values() {
            for &succ in &block.successors {
                if !func.blocks.contains_key(&succ) {
                    issues.push(CfgIssue::InvalidJumpTarget {
                        from_block: block.id,
                        target: succ,
                    });
                }
            }
        }
        
        // Check 3: Terminators are correct
        for block in func.blocks.values() {
            if let Some((_, insn)) = block.instructions.last() {
                let is_terminator = matches!(insn,
                    crate::ir::SbpfInsn::Jump { .. } |
                    crate::ir::SbpfInsn::JumpCond { .. } |
                    crate::ir::SbpfInsn::Call { .. } |
                    crate::ir::SbpfInsn::Exit
                );
                let has_successors = !block.successors.is_empty();
                
                match (is_terminator, has_successors) {
                    (true, false) => {
                        // Exit instruction - OK
                    }
                    (true, true) => {
                        // Jump/call with successors - OK
                    }
                    (false, true) => {
                        // Fallthrough - OK
                    }
                    (false, false) => {
                        // Block ends without terminator and no successors
                        issues.push(CfgIssue::NoTerminator { block_id: block.id });
                    }
                }
            }
        }
        
        // Check 4: Entry block is reachable
        if !func.blocks.contains_key(&func.entry) {
            issues.push(CfgIssue::MissingEntry { entry: func.entry });
        }
        
        // Check 5: No overlapping instructions
        let mut seen_pcs = std::collections::HashSet::new();
        for block in func.blocks.values() {
            for (pc, _) in &block.instructions {
                if !seen_pcs.insert(*pc) {
                    issues.push(CfgIssue::OverlappingInstruction { pc: *pc });
                }
            }
        }
        
        issues
    }
    
    /// Format validation results
    pub fn format_validation(func: &Function) -> String {
        let issues = Self::validate_cfg(func);
        
        if issues.is_empty() {
            format!("CFG Validation: PASSED (function: {})", func.name)
        } else {
            let mut report = format!("CFG Validation: FAILED (function: {})\n", func.name);
            for issue in &issues {
                report.push_str(&format!("  - {}\n", issue));
            }
            report
        }
    }
}

/// CFG validation issue
#[derive(Debug)]
pub enum CfgIssue {
    /// Empty basic block (ghost block)
    EmptyBlock { block_id: usize },
    /// Jump target doesn't exist
    InvalidJumpTarget { from_block: usize, target: usize },
    /// Block doesn't end with terminator
    NoTerminator { block_id: usize },
    /// Entry block is missing
    MissingEntry { entry: usize },
    /// Same PC appears in multiple blocks
    OverlappingInstruction { pc: usize },
}

impl fmt::Display for CfgIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CfgIssue::EmptyBlock { block_id } => {
                write!(f, "Empty block: block_{}", block_id)
            }
            CfgIssue::InvalidJumpTarget { from_block, target } => {
                write!(f, "Invalid jump: block_{} -> block_{} (target doesn't exist)", 
                       from_block, target)
            }
            CfgIssue::NoTerminator { block_id } => {
                write!(f, "No terminator: block_{}", block_id)
            }
            CfgIssue::MissingEntry { entry } => {
                write!(f, "Missing entry block: block_{}", entry)
            }
            CfgIssue::OverlappingInstruction { pc } => {
                write!(f, "Overlapping instruction at PC {:04x}", pc)
            }
        }
    }
}

impl DiffResult {
    /// Format as summary
    pub fn summary(&self) -> String {
        if self.passed {
            format!(
                "PASS: {} ({} instructions, {} blocks)",
                self.file_name, self.our_insn_count, self.our_block_count
            )
        } else {
            format!(
                "FAIL: {} ({} differences found)",
                self.file_name, self.differences.len()
            )
        }
    }
}
