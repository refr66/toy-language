//! Instruction coverage statistics
//!
//! Tracks which sBPF opcodes are supported by the lifter

use std::collections::HashSet;
use crate::ebpf::*;

/// Total number of sBPF opcodes (approximate)
pub const TOTAL_OPCODES: usize = 102;

/// Instruction coverage statistics
pub struct CoverageStats {
    /// Set of supported opcodes
    supported: HashSet<u8>,
    /// Set of encountered but unsupported opcodes
    unsupported: HashSet<u8>,
}

impl CoverageStats {
    /// Create a new coverage tracker with all supported opcodes
    pub fn new() -> Self {
        let mut supported = HashSet::new();
        
        // Load immediate
        supported.insert(LD_DW_IMM);
        
        // Load from memory
        supported.insert(LD_B_REG);
        supported.insert(LD_H_REG);
        supported.insert(LD_W_REG);
        supported.insert(LD_DW_REG);
        
        // Store immediate
        supported.insert(ST_B_IMM);
        supported.insert(ST_H_IMM);
        supported.insert(ST_W_IMM);
        supported.insert(ST_DW_IMM);
        
        // Store register
        supported.insert(ST_B_REG);
        supported.insert(ST_H_REG);
        supported.insert(ST_W_REG);
        supported.insert(ST_DW_REG);
        
        // 32-bit ALU immediate
        supported.insert(ADD32_IMM);
        supported.insert(SUB32_IMM);
        supported.insert(MUL32_IMM);
        supported.insert(DIV32_IMM);
        supported.insert(OR32_IMM);
        supported.insert(AND32_IMM);
        supported.insert(LSH32_IMM);
        supported.insert(RSH32_IMM);
        supported.insert(NEG32);
        supported.insert(MOD32_IMM);
        supported.insert(XOR32_IMM);
        supported.insert(MOV32_IMM);
        supported.insert(ARSH32_IMM);
        
        // 32-bit ALU register
        supported.insert(ADD32_REG);
        supported.insert(SUB32_REG);
        supported.insert(MUL32_REG);
        supported.insert(DIV32_REG);
        supported.insert(OR32_REG);
        supported.insert(AND32_REG);
        supported.insert(LSH32_REG);
        supported.insert(RSH32_REG);
        supported.insert(MOD32_REG);
        supported.insert(XOR32_REG);
        supported.insert(MOV32_REG);
        supported.insert(ARSH32_REG);
        
        // 64-bit ALU immediate
        supported.insert(ADD64_IMM);
        supported.insert(SUB64_IMM);
        supported.insert(MUL64_IMM);
        supported.insert(DIV64_IMM);
        supported.insert(OR64_IMM);
        supported.insert(AND64_IMM);
        supported.insert(LSH64_IMM);
        supported.insert(RSH64_IMM);
        supported.insert(NEG64);
        supported.insert(MOD64_IMM);
        supported.insert(XOR64_IMM);
        supported.insert(MOV64_IMM);
        supported.insert(ARSH64_IMM);
        
        // 64-bit ALU register
        supported.insert(ADD64_REG);
        supported.insert(SUB64_REG);
        supported.insert(MUL64_REG);
        supported.insert(DIV64_REG);
        supported.insert(OR64_REG);
        supported.insert(AND64_REG);
        supported.insert(LSH64_REG);
        supported.insert(RSH64_REG);
        supported.insert(MOD64_REG);
        supported.insert(XOR64_REG);
        supported.insert(MOV64_REG);
        supported.insert(ARSH64_REG);
        
        // Endianness
        supported.insert(LE);
        supported.insert(BE);
        
        // Jumps
        supported.insert(JA);
        supported.insert(JEQ64_IMM);
        supported.insert(JEQ64_REG);
        supported.insert(JGT64_IMM);
        supported.insert(JGT64_REG);
        supported.insert(JGE64_IMM);
        supported.insert(JGE64_REG);
        supported.insert(JLT64_IMM);
        supported.insert(JLT64_REG);
        supported.insert(JLE64_IMM);
        supported.insert(JLE64_REG);
        supported.insert(JSET64_IMM);
        supported.insert(JSET64_REG);
        supported.insert(JNE64_IMM);
        supported.insert(JNE64_REG);
        supported.insert(JSGT64_IMM);
        supported.insert(JSGT64_REG);
        supported.insert(JSGE64_IMM);
        supported.insert(JSGE64_REG);
        supported.insert(JSLT64_IMM);
        supported.insert(JSLT64_REG);
        supported.insert(JSLE64_IMM);
        supported.insert(JSLE64_REG);
        
        // Calls
        supported.insert(CALL_IMM);
        supported.insert(CALL_REG);
        
        // Exit
        supported.insert(EXIT);
        
        Self {
            supported,
            unsupported: HashSet::new(),
        }
    }
    
    /// Check if an opcode is supported
    pub fn is_supported(&self, opcode: u8) -> bool {
        self.supported.contains(&opcode)
    }
    
    /// Record an encountered opcode
    pub fn record(&mut self, opcode: u8) {
        if !self.supported.contains(&opcode) {
            self.unsupported.insert(opcode);
        }
    }
    
    /// Get number of supported opcodes
    pub fn supported_count(&self) -> usize {
        self.supported.len()
    }
    
    /// Get total opcode count
    pub fn total_count(&self) -> usize {
        TOTAL_OPCODES
    }
    
    /// Get coverage percentage
    pub fn coverage_percent(&self) -> f64 {
        (self.supported.len() as f64 / TOTAL_OPCODES as f64) * 100.0
    }
    
    /// Get list of unsupported opcodes encountered
    pub fn unsupported_encountered(&self) -> Vec<u8> {
        self.unsupported.iter().cloned().collect()
    }
    
    /// Format coverage summary
    pub fn summary(&self) -> String {
        format!(
            "Instruction Coverage: {}/{} opcodes supported ({:.1}%)",
            self.supported.len(),
            TOTAL_OPCODES,
            self.coverage_percent()
        )
    }
    
    /// Detailed report of supported opcodes by category
    pub fn detailed_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== Instruction Coverage Report ===\n\n");
        
        // Memory operations
        let mem_ops = [LD_DW_IMM, LD_B_REG, LD_H_REG, LD_W_REG, LD_DW_REG,
                       ST_B_IMM, ST_H_IMM, ST_W_IMM, ST_DW_IMM,
                       ST_B_REG, ST_H_REG, ST_W_REG, ST_DW_REG];
        let mem_supported: usize = mem_ops.iter().filter(|&&op| self.supported.contains(&op)).count();
        report.push_str(&format!("Memory Operations: {}/{}\n", mem_supported, mem_ops.len()));
        
        // ALU 32-bit
        let alu32_ops = [ADD32_IMM, ADD32_REG, SUB32_IMM, SUB32_REG,
                         MUL32_IMM, MUL32_REG, DIV32_IMM, DIV32_REG,
                         OR32_IMM, OR32_REG, AND32_IMM, AND32_REG,
                         LSH32_IMM, LSH32_REG, RSH32_IMM, RSH32_REG,
                         NEG32, MOD32_IMM, MOD32_REG,
                         XOR32_IMM, XOR32_REG, MOV32_IMM, MOV32_REG,
                         ARSH32_IMM, ARSH32_REG];
        let alu32_supported: usize = alu32_ops.iter().filter(|&&op| self.supported.contains(&op)).count();
        report.push_str(&format!("32-bit ALU: {}/{}\n", alu32_supported, alu32_ops.len()));
        
        // ALU 64-bit
        let alu64_ops = [ADD64_IMM, ADD64_REG, SUB64_IMM, SUB64_REG,
                         MUL64_IMM, MUL64_REG, DIV64_IMM, DIV64_REG,
                         OR64_IMM, OR64_REG, AND64_IMM, AND64_REG,
                         LSH64_IMM, LSH64_REG, RSH64_IMM, RSH64_REG,
                         NEG64, MOD64_IMM, MOD64_REG,
                         XOR64_IMM, XOR64_REG, MOV64_IMM, MOV64_REG,
                         ARSH64_IMM, ARSH64_REG];
        let alu64_supported: usize = alu64_ops.iter().filter(|&&op| self.supported.contains(&op)).count();
        report.push_str(&format!("64-bit ALU: {}/{}\n", alu64_supported, alu64_ops.len()));
        
        // Jump operations
        let jmp_ops = [JA, JEQ64_IMM, JEQ64_REG, JGT64_IMM, JGT64_REG,
                       JGE64_IMM, JGE64_REG, JLT64_IMM, JLT64_REG,
                       JLE64_IMM, JLE64_REG, JSET64_IMM, JSET64_REG,
                       JNE64_IMM, JNE64_REG, JSGT64_IMM, JSGT64_REG,
                       JSGE64_IMM, JSGE64_REG, JSLT64_IMM, JSLT64_REG,
                       JSLE64_IMM, JSLE64_REG];
        let jmp_supported: usize = jmp_ops.iter().filter(|&&op| self.supported.contains(&op)).count();
        report.push_str(&format!("Jump Operations: {}/{}\n", jmp_supported, jmp_ops.len()));
        
        // Control flow
        let ctrl_ops = [CALL_IMM, CALL_REG, EXIT];
        let ctrl_supported: usize = ctrl_ops.iter().filter(|&&op| self.supported.contains(&op)).count();
        report.push_str(&format!("Control Flow: {}/{}\n", ctrl_supported, ctrl_ops.len()));
        
        // Endianness
        let end_ops = [LE, BE];
        let end_supported: usize = end_ops.iter().filter(|&&op| self.supported.contains(&op)).count();
        report.push_str(&format!("Endianness: {}/{}\n", end_supported, end_ops.len()));
        
        report.push_str(&format!("\nTotal: {}/{} ({:.1}%)\n", 
                                  self.supported.len(), 
                                  TOTAL_OPCODES,
                                  self.coverage_percent()));
        
        if !self.unsupported.is_empty() {
            report.push_str("\nUnsupported opcodes encountered:\n");
            for op in &self.unsupported {
                report.push_str(&format!("  0x{:02x}\n", op));
            }
        }
        
        report
    }
}

impl Default for CoverageStats {
    fn default() -> Self {
        Self::new()
    }
}
