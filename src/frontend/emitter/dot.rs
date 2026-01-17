//! DOT (Graphviz) CFG generator
//!
//! Generates .dot files for visualizing control flow graphs

use std::fmt::Write;
use crate::ir::{Function, SbpfProgram, SbpfInsn, CallTarget};

/// DOT file emitter for CFG visualization
pub struct DotEmitter;

impl DotEmitter {
    /// Generate DOT representation of a function's CFG
    pub fn emit_function(func: &Function) -> String {
        let mut dot = String::new();
        
        writeln!(dot, "digraph {} {{", Self::escape_name(&func.name)).unwrap();
        writeln!(dot, "    rankdir=TB;").unwrap();
        writeln!(dot, "    node [shape=box, fontname=\"Courier\"];").unwrap();
        writeln!(dot, "    edge [fontname=\"Courier\"];").unwrap();
        writeln!(dot).unwrap();
        
        // Emit nodes (basic blocks)
        for block in func.blocks.values() {
            let label = Self::format_block_label(block);
            writeln!(dot, "    block_{} [label=\"{}\"];", block.id, label).unwrap();
        }
        
        writeln!(dot).unwrap();
        
        // Emit edges
        for block in func.blocks.values() {
            for &succ in &block.successors {
                let edge_label = Self::get_edge_label(block, succ);
                if edge_label.is_empty() {
                    writeln!(dot, "    block_{} -> block_{};", block.id, succ).unwrap();
                } else {
                    writeln!(dot, "    block_{} -> block_{} [label=\"{}\"];", 
                             block.id, succ, edge_label).unwrap();
                }
            }
        }
        
        writeln!(dot, "}}").unwrap();
        dot
    }
    
    /// Generate DOT representation of the entire program
    pub fn emit_program(program: &SbpfProgram) -> String {
        let mut dot = String::new();
        
        writeln!(dot, "digraph program {{").unwrap();
        writeln!(dot, "    rankdir=TB;").unwrap();
        writeln!(dot, "    compound=true;").unwrap();
        writeln!(dot, "    node [shape=box, fontname=\"Courier\"];").unwrap();
        writeln!(dot, "    edge [fontname=\"Courier\"];").unwrap();
        writeln!(dot).unwrap();
        
        for func in program.functions.values() {
            writeln!(dot, "    subgraph cluster_{} {{", Self::escape_name(&func.name)).unwrap();
            writeln!(dot, "        label=\"{}\";", func.name).unwrap();
            writeln!(dot, "        style=dashed;").unwrap();
            
            // Emit nodes
            for block in func.blocks.values() {
                let label = Self::format_block_label(block);
                writeln!(dot, "        {}_{} [label=\"{}\"];", 
                         Self::escape_name(&func.name), block.id, label).unwrap();
            }
            
            // Emit edges
            for block in func.blocks.values() {
                for &succ in &block.successors {
                    let edge_label = Self::get_edge_label(block, succ);
                    if edge_label.is_empty() {
                        writeln!(dot, "        {}_{} -> {}_{};", 
                                 Self::escape_name(&func.name), block.id,
                                 Self::escape_name(&func.name), succ).unwrap();
                    } else {
                        writeln!(dot, "        {}_{} -> {}_{} [label=\"{}\"];", 
                                 Self::escape_name(&func.name), block.id,
                                 Self::escape_name(&func.name), succ,
                                 edge_label).unwrap();
                    }
                }
            }
            
            writeln!(dot, "    }}").unwrap();
            writeln!(dot).unwrap();
        }
        
        writeln!(dot, "}}").unwrap();
        dot
    }
    
    /// Format block label for DOT node
    fn format_block_label(block: &crate::ir::BasicBlock) -> String {
        let mut label = format!("{}\\n", block.label);
        label.push_str("─────────────\\n");
        
        for (pc, insn) in &block.instructions {
            let insn_str = format!("{:04x}: {}\\n", pc, insn);
            // Escape special characters for DOT
            let escaped = insn_str
                .replace('\"', "\\\"")
                .replace('<', "\\<")
                .replace('>', "\\>");
            label.push_str(&escaped);
        }
        
        label
    }
    
    /// Get edge label based on terminator instruction
    fn get_edge_label(block: &crate::ir::BasicBlock, succ: usize) -> String {
        if let Some((_, insn)) = block.instructions.last() {
            match insn {
                SbpfInsn::JumpCond { target, .. } => {
                    if *target == succ {
                        "true".to_string()
                    } else {
                        "false".to_string()
                    }
                }
                SbpfInsn::Call { target: CallTarget::Internal { pc, .. } } => {
                    if *pc == succ {
                        "call".to_string()
                    } else {
                        "return".to_string()
                    }
                }
                _ => String::new(),
            }
        } else {
            String::new()
        }
    }
    
    /// Escape function name for DOT identifier
    fn escape_name(name: &str) -> String {
        name.chars()
            .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BasicBlock, Function};
    
    #[test]
    fn test_emit_simple_function() {
        let mut func = Function::new("test".to_string(), 0);
        let mut block = BasicBlock::new(0);
        block.label = "entry".to_string();
        block.instructions.push((0, SbpfInsn::Exit));
        func.add_block(block);
        
        let dot = DotEmitter::emit_function(&func);
        assert!(dot.contains("digraph test"));
        assert!(dot.contains("block_0"));
    }
}
