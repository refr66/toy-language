//! ELF parsing and relocation processing
//!
//! This module handles:
//! - ELF file parsing using the elf_parser
//! - .text section location
//! - Relocation table processing
//! - Symbol table extraction

use std::collections::{BTreeMap, HashMap};
//use std::sbpf::ebpf::EBPFInsn;
use std::sbpf::elf_parser::{
    consts::{SHF_EXECINSTR, SHF_ALLOC},
    types::Elf64Rel,
    Elf64, ElfParserError,
};

/// Error types for ELF loading
#[derive(Debug, thiserror::Error)]
pub enum ElfError {
    #[error("ELF parsing error: {0}")]
    ParseError(#[from] ElfParserError),
    
    #[error("No .text section found")]
    NoTextSection,
    
    #[error("Invalid section bounds")]
    InvalidSectionBounds,
}

/// Relocation entry for instruction patching
#[derive(Debug, Clone)]
pub struct Relocation {
    /// Offset in .text section
    pub offset: usize,
    /// Relocation type
    pub r_type: u32,
    /// Symbol index
    pub sym_idx: u32,
    /// Symbol name (if resolved)
    pub symbol_name: Option<String>,
}

/// Information extracted from an ELF file
#[derive(Debug)]
pub struct ElfInfo {
    /// Virtual address of .text section
    pub text_vaddr: u64,
    /// File offset of .text section
    pub text_offset: usize,
    /// Size of .text section
    pub text_size: usize,
    /// Entry point virtual address
    pub entry_vaddr: u64,
    /// Relocations indexed by text offset
    pub relocations: HashMap<usize, Relocation>,
    /// Function symbols (vaddr -> name)
    pub functions: BTreeMap<u64, String>,
}

/// ELF loader: extracts information from sBPF ELF files
pub struct ElfLoader;

impl ElfLoader {
    /// Load and parse an ELF file, extracting all relevant information
    pub fn load(elf_bytes: &[u8]) -> Result<ElfInfo, ElfError> {
        let elf = Elf64::parse(elf_bytes)?;
        
        // Find .text section
        let (text_vaddr, text_offset, text_size) = Self::find_text_section(&elf, elf_bytes)?;
        
        // Get entry point
        let entry_vaddr = elf.file_header().e_entry;
        
        // Load relocations
        let relocations = Self::load_relocations(&elf, text_vaddr);
        
        // Load function symbols
        let functions = Self::load_functions(&elf);
        
        Ok(ElfInfo {
            text_vaddr,
            text_offset,
            text_size,
            entry_vaddr,
            relocations,
            functions,
        })
    }
    
    /// Find the .text section (vaddr, file_offset, size)
    fn find_text_section(
        elf: &Elf64,
        elf_bytes: &[u8],
    ) -> Result<(u64, usize, usize), ElfError> {
        // First try: look for executable, allocated section
        for section_header in elf.section_header_table() {
            if section_header.sh_flags & (SHF_EXECINSTR | SHF_ALLOC) == (SHF_EXECINSTR | SHF_ALLOC) {
                let offset = section_header.sh_offset as usize;
                let size = section_header.sh_size as usize;
                if offset + size <= elf_bytes.len() {
                    return Ok((section_header.sh_addr, offset, size));
                }
            }
        }
        
        // Fallback: look for section named ".text"
        let shstrtab = elf.section_header_table()
            .get(elf.file_header().e_shstrndx as usize);
        
        if let Some(shstrtab) = shstrtab {
            for section_header in elf.section_header_table() {
                if let Ok(name) = Elf64::get_string_in_section(
                    elf_bytes,
                    shstrtab,
                    section_header.sh_name,
                    16,
                ) {
                    if name == b".text" {
                        let offset = section_header.sh_offset as usize;
                        let size = section_header.sh_size as usize;
                        if offset + size <= elf_bytes.len() {
                            return Ok((section_header.sh_addr, offset, size));
                        }
                    }
                }
            }
        }
        
        Err(ElfError::NoTextSection)
    }
    
    /// Load relocations from ELF
    fn load_relocations(elf: &Elf64, text_vaddr: u64) -> HashMap<usize, Relocation> {
        let mut relocations = HashMap::new();
        
        if let Some(relocs) = elf.dynamic_relocations_table() {
            for rel in relocs {
                let offset = rel.r_offset as usize;
                if offset >= text_vaddr as usize {
                    let text_offset = offset - text_vaddr as usize;
                    let symbol_name = Self::get_relocation_symbol_name(elf, rel);
                    
                    relocations.insert(text_offset, Relocation {
                        offset: text_offset,
                        r_type: rel.r_type(),
                        sym_idx: rel.r_sym(),
                        symbol_name,
                    });
                }
            }
        }
        
        relocations
    }
    
    /// Get symbol name for a relocation
    fn get_relocation_symbol_name(elf: &Elf64, rel: &Elf64Rel) -> Option<String> {
        let sym_idx = rel.r_sym() as usize;
        if let Some(symbols) = elf.dynamic_symbol_table() {
            if let Some(sym) = symbols.get(sym_idx) {
                if let Ok(name) = elf.dynamic_symbol_name(sym.st_name) {
                    return Some(String::from_utf8_lossy(name).to_string());
                }
            }
        }
        None
    }
    
    /// Load function symbols from ELF
    fn load_functions(elf: &Elf64) -> BTreeMap<u64, String> {
        let mut functions = BTreeMap::new();
        
        if let Ok(Some(symbols)) = elf.symbol_table() {
            for sym in symbols {
                if sym.is_function() && sym.st_value != 0 {
                    if let Ok(name) = elf.symbol_name(sym.st_name) {
                        let name_str = String::from_utf8_lossy(name).to_string();
                        functions.insert(sym.st_value, name_str);
                    }
                }
            }
        }
        
        functions
    }
}
