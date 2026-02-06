//! Witness Serialization
//!
//! JSON export/import for the trust artifact. No external deps.

use crate::lowering::target::{MicroOp, SimdWidth};
use crate::lowering::witness::Witness;
use crate::semantic_hash::SemanticHash;

pub struct SerializedWitness {
    pub version: &'static str,
    pub phase_a_hash: [u8; 32],
    pub phase_b_hash: [u8; 32],
    pub vector_width: u16,
    pub code_size: u16,
    pub microop_count: u16,
    pub regalloc_fingerprint: [u8; 8],
    pub entries: Vec<SerializedEntry>,
    pub microops: Vec<SerializedMicroOp>,
}

#[derive(Clone, Debug)]
pub struct SerializedEntry {
    pub rif_node: u32,
    pub microop_idx: u16,
    pub mask_before: u16,
    pub mask_after: u16,
    pub microop_tag: u8,
}

#[derive(Clone, Debug)]
pub struct SerializedMicroOp {
    pub op_type: String,
    pub details: String,
}

impl SerializedWitness {
    pub fn from_kernel(
        witness: &Witness,
        ops: &[MicroOp],
        phase_a_hash: &SemanticHash,
        phase_b_hash: &SemanticHash,
        vector_width: SimdWidth,
        code_size: usize,
    ) -> Self {
        let entries: Vec<SerializedEntry> = witness.as_slice()
            .iter()
            .map(|e| SerializedEntry {
                rif_node: e.rif_node.0,
                microop_idx: e.microop_idx,
                mask_before: e.mask_before.0,
                mask_after: e.mask_after.0,
                microop_tag: e.microop_tag,
            })
            .collect();

        let microops: Vec<SerializedMicroOp> = ops.iter()
            .map(|op| SerializedMicroOp {
                op_type: op.type_name().to_string(),
                details: op.details_string(),
            })
            .collect();

        let mut regalloc_fingerprint = [0u8; 8];
        let mut hasher = crate::semantic_hash::Hasher::new();
        hasher.update(b"REGALLOC_V0");
        for op in ops {
            hasher.update(&[op.discriminant()]);
            if let Some((dst, src1, src2)) = op.registers() {
                hasher.update(&[dst, src1.unwrap_or(0), src2.unwrap_or(0)]);
            }
        }
        let hash = hasher.finalize();
        regalloc_fingerprint.copy_from_slice(&hash.as_bytes()[0..8]);

        Self {
            version: "1.0",
            phase_a_hash: *phase_a_hash.as_bytes(),
            phase_b_hash: *phase_b_hash.as_bytes(),
            vector_width: vector_width.bits(),
            code_size: code_size as u16,
            microop_count: ops.len() as u16,
            regalloc_fingerprint,
            entries,
            microops,
        }
    }

    pub fn to_json(&self) -> String {
        let mut json = String::with_capacity(4096);
        
        json.push_str("{\n");
        json.push_str(&format!("  \"version\": \"{}\",\n", self.version));
        json.push_str(&format!("  \"phase_a_hash\": \"{}\",\n", hex_encode(&self.phase_a_hash)));
        json.push_str(&format!("  \"phase_b_hash\": \"{}\",\n", hex_encode(&self.phase_b_hash)));
        json.push_str(&format!("  \"vector_width\": {},\n", self.vector_width));
        json.push_str(&format!("  \"code_size\": {},\n", self.code_size));
        json.push_str(&format!("  \"microop_count\": {},\n", self.microop_count));
        json.push_str(&format!("  \"regalloc_fingerprint\": \"{}\",\n", hex_encode(&self.regalloc_fingerprint)));
        
        json.push_str("  \"entries\": [\n");
        for (i, entry) in self.entries.iter().enumerate() {
            json.push_str("    {\n");
            json.push_str(&format!("      \"rif_node\": {},\n", entry.rif_node));
            json.push_str(&format!("      \"microop_idx\": {},\n", entry.microop_idx));
            json.push_str(&format!("      \"mask_before\": {},\n", entry.mask_before));
            json.push_str(&format!("      \"mask_after\": {},\n", entry.mask_after));
            json.push_str(&format!("      \"microop_tag\": {}\n", entry.microop_tag));
            if i < self.entries.len() - 1 {
                json.push_str("    },\n");
            } else {
                json.push_str("    }\n");
            }
        }
        json.push_str("  ],\n");
        
        json.push_str("  \"microops\": [\n");
        for (i, op) in self.microops.iter().enumerate() {
            json.push_str("    {\n");
            json.push_str(&format!("      \"type\": \"{}\",\n", op.op_type));
            json.push_str(&format!("      \"details\": \"{}\"\n", op.details));
            if i < self.microops.len() - 1 {
                json.push_str("    },\n");
            } else {
                json.push_str("    }\n");
            }
        }
        json.push_str("  ]\n");
        
        json.push_str("}\n");
        json
    }

    pub fn from_json(json: &str) -> Result<Self, WitnessParseError> {
        let mut witness = Self {
            version: "1.0",
            phase_a_hash: [0u8; 32],
            phase_b_hash: [0u8; 32],
            vector_width: 256,
            code_size: 0,
            microop_count: 0,
            regalloc_fingerprint: [0u8; 8],
            entries: Vec::new(),
            microops: Vec::new(),
        };

        for line in json.lines() {
            let line = line.trim();
            
            if line.starts_with("\"phase_a_hash\":") {
                if let Some(hex) = extract_string_value(line) {
                    witness.phase_a_hash = hex_decode_32(&hex)?;
                }
            } else if line.starts_with("\"phase_b_hash\":") {
                if let Some(hex) = extract_string_value(line) {
                    witness.phase_b_hash = hex_decode_32(&hex)?;
                }
            } else if line.starts_with("\"vector_width\":") {
                if let Some(val) = extract_number_value(line) {
                    witness.vector_width = val as u16;
                }
            } else if line.starts_with("\"code_size\":") {
                if let Some(val) = extract_number_value(line) {
                    witness.code_size = val as u16;
                }
            } else if line.starts_with("\"microop_count\":") {
                if let Some(val) = extract_number_value(line) {
                    witness.microop_count = val as u16;
                }
            } else if line.starts_with("\"regalloc_fingerprint\":") {
                if let Some(hex) = extract_string_value(line) {
                    witness.regalloc_fingerprint = hex_decode_8(&hex)?;
                }
            }
        }

        let entries_start = json.find("\"entries\":");
        let entries_end = json.find("\"microops\":");
        
        if let (Some(start), Some(end)) = (entries_start, entries_end) {
            let entries_json = &json[start..end];
            witness.entries = parse_entries(entries_json)?;
        }

        Ok(witness)
    }

    pub fn verify(&self) -> Result<(), WitnessVerifyError> {
        if self.entries.len() != self.microop_count as usize {
            return Err(WitnessVerifyError::EntryCountMismatch {
                expected: self.microop_count,
                actual: self.entries.len() as u16,
            });
        }

        for (i, entry) in self.entries.iter().enumerate() {
            if (entry.mask_after & !entry.mask_before) != 0 {
                return Err(WitnessVerifyError::MonotonicityViolation {
                    entry_idx: i as u16,
                });
            }
        }

        let mut hasher = crate::semantic_hash::Hasher::new();
        hasher.update(b"SIPHON_WITNESS_V0");
        hasher.update(&self.phase_a_hash);
        hasher.update(&(self.entries.len() as u32).to_be_bytes());
        
        for entry in &self.entries {
            let mut buf = [0u8; 12];
            buf[0..4].copy_from_slice(&entry.rif_node.to_be_bytes());
            buf[4..6].copy_from_slice(&entry.microop_idx.to_be_bytes());
            buf[6..8].copy_from_slice(&entry.mask_before.to_be_bytes());
            buf[8..10].copy_from_slice(&entry.mask_after.to_be_bytes());
            buf[10] = entry.microop_tag;
            buf[11] = 0;
            hasher.update(&buf);
        }
        
        let computed_hash = hasher.finalize();
        if computed_hash.as_bytes() != &self.phase_b_hash {
            return Err(WitnessVerifyError::HashMismatch);
        }

        Ok(())
    }

    pub fn phase_a_hash(&self) -> SemanticHash {
        SemanticHash::from_bytes(self.phase_a_hash)
    }

    pub fn phase_b_hash(&self) -> SemanticHash {
        SemanticHash::from_bytes(self.phase_b_hash)
    }

    pub fn verify_against_graph(&self, graph: &crate::rif::RifGraph, expected_sha: &SemanticHash) -> Result<(), WitnessVerifyError> {
        self.verify()?;

        if self.phase_a_hash != *expected_sha.as_bytes() {
            return Err(WitnessVerifyError::PhaseAHashMismatch);
        }

        let graph_len = graph.nodes.len() as u32;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.rif_node >= graph_len {
                return Err(WitnessVerifyError::RifNodeOutOfBounds {
                    entry_idx: i as u16,
                    rif_node: entry.rif_node,
                    graph_len,
                });
            }

            let rif_node = &graph.nodes[entry.rif_node as usize];
            if !rif_node.is_microop_tag_valid(entry.microop_tag) {
                return Err(WitnessVerifyError::MicroOpTagMismatch {
                    entry_idx: i as u16,
                    expected_tag: rif_node.discriminant(),
                    actual_tag: entry.microop_tag,
                });
            }
        }

        Ok(())
    }

}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WitnessParseError {
    InvalidHex,
    InvalidFormat,
    MissingField(&'static str),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WitnessVerifyError {
    EntryCountMismatch { expected: u16, actual: u16 },
    MonotonicityViolation { entry_idx: u16 },
    HashMismatch,
    PhaseAHashMismatch,
    RifNodeOutOfBounds { entry_idx: u16, rif_node: u32, graph_len: u32 },
    MicroOpTagMismatch { entry_idx: u16, expected_tag: u8, actual_tag: u8 },
}


fn hex_encode(bytes: &[u8]) -> String {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX_CHARS[(b >> 4) as usize] as char);
        s.push(HEX_CHARS[(b & 0xf) as usize] as char);
    }
    s
}

fn hex_decode_32(s: &str) -> Result<[u8; 32], WitnessParseError> {
    if s.len() != 64 {
        return Err(WitnessParseError::InvalidHex);
    }
    let mut bytes = [0u8; 32];
    for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
        bytes[i] = hex_byte(chunk[0])? << 4 | hex_byte(chunk[1])?;
    }
    Ok(bytes)
}

fn hex_decode_8(s: &str) -> Result<[u8; 8], WitnessParseError> {
    if s.len() != 16 {
        return Err(WitnessParseError::InvalidHex);
    }
    let mut bytes = [0u8; 8];
    for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
        bytes[i] = hex_byte(chunk[0])? << 4 | hex_byte(chunk[1])?;
    }
    Ok(bytes)
}

fn hex_byte(c: u8) -> Result<u8, WitnessParseError> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(WitnessParseError::InvalidHex),
    }
}

fn extract_string_value(line: &str) -> Option<String> {
    let start = line.find('"')? + 1;
    let rest = &line[start..];
    let end = rest.find('"')?;
    let rest = &rest[end + 1..];
    let start2 = rest.find('"')? + 1;
    let rest2 = &rest[start2..];
    let end2 = rest2.find('"')?;
    Some(rest2[..end2].to_string())
}

fn extract_number_value(line: &str) -> Option<u64> {
    let colon = line.find(':')?;
    let rest = line[colon + 1..].trim().trim_end_matches(',');
    rest.parse().ok()
}

fn parse_entries(json: &str) -> Result<Vec<SerializedEntry>, WitnessParseError> {
    let mut entries = Vec::new();
    let mut current_entry: Option<SerializedEntry> = None;
    
    for line in json.lines() {
        let line = line.trim();
        
        if line == "{" {
            current_entry = Some(SerializedEntry {
                rif_node: 0,
                microop_idx: 0,
                mask_before: 0,
                mask_after: 0,
                microop_tag: 0,
            });
        } else if line.starts_with("}") {
            if let Some(entry) = current_entry.take() {
                entries.push(entry);
            }
        } else if let Some(ref mut entry) = current_entry {
            if line.starts_with("\"rif_node\":") {
                if let Some(val) = extract_number_value(line) {
                    entry.rif_node = val as u32;
                }
            } else if line.starts_with("\"microop_idx\":") {
                if let Some(val) = extract_number_value(line) {
                    entry.microop_idx = val as u16;
                }
            } else if line.starts_with("\"mask_before\":") {
                if let Some(val) = extract_number_value(line) {
                    entry.mask_before = val as u16;
                }
            } else if line.starts_with("\"mask_after\":") {
                if let Some(val) = extract_number_value(line) {
                    entry.mask_after = val as u16;
                }
            } else if line.starts_with("\"microop_tag\":") {
                if let Some(val) = extract_number_value(line) {
                    entry.microop_tag = val as u8;
                }
            }
        }
    }
    
    Ok(entries)
}

/// Replay contract for deterministic reproduction.
pub struct ReplayContract {
    pub witness: SerializedWitness,
    pub input_packet: Vec<u8>,
    pub expected_output: Vec<u8>,
}

impl ReplayContract {
    pub fn capture(
        witness: SerializedWitness,
        input_packet: &[u8],
        output: &[u8],
    ) -> Self {
        Self {
            witness,
            input_packet: input_packet.to_vec(),
            expected_output: output.to_vec(),
        }
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> String {
        let mut json = String::with_capacity(8192);
        
        json.push_str("{\n");
        json.push_str("  \"witness\": ");
        
        let witness_json = self.witness.to_json();
        for (i, line) in witness_json.lines().enumerate() {
            if i == 0 {
                json.push_str(line);
                json.push('\n');
            } else {
                json.push_str("  ");
                json.push_str(line);
                json.push('\n');
            }
        }
        
        json.push_str(",\n");
        json.push_str(&format!("  \"input_packet\": \"{}\",\n", hex_encode(&self.input_packet)));
        json.push_str(&format!("  \"expected_output\": \"{}\"\n", hex_encode(&self.expected_output)));
        json.push_str("}\n");
        
        json
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_roundtrip() {
        let bytes = [0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0];
        let hex = hex_encode(&bytes);
        assert_eq!(hex, "123456789abcdef0");
        
        let decoded = hex_decode_8(&hex).unwrap();
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_witness_serialization() {
        let witness = SerializedWitness {
            version: "1.0",
            phase_a_hash: [0x42u8; 32],
            phase_b_hash: [0x43u8; 32],
            vector_width: 256,
            code_size: 156,
            microop_count: 2,
            regalloc_fingerprint: [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
            entries: vec![
                SerializedEntry {
                    rif_node: 0,
                    microop_idx: 0,
                    mask_before: 255,
                    mask_after: 255,
                    microop_tag: 1,
                },
                SerializedEntry {
                    rif_node: 1,
                    microop_idx: 1,
                    mask_before: 255,
                    mask_after: 15,
                    microop_tag: 2,
                },
            ],
            microops: vec![
                SerializedMicroOp {
                    op_type: "LoadVector".to_string(),
                    details: "dst=v0, offset=0".to_string(),
                },
            ],
        };

        let json = witness.to_json();
        assert!(json.contains("\"phase_a_hash\":"));
        assert!(json.contains("\"entries\":"));
    }

    #[test]
    fn test_verify_against_graph() {
        use crate::rif::{RifGraph, RifNode, RifVersion, ScalarType, MemoryAccess, MemoryRegion, Alignment, NodeIndex};
        use crate::semantic_hash::compute_semantic_hash;

        let nodes = vec![
            RifNode::Load {
                scalar_type: ScalarType::U64,
                access: MemoryAccess {
                    region: MemoryRegion::PacketInput,
                    offset: 0,
                    length: 8,
                    mask_node_idx: None,
                    alignment: Alignment::Natural,
                },
            },
            RifNode::Emit {
                field_id: 0,
                value_node: NodeIndex(0),
                mask: None,
            },
        ];

        let graph = RifGraph {
            version: RifVersion::CURRENT,
            protocol_version: 1,
            nodes: &nodes,
            max_packet_length: 64,
            version_discriminator_node: NodeIndex(0),
        };

        let sha = compute_semantic_hash(&graph).unwrap();

        let witness = SerializedWitness {
            version: "1.0",
            phase_a_hash: *sha.as_bytes(),
            phase_b_hash: [0u8; 32],
            vector_width: 256,
            code_size: 100,
            microop_count: 2,
            regalloc_fingerprint: [0u8; 8],
            entries: vec![
                SerializedEntry {
                    rif_node: 0,
                    microop_idx: 0,
                    mask_before: 255,
                    mask_after: 255,
                    microop_tag: 0, // LoadVector
                },
                SerializedEntry {
                    rif_node: 1,
                    microop_idx: 1,
                    mask_before: 255,
                    mask_after: 255,
                    microop_tag: 9, // Emit
                },
            ],
            microops: vec![],
        };

        // Should fail hash verification (phase_b_hash is wrong)
        let result = witness.verify_against_graph(&graph, &sha);
        assert!(result.is_err());

        // Test with out-of-bounds rif_node
        let bad_witness = SerializedWitness {
            version: "1.0",
            phase_a_hash: *sha.as_bytes(),
            phase_b_hash: [0u8; 32],
            vector_width: 256,
            code_size: 100,
            microop_count: 1,
            regalloc_fingerprint: [0u8; 8],
            entries: vec![
                SerializedEntry {
                    rif_node: 999, // Out of bounds
                    microop_idx: 0,
                    mask_before: 255,
                    mask_after: 255,
                    microop_tag: 0,
                },
            ],
            microops: vec![],
        };

        let result = bad_witness.verify_against_graph(&graph, &sha);
        // Will fail on entry count mismatch or hash mismatch before bounds check
        assert!(result.is_err());
    }

    #[test]
    fn test_phase_a_hash_mismatch() {
        use crate::rif::{RifGraph, RifNode, RifVersion, ScalarType, MemoryAccess, MemoryRegion, Alignment, NodeIndex};
        use crate::semantic_hash::compute_semantic_hash;

        let nodes = vec![
            RifNode::Load {
                scalar_type: ScalarType::U64,
                access: MemoryAccess {
                    region: MemoryRegion::PacketInput,
                    offset: 0,
                    length: 8,
                    mask_node_idx: None,
                    alignment: Alignment::Natural,
                },
            },
            RifNode::Emit {
                field_id: 0,
                value_node: NodeIndex(0),
                mask: None,
            },
        ];

        let graph = RifGraph {
            version: RifVersion::CURRENT,
            protocol_version: 1,
            nodes: &nodes,
            max_packet_length: 64,
            version_discriminator_node: NodeIndex(0),
        };

        let sha = compute_semantic_hash(&graph).unwrap();

        let witness = SerializedWitness {
            version: "1.0",
            phase_a_hash: [0xFFu8; 32], // Wrong hash
            phase_b_hash: [0u8; 32],
            vector_width: 256,
            code_size: 100,
            microop_count: 0,
            regalloc_fingerprint: [0u8; 8],
            entries: vec![],
            microops: vec![],
        };

        let result = witness.verify_against_graph(&graph, &sha);
        match result {
            Err(WitnessVerifyError::PhaseAHashMismatch) => (),
            Err(WitnessVerifyError::EntryCountMismatch { .. }) => (),
            Err(WitnessVerifyError::HashMismatch) => (),
            other => panic!("Expected PhaseAHashMismatch, EntryCountMismatch, or HashMismatch, got {:?}", other),
        }
    }
}
