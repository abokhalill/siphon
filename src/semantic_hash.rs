//! Semantic Hash (SH_A)

use crate::normalize::canonicalize_node;
use crate::rif::{MemoryRegion, NodeIndex, RifGraph, RifNode};

/// 256-bit BLAKE3 hash.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SemanticHash {
    bytes: [u8; 32],
}

impl SemanticHash {
    #[inline]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }

    #[inline]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self { bytes }
    }

    /// Hex string, no heap allocation.
    pub fn to_hex(&self) -> [u8; 64] {
        const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";
        let mut hex = [0u8; 64];
        for (i, byte) in self.bytes.iter().enumerate() {
            hex[i * 2] = HEX_CHARS[(byte >> 4) as usize];
            hex[i * 2 + 1] = HEX_CHARS[(byte & 0x0f) as usize];
        }
        hex
    }
}

/// Memory access manifest entry for the witness.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ManifestEntry {
    pub region: MemoryRegion,
    pub offset: u32,
    pub length: u16,
    pub mask: Option<NodeIndex>,
    pub node_idx: NodeIndex,
}

impl ManifestEntry {
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0] = self.region.discriminant();
        buf[1..5].copy_from_slice(&self.offset.to_be_bytes());
        buf[5..7].copy_from_slice(&self.length.to_be_bytes());
        match self.mask {
            Some(idx) => {
                buf[7] = 1;
                buf[8..12].copy_from_slice(&idx.to_bytes());
            }
            None => {
                buf[7] = 0;
            }
        }
        buf[12..16].copy_from_slice(&self.node_idx.to_bytes());
        buf
    }
}

/// Extract memory access manifest from RIF graph. Deterministic order.
pub fn extract_manifest<'a>(graph: &RifGraph<'a>) -> Result<ManifestBuilder, &'static str> {
    let mut builder = ManifestBuilder::new();

    for (idx, node) in graph.nodes.iter().enumerate() {
        let node_idx = NodeIndex(idx as u32);
        match node {
            RifNode::Load { access, .. } => {
                builder.push(ManifestEntry {
                    region: access.region,
                    offset: access.offset,
                    length: access.length,
                    mask: access.mask_node_idx,
                    node_idx,
                })?;
            }
            RifNode::Store { access, .. } => {
                builder.push(ManifestEntry {
                    region: access.region,
                    offset: access.offset,
                    length: access.length,
                    mask: access.mask_node_idx,
                    node_idx,
                })?;
            }
            _ => {}
        }
    }

    Ok(builder)
}

/// Fixed-capacity manifest (256 max). If you need more, your protocol is too complex.
pub struct ManifestBuilder {
    entries: [ManifestEntry; 256],
    count: usize,
}

impl ManifestBuilder {
    pub const fn new() -> Self {
        Self {
            entries: [ManifestEntry {
                region: MemoryRegion::PacketInput,
                offset: 0,
                length: 0,
                mask: None,
                node_idx: NodeIndex(0),
            }; 256],
            count: 0,
        }
    }

    pub fn push(&mut self, entry: ManifestEntry) -> Result<(), &'static str> {
        if self.count >= 256 {
            return Err("manifest capacity exceeded (256)");
        }
        self.entries[self.count] = entry;
        self.count += 1;
        Ok(())
    }

    pub fn as_slice(&self) -> &[ManifestEntry] {
        &self.entries[..self.count]
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Guard chain entry for monotonicity proof.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GuardChainEntry {
    pub node_idx: NodeIndex,
    pub parent: Option<NodeIndex>,
    pub condition: NodeIndex,
}

impl GuardChainEntry {
    pub fn to_bytes(&self) -> [u8; 13] {
        let mut buf = [0u8; 13];
        buf[0..4].copy_from_slice(&self.node_idx.to_bytes());
        match self.parent {
            Some(idx) => {
                buf[4] = 1;
                buf[5..9].copy_from_slice(&idx.to_bytes());
            }
            None => {
                buf[4] = 0;
            }
        }
        buf[9..13].copy_from_slice(&self.condition.to_bytes());
        buf
    }
}

/// Extract guard chain for monotonicity hashing.
pub fn extract_guard_chain<'a>(graph: &RifGraph<'a>) -> Result<GuardChainBuilder, &'static str> {
    let mut builder = GuardChainBuilder::new();

    for (idx, node) in graph.nodes.iter().enumerate() {
        if let RifNode::Guard { parent_mask, condition } = node {
            builder.push(GuardChainEntry {
                node_idx: NodeIndex(idx as u32),
                parent: *parent_mask,
                condition: *condition,
            })?;
        }
    }

    Ok(builder)
}

/// Fixed-capacity guard chain (64 max).
pub struct GuardChainBuilder {
    entries: [GuardChainEntry; 64],
    count: usize,
}

impl GuardChainBuilder {
    pub const fn new() -> Self {
        Self {
            entries: [GuardChainEntry {
                node_idx: NodeIndex(0),
                parent: None,
                condition: NodeIndex(0),
            }; 64],
            count: 0,
        }
    }

    pub fn push(&mut self, entry: GuardChainEntry) -> Result<(), &'static str> {
        if self.count >= 64 {
            return Err("guard chain capacity exceeded (64)");
        }
        self.entries[self.count] = entry;
        self.count += 1;
        Ok(())
    }

    pub fn as_slice(&self) -> &[GuardChainEntry] {
        &self.entries[..self.count]
    }
}

/// BLAKE3 hasher 
pub struct Hasher {
    inner: blake3::Hasher,
}

impl Hasher {
    pub fn new() -> Self {
        Self {
            inner: blake3::Hasher::new(),
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.inner.update(data);
    }

    pub fn finalize(self) -> SemanticHash {
        let hash = self.inner.finalize();
        SemanticHash {
            bytes: *hash.as_bytes(),
        }
    }
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute SH_A. Validates graph first, then hashes everything in deterministic order.
pub fn compute_semantic_hash<'a>(graph: &RifGraph<'a>) -> Result<SemanticHash, &'static str> {
    graph.validate()?;

    let mut hasher = Hasher::new();

    hasher.update(b"SIPHON_RIF_V0");
    hasher.update(&graph.version.to_bytes());
    hasher.update(&graph.protocol_version.to_be_bytes());
    hasher.update(&graph.max_packet_length.to_be_bytes());
    hasher.update(&(graph.nodes.len() as u32).to_be_bytes());
    hasher.update(&graph.version_discriminator_node.to_bytes());

    for node in graph.nodes.iter() {
        hasher.update(canonicalize_node(node).as_bytes());
    }

    let manifest = extract_manifest(graph)?;
    hasher.update(&(manifest.len() as u32).to_be_bytes());
    for entry in manifest.as_slice() {
        hasher.update(&entry.to_bytes());
    }

    let guards = extract_guard_chain(graph)?;
    hasher.update(&(guards.as_slice().len() as u32).to_be_bytes());
    for entry in guards.as_slice() {
        hasher.update(&entry.to_bytes());
    }

    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rif::*;

    #[test]
    fn test_semantic_hash_determinism() {
        // Create a minimal valid RIF graph
        let nodes = [
            // Node 0: Load version byte
            RifNode::Load {
                scalar_type: ScalarType::U8,
                access: MemoryAccess {
                    region: MemoryRegion::PacketInput,
                    offset: 0,
                    length: 1,
                    mask_node_idx: None,
                    alignment: Alignment::Natural,
                },
            },
            // Node 1: Emit the version
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
            max_packet_length: 1500,
            version_discriminator_node: NodeIndex(0),
        };

        // Compute hash twice
        let hash1 = compute_semantic_hash(&graph).unwrap();
        let hash2 = compute_semantic_hash(&graph).unwrap();

        // Must be identical
        assert_eq!(hash1.as_bytes(), hash2.as_bytes());
    }

    #[test]
    fn test_different_graphs_different_hashes() {
        let nodes1 = [
            RifNode::Load {
                scalar_type: ScalarType::U8,
                access: MemoryAccess {
                    region: MemoryRegion::PacketInput,
                    offset: 0,
                    length: 1,
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

        let nodes2 = [
            RifNode::Load {
                scalar_type: ScalarType::U16, // Different type
                access: MemoryAccess {
                    region: MemoryRegion::PacketInput,
                    offset: 0,
                    length: 2,
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

        let graph1 = RifGraph {
            version: RifVersion::CURRENT,
            protocol_version: 1,
            nodes: &nodes1,
            max_packet_length: 1500,
            version_discriminator_node: NodeIndex(0),
        };

        let graph2 = RifGraph {
            version: RifVersion::CURRENT,
            protocol_version: 1,
            nodes: &nodes2,
            max_packet_length: 1500,
            version_discriminator_node: NodeIndex(0),
        };

        let hash1 = compute_semantic_hash(&graph1).unwrap();
        let hash2 = compute_semantic_hash(&graph2).unwrap();

        assert_ne!(hash1.as_bytes(), hash2.as_bytes());
    }
}
