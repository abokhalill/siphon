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
pub fn extract_manifest<'a>(graph: &RifGraph<'a>) -> ManifestBuilder {
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
                });
            }
            RifNode::Store { access, .. } => {
                builder.push(ManifestEntry {
                    region: access.region,
                    offset: access.offset,
                    length: access.length,
                    mask: access.mask_node_idx,
                    node_idx,
                });
            }
            _ => {}
        }
    }

    builder
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

    pub fn push(&mut self, entry: ManifestEntry) {
        if self.count < 256 {
            self.entries[self.count] = entry;
            self.count += 1;
        }
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
pub fn extract_guard_chain<'a>(graph: &RifGraph<'a>) -> GuardChainBuilder {
    let mut builder = GuardChainBuilder::new();

    for (idx, node) in graph.nodes.iter().enumerate() {
        if let RifNode::Guard { parent_mask, condition } = node {
            builder.push(GuardChainEntry {
                node_idx: NodeIndex(idx as u32),
                parent: *parent_mask,
                condition: *condition,
            });
        }
    }

    builder
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

    pub fn push(&mut self, entry: GuardChainEntry) {
        if self.count < 64 {
            self.entries[self.count] = entry;
            self.count += 1;
        }
    }

    pub fn as_slice(&self) -> &[GuardChainEntry] {
        &self.entries[..self.count]
    }
}

/// BLAKE3 hasher. Minimal TCB implementation—use audited crate in prod.
pub struct Hasher {
    state: Blake3State,
}

impl Hasher {
    pub fn new() -> Self {
        Self {
            state: Blake3State::new(),
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.state.update(data);
    }

    pub fn finalize(self) -> SemanticHash {
        SemanticHash {
            bytes: self.state.finalize(),
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

    let manifest = extract_manifest(graph);
    hasher.update(&(manifest.len() as u32).to_be_bytes());
    for entry in manifest.as_slice() {
        hasher.update(&entry.to_bytes());
    }

    let guards = extract_guard_chain(graph);
    hasher.update(&(guards.as_slice().len() as u32).to_be_bytes());
    for entry in guards.as_slice() {
        hasher.update(&entry.to_bytes());
    }

    Ok(hasher.finalize())
}

/// BLAKE3 state. See https://github.com/BLAKE3-team/BLAKE3-specs
struct Blake3State {
    cv: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const MSG_SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

impl Blake3State {
    fn new() -> Self {
        Self {
            cv: IV,
            buf: [0u8; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            let space = 64 - self.buf_len;
            let to_copy = data.len().min(space);

            self.buf[self.buf_len..self.buf_len + to_copy].copy_from_slice(&data[..to_copy]);
            self.buf_len += to_copy;
            data = &data[to_copy..];

            if self.buf_len == 64 {
                self.compress_block(false);
                self.buf_len = 0;
            }
        }
    }

    fn compress_block(&mut self, is_final: bool) {
        let mut m = [0u32; 16];
        for (i, chunk) in self.buf.chunks_exact(4).enumerate() {
            m[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }

        let block_len = if is_final { self.buf_len as u32 } else { 64 };
        let counter = self.total_len / 64;

        let flags = if self.total_len == 0 { 1 } else { 0 }
            | if is_final { 2 | 8 } else { 0 };

        let mut state = [
            self.cv[0], self.cv[1], self.cv[2], self.cv[3],
            self.cv[4], self.cv[5], self.cv[6], self.cv[7],
            IV[0], IV[1], IV[2], IV[3],
            counter as u32, (counter >> 32) as u32, block_len, flags,
        ];

        // 7 rounds
        for round in 0..7 {
            let s = &MSG_SCHEDULE[round];
            // Column step
            g(&mut state, 0, 4, 8, 12, m[s[0]], m[s[1]]);
            g(&mut state, 1, 5, 9, 13, m[s[2]], m[s[3]]);
            g(&mut state, 2, 6, 10, 14, m[s[4]], m[s[5]]);
            g(&mut state, 3, 7, 11, 15, m[s[6]], m[s[7]]);
            // Diagonal step
            g(&mut state, 0, 5, 10, 15, m[s[8]], m[s[9]]);
            g(&mut state, 1, 6, 11, 12, m[s[10]], m[s[11]]);
            g(&mut state, 2, 7, 8, 13, m[s[12]], m[s[13]]);
            g(&mut state, 3, 4, 9, 14, m[s[14]], m[s[15]]);
        }

        // XOR the two halves
        for i in 0..8 {
            self.cv[i] = state[i] ^ state[i + 8];
        }

        self.total_len += 64;
    }

    fn finalize(mut self) -> [u8; 32] {
        // Pad remaining buffer with zeros
        for i in self.buf_len..64 {
            self.buf[i] = 0;
        }

        self.compress_block(true);

        // Output chaining value as bytes (little-endian)
        let mut out = [0u8; 32];
        for (i, word) in self.cv.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
        }
        out
    }
}

/// BLAKE3 G function (quarter round)
#[inline]
fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
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
