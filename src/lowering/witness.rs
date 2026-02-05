//! Witness Generation 

use crate::rif::NodeIndex;
use crate::lowering::target::{LaneMask, MicroOp, VReg};
use crate::semantic_hash::{Hasher, SemanticHash};

/// One entry per MicroOp. The receipt.
#[derive(Clone, Copy, Debug)]
pub struct WitnessEntry {
    pub rif_node: NodeIndex,
    pub microop_idx: u16,
    pub mask_before: LaneMask,
    pub mask_after: LaneMask,
    pub microop_tag: u8,
}

impl WitnessEntry {
    pub fn to_bytes(&self) -> [u8; 12] {
        let mut buf = [0u8; 12];
        buf[0..4].copy_from_slice(&self.rif_node.0.to_be_bytes());
        buf[4..6].copy_from_slice(&self.microop_idx.to_be_bytes());
        buf[6..8].copy_from_slice(&self.mask_before.0.to_be_bytes());
        buf[8..10].copy_from_slice(&self.mask_after.0.to_be_bytes());
        buf[10] = self.microop_tag;
        buf[11] = 0; // padding for alignment
        buf
    }

    #[inline]
    pub fn is_monotonic(&self) -> bool {
        self.mask_after.is_subset_of(self.mask_before)
    }
}

/// Complete witness. 4096 entries max—if you need more, rethink your life choices.
pub struct Witness {
    entries: [WitnessEntry; 4096],
    len: usize,
    phase_a_hash: SemanticHash,
    phase_b_hash: Option<SemanticHash>,
}

impl Witness {
    pub fn new(phase_a_hash: SemanticHash) -> Self {
        Self {
            entries: [WitnessEntry {
                rif_node: NodeIndex(0),
                microop_idx: 0,
                mask_before: LaneMask::NONE,
                mask_after: LaneMask::NONE,
                microop_tag: 0,
            }; 4096],
            len: 0,
            phase_a_hash,
            phase_b_hash: None,
        }
    }

    pub fn push(&mut self, entry: WitnessEntry) -> Result<(), WitnessError> {
        if self.len >= 4096 {
            return Err(WitnessError::TooManyEntries);
        }
        
        if !entry.is_monotonic() {
            return Err(WitnessError::MonotonicityViolation {
                microop_idx: entry.microop_idx,
                mask_before: entry.mask_before,
                mask_after: entry.mask_after,
            });
        }

        self.entries[self.len] = entry;
        self.len += 1;
        Ok(())
    }

    pub fn as_slice(&self) -> &[WitnessEntry] {
        &self.entries[..self.len]
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn finalize(&mut self) -> SemanticHash {
        let mut hasher = Hasher::new();
        
        // Domain separator
        hasher.update(b"SIPHON_WITNESS_V0");
        
        // Include Phase A hash (we're proving equivalence to this)
        hasher.update(self.phase_a_hash.as_bytes());
        
        // Entry count
        hasher.update(&(self.len as u32).to_be_bytes());
        
        // All entries
        for entry in &self.entries[..self.len] {
            hasher.update(&entry.to_bytes());
        }
        
        let hash = hasher.finalize();
        self.phase_b_hash = Some(hash);
        hash
    }

    pub fn phase_b_hash(&self) -> Option<SemanticHash> {
        self.phase_b_hash
    }

    pub fn verify(&self) -> Result<(), WitnessError> {
        for (i, entry) in self.entries[..self.len].iter().enumerate() {
            if !entry.is_monotonic() {
                return Err(WitnessError::MonotonicityViolation {
                    microop_idx: i as u16,
                    mask_before: entry.mask_before,
                    mask_after: entry.mask_after,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WitnessError {
    TooManyEntries,
    MonotonicityViolation {
        microop_idx: u16,
        mask_before: LaneMask,
        mask_after: LaneMask,
    },
    HashMismatch,
}

/// Tracks mask state during lowering, builds the witness.
pub struct WitnessGenerator {
    witness: Witness,
    current_mask: LaneMask,
    microop_counter: u16,
    mask_state: [LaneMask; 32],
}

impl WitnessGenerator {
    pub fn new(phase_a_hash: SemanticHash, initial_mask: LaneMask) -> Self {
        let mut mask_state = [LaneMask::NONE; 32];
        // All mask registers start as "all active" conceptually
        for m in mask_state.iter_mut() {
            *m = initial_mask;
        }
        
        Self {
            witness: Witness::new(phase_a_hash),
            current_mask: initial_mask,
            microop_counter: 0,
            mask_state,
        }
    }

    pub fn record(
        &mut self,
        rif_node: NodeIndex,
        op: &MicroOp,
    ) -> Result<(), WitnessError> {
        let mask_before = self.current_mask;
        
        // Compute mask_after based on the operation
        let mask_after = self.compute_mask_after(op, mask_before);
        
        let entry = WitnessEntry {
            rif_node,
            microop_idx: self.microop_counter,
            mask_before,
            mask_after,
            microop_tag: op.discriminant(),
        };
        
        self.witness.push(entry)?;
        self.microop_counter += 1;
        
        // Update current mask if this op produces a mask
        if let Some(new_mask) = self.extract_produced_mask(op) {
            self.current_mask = new_mask;
        }
        
        Ok(())
    }

    fn compute_mask_after(&self, op: &MicroOp, mask_before: LaneMask) -> LaneMask {
        match op {
            MicroOp::MaskAnd { src1, src2, .. } => {
                let m1 = self.get_mask_reg(*src1);
                let m2 = self.get_mask_reg(*src2);
                m1.and(m2)
            }
            _ => mask_before,
        }
    }

    fn extract_produced_mask(&self, op: &MicroOp) -> Option<LaneMask> {
        match op {
            MicroOp::MaskAnd { src1, src2, .. } => {
                let m1 = self.get_mask_reg(*src1);
                let m2 = self.get_mask_reg(*src2);
                Some(m1.and(m2))
            }
            _ => None,
        }
    }

    fn get_mask_reg(&self, reg: VReg) -> LaneMask {
        if (reg.0 as usize) < 32 {
            self.mask_state[reg.0 as usize]
        } else {
            LaneMask::NONE
        }
    }

    pub fn set_mask_reg(&mut self, reg: VReg, mask: LaneMask) {
        if (reg.0 as usize) < 32 {
            self.mask_state[reg.0 as usize] = mask;
        }
    }

    pub fn refine_mask(&mut self, new_mask: LaneMask) -> Result<(), WitnessError> {
        if !new_mask.is_subset_of(self.current_mask) {
            return Err(WitnessError::MonotonicityViolation {
                microop_idx: self.microop_counter,
                mask_before: self.current_mask,
                mask_after: new_mask,
            });
        }
        self.current_mask = new_mask;
        Ok(())
    }

    pub fn current_mask(&self) -> LaneMask {
        self.current_mask
    }

    pub fn finalize(mut self) -> Witness {
        self.witness.finalize();
        self.witness
    }

    pub fn witness(&self) -> &Witness {
        &self.witness
    }
}

/// Memory access record for safety verification.
#[derive(Clone, Copy, Debug)]
pub struct MemoryAccessRecord {
    pub rif_node: NodeIndex,
    pub offset: u32,
    pub length: u16,
    pub mask: LaneMask,
    pub is_load: bool,
}

impl MemoryAccessRecord {
    #[inline]
    pub fn is_safe(&self, packet_len: u16) -> bool {
        if self.mask.0 == 0 {
            return true;
        }
        (self.offset as u64) + (self.length as u64) <= (packet_len as u64)
    }
}

/// Ensures no access can fault. Masked-off lanes are safe by definition.
pub struct MemorySafetyVerifier {
    accesses: [MemoryAccessRecord; 512],
    len: usize,
    max_packet_len: u16,
}

impl MemorySafetyVerifier {
    pub fn new(max_packet_len: u16) -> Self {
        Self {
            accesses: [MemoryAccessRecord {
                rif_node: NodeIndex(0),
                offset: 0,
                length: 0,
                mask: LaneMask::NONE,
                is_load: true,
            }; 512],
            len: 0,
            max_packet_len,
        }
    }

    pub fn record_access(&mut self, record: MemoryAccessRecord) -> Result<(), WitnessError> {
        if self.len >= 512 {
            return Err(WitnessError::TooManyEntries);
        }
        self.accesses[self.len] = record;
        self.len += 1;
        Ok(())
    }

    pub fn verify_all(&self) -> Result<(), MemorySafetyError> {
        for (i, access) in self.accesses[..self.len].iter().enumerate() {
            if !access.is_safe(self.max_packet_len) {
                return Err(MemorySafetyError::OutOfBounds {
                    access_idx: i as u16,
                    offset: access.offset,
                    length: access.length,
                    packet_len: self.max_packet_len,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemorySafetyError {
    OutOfBounds {
        access_idx: u16,
        offset: u32,
        length: u16,
        packet_len: u16,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_witness_entry_monotonicity() {
        let good = WitnessEntry {
            rif_node: NodeIndex(0),
            microop_idx: 0,
            mask_before: LaneMask(0xFF),
            mask_after: LaneMask(0x0F), // subset — OK
            microop_tag: 0,
        };
        assert!(good.is_monotonic());

        let bad = WitnessEntry {
            rif_node: NodeIndex(0),
            microop_idx: 0,
            mask_before: LaneMask(0x0F),
            mask_after: LaneMask(0xFF), // superset — BAD
            microop_tag: 0,
        };
        assert!(!bad.is_monotonic());
    }

    #[test]
    fn test_witness_generator_refine() {
        let hash = SemanticHash::from_bytes([0u8; 32]);
        let mut gen = WitnessGenerator::new(hash, LaneMask::ALL_ACTIVE_8);
        
        // Valid refinement
        assert!(gen.refine_mask(LaneMask(0x0F)).is_ok());
        assert_eq!(gen.current_mask(), LaneMask(0x0F));
        
        // Invalid refinement (widening)
        assert!(gen.refine_mask(LaneMask(0xFF)).is_err());
    }

    #[test]
    fn test_memory_safety() {
        let mut verifier = MemorySafetyVerifier::new(1500);
        
        // Safe access
        verifier.record_access(MemoryAccessRecord {
            rif_node: NodeIndex(0),
            offset: 0,
            length: 100,
            mask: LaneMask::ALL_ACTIVE_8,
            is_load: true,
        }).unwrap();
        
        assert!(verifier.verify_all().is_ok());
        
        // Unsafe access
        let mut verifier2 = MemorySafetyVerifier::new(100);
        verifier2.record_access(MemoryAccessRecord {
            rif_node: NodeIndex(0),
            offset: 50,
            length: 100, // 50 + 100 = 150 > 100
            mask: LaneMask::ALL_ACTIVE_8,
            is_load: true,
        }).unwrap();
        
        assert!(verifier2.verify_all().is_err());
    }
}
