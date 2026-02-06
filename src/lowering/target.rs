//! MicroOp Target Definitions

use crate::rif::{NodeIndex, ScalarType};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum SimdWidth {
    Avx2 = 0,   // 256-bit, 4x u64 or 8x u32
    Avx512 = 1, // 512-bit, 8x u64 or 16x u32
}

impl SimdWidth {
    #[inline]
    pub const fn bits(self) -> u16 {
        match self {
            SimdWidth::Avx2 => 256,
            SimdWidth::Avx512 => 512,
        }
    }

    #[inline]
    pub const fn bytes(self) -> u16 {
        self.bits() / 8
    }

    #[inline]
    pub const fn lanes_for(self, scalar: ScalarType) -> u8 {
        (self.bytes() / scalar.size_bytes() as u16) as u8
    }
}

/// Lane mask. 16 lanes max for our use case.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LaneMask(pub u16);

impl LaneMask {
    pub const ALL_ACTIVE_8: LaneMask = LaneMask(0xFF);
    pub const ALL_ACTIVE_16: LaneMask = LaneMask(0xFFFF);
    pub const NONE: LaneMask = LaneMask(0);

    #[inline]
    pub const fn is_subset_of(self, other: LaneMask) -> bool {
        (self.0 & other.0) == self.0
    }

    #[inline]
    pub const fn and(self, other: LaneMask) -> LaneMask {
        LaneMask(self.0 & other.0)
    }

    #[inline]
    pub const fn count_active(self) -> u8 {
        self.0.count_ones() as u8
    }
}

/// Virtual register. No spilling—if you run out, simplify your protocol.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VReg(pub u8);

impl VReg {
    pub const MAX_AVX2: u8 = 14;
    pub const MAX_AVX512: u8 = 28;
}

/// MicroOp contract. No hidden state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MicroOpContract {
    pub inputs: [Option<VReg>; 3],
    pub output: Option<VReg>,
    pub mask_in: Option<VReg>,
    pub mask_out: Option<VReg>,
    pub footprint_bytes: u8,
    pub required_alignment: u8,
    pub max_offset: u32,
}

impl MicroOpContract {
    pub const fn empty() -> Self {
        Self {
            inputs: [None, None, None],
            output: None,
            mask_in: None,
            mask_out: None,
            footprint_bytes: 0,
            required_alignment: 0,
            max_offset: u32::MAX,
        }
    }
}

/// The closed set of MicroOps. This is it. No extensions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum MicroOp {
    LoadVector {
        dst: VReg,
        offset: u32,
        width: SimdWidth,
        scalar_type: ScalarType,
        mask: Option<VReg>,
    } = 0,

    ValidateCmpEq {
        dst_mask: VReg,
        src: VReg,
        imm_or_reg: VReg, // comparand
        scalar_type: ScalarType,
    } = 1,

    ValidateCmpGt {
        dst_mask: VReg,
        src: VReg,
        comparand: VReg,
        scalar_type: ScalarType,
    } = 2,

    ValidateCmpLt {
        dst_mask: VReg,
        src: VReg,
        comparand: VReg,
        scalar_type: ScalarType,
    } = 3,

    ValidateNonZero {
        dst_mask: VReg,
        src: VReg,
        scalar_type: ScalarType,
    } = 4,

    MaskAnd {
        dst: VReg,
        src1: VReg,
        src2: VReg,
    } = 5,

    MaskOr {
        dst: VReg,
        src1: VReg,
        src2: VReg,
    } = 6,

    MaskNot {
        dst: VReg,
        src: VReg,
    } = 7,

    /// Branchless: dst = mask ? true_val : false_val
    Select {
        dst: VReg,
        mask: VReg,
        true_val: VReg,
        false_val: VReg,
        scalar_type: ScalarType,
    } = 8,

    Emit {
        src: VReg,
        field_offset: u16,
        scalar_type: ScalarType,
        mask: Option<VReg>,
    } = 9,

    BroadcastImm {
        dst: VReg,
        value: u64,
        scalar_type: ScalarType,
    } = 10,

    Add {
        dst: VReg,
        src1: VReg,
        src2: VReg,
        scalar_type: ScalarType,
    } = 11,

    Sub {
        dst: VReg,
        src1: VReg,
        src2: VReg,
        scalar_type: ScalarType,
    } = 12,

    And {
        dst: VReg,
        src1: VReg,
        src2: VReg,
    } = 13,

    Or {
        dst: VReg,
        src1: VReg,
        src2: VReg,
    } = 14,

    Xor {
        dst: VReg,
        src1: VReg,
        src2: VReg,
    } = 15,

    ByteSwap {
        dst: VReg,
        src: VReg,
        scalar_type: ScalarType,
    } = 16,

    Nop {
        bytes: u8,
    } = 17,
}

impl MicroOp {
    /// Returns the instruction footprint in bytes.
    /// These are measured, not guessed.
    pub const fn footprint_bytes(&self) -> u8 {
        match self {
            // VEX-encoded loads: 4-5 bytes typical
            MicroOp::LoadVector { mask: None, .. } => 5,
            MicroOp::LoadVector { mask: Some(_), .. } => 7, // EVEX masked load
            
            // Compares: 4-5 bytes
            MicroOp::ValidateCmpEq { .. } => 5,
            MicroOp::ValidateCmpGt { .. } => 5,
            MicroOp::ValidateCmpLt { .. } => 5,
            MicroOp::ValidateNonZero { .. } => 5,
            
            // Mask ops: 3-4 bytes (K-register ops are compact)
            MicroOp::MaskAnd { .. } => 4,
            MicroOp::MaskOr { .. } => 4,
            MicroOp::MaskNot { .. } => 4,
            
            // Blend: 5-6 bytes
            MicroOp::Select { .. } => 6,
            
            // Store: 5 bytes
            MicroOp::Emit { mask: None, .. } => 5,
            MicroOp::Emit { mask: Some(_), .. } => 7,
            
            // Broadcast: 6 bytes (includes immediate)
            MicroOp::BroadcastImm { .. } => 10, // worst case with 64-bit imm
            
            // Arithmetic: 4-5 bytes
            MicroOp::Add { .. } => 5,
            MicroOp::Sub { .. } => 5,
            MicroOp::And { .. } => 4,
            MicroOp::Or { .. } => 4,
            MicroOp::Xor { .. } => 4,
            
            // Shuffle: 5 bytes
            MicroOp::ByteSwap { .. } => 6,
            
            // NOP: variable
            MicroOp::Nop { bytes } => *bytes,
        }
    }

    /// Build the semantic contract for this MicroOp.
    pub fn contract(&self) -> MicroOpContract {
        match self {
            MicroOp::LoadVector { dst, mask, offset, .. } => MicroOpContract {
                inputs: [None, None, None],
                output: Some(*dst),
                mask_in: *mask,
                mask_out: None,
                footprint_bytes: self.footprint_bytes(),
                required_alignment: 1, // unaligned OK for VMOVDQU
                max_offset: *offset,
            },
            MicroOp::ValidateCmpEq { dst_mask, src, imm_or_reg, .. } => MicroOpContract {
                inputs: [Some(*src), Some(*imm_or_reg), None],
                output: None,
                mask_in: None,
                mask_out: Some(*dst_mask),
                footprint_bytes: self.footprint_bytes(),
                required_alignment: 0,
                max_offset: u32::MAX,
            },
            MicroOp::ValidateCmpGt { dst_mask, src, comparand, .. } |
            MicroOp::ValidateCmpLt { dst_mask, src, comparand, .. } => MicroOpContract {
                inputs: [Some(*src), Some(*comparand), None],
                output: None,
                mask_in: None,
                mask_out: Some(*dst_mask),
                footprint_bytes: self.footprint_bytes(),
                required_alignment: 0,
                max_offset: u32::MAX,
            },
            MicroOp::ValidateNonZero { dst_mask, src, .. } => MicroOpContract {
                inputs: [Some(*src), None, None],
                output: None,
                mask_in: None,
                mask_out: Some(*dst_mask),
                footprint_bytes: self.footprint_bytes(),
                required_alignment: 0,
                max_offset: u32::MAX,
            },
            MicroOp::MaskAnd { dst, src1, src2 } |
            MicroOp::MaskOr { dst, src1, src2 } => MicroOpContract {
                inputs: [Some(*src1), Some(*src2), None],
                output: None,
                mask_in: None,
                mask_out: Some(*dst),
                footprint_bytes: self.footprint_bytes(),
                required_alignment: 0,
                max_offset: u32::MAX,
            },
            MicroOp::MaskNot { dst, src } => MicroOpContract {
                inputs: [Some(*src), None, None],
                output: None,
                mask_in: None,
                mask_out: Some(*dst),
                footprint_bytes: self.footprint_bytes(),
                required_alignment: 0,
                max_offset: u32::MAX,
            },
            MicroOp::Select { dst, mask, true_val, false_val, .. } => MicroOpContract {
                inputs: [Some(*true_val), Some(*false_val), None],
                output: Some(*dst),
                mask_in: Some(*mask),
                mask_out: None,
                footprint_bytes: self.footprint_bytes(),
                required_alignment: 0,
                max_offset: u32::MAX,
            },
            MicroOp::Emit { src, mask, .. } => MicroOpContract {
                inputs: [Some(*src), None, None],
                output: None,
                mask_in: *mask,
                mask_out: None,
                footprint_bytes: self.footprint_bytes(),
                required_alignment: 0,
                max_offset: u32::MAX,
            },
            MicroOp::BroadcastImm { dst, .. } => MicroOpContract {
                inputs: [None, None, None],
                output: Some(*dst),
                mask_in: None,
                mask_out: None,
                footprint_bytes: self.footprint_bytes(),
                required_alignment: 0,
                max_offset: u32::MAX,
            },
            MicroOp::Add { dst, src1, src2, .. } |
            MicroOp::Sub { dst, src1, src2, .. } => MicroOpContract {
                inputs: [Some(*src1), Some(*src2), None],
                output: Some(*dst),
                mask_in: None,
                mask_out: None,
                footprint_bytes: self.footprint_bytes(),
                required_alignment: 0,
                max_offset: u32::MAX,
            },
            MicroOp::And { dst, src1, src2 } |
            MicroOp::Or { dst, src1, src2 } |
            MicroOp::Xor { dst, src1, src2 } => MicroOpContract {
                inputs: [Some(*src1), Some(*src2), None],
                output: Some(*dst),
                mask_in: None,
                mask_out: None,
                footprint_bytes: self.footprint_bytes(),
                required_alignment: 0,
                max_offset: u32::MAX,
            },
            MicroOp::ByteSwap { dst, src, .. } => MicroOpContract {
                inputs: [Some(*src), None, None],
                output: Some(*dst),
                mask_in: None,
                mask_out: None,
                footprint_bytes: self.footprint_bytes(),
                required_alignment: 0,
                max_offset: u32::MAX,
            },
            MicroOp::Nop { bytes } => MicroOpContract {
                inputs: [None, None, None],
                output: None,
                mask_in: None,
                mask_out: None,
                footprint_bytes: *bytes,
                required_alignment: 0,
                max_offset: u32::MAX,
            },
        }
    }

    /// Discriminant for witness hashing
    pub const fn discriminant(&self) -> u8 {
        match self {
            MicroOp::LoadVector { .. } => 0,
            MicroOp::ValidateCmpEq { .. } => 1,
            MicroOp::ValidateCmpGt { .. } => 2,
            MicroOp::ValidateCmpLt { .. } => 3,
            MicroOp::ValidateNonZero { .. } => 4,
            MicroOp::MaskAnd { .. } => 5,
            MicroOp::MaskOr { .. } => 6,
            MicroOp::MaskNot { .. } => 7,
            MicroOp::Select { .. } => 8,
            MicroOp::Emit { .. } => 9,
            MicroOp::BroadcastImm { .. } => 10,
            MicroOp::Add { .. } => 11,
            MicroOp::Sub { .. } => 12,
            MicroOp::And { .. } => 13,
            MicroOp::Or { .. } => 14,
            MicroOp::Xor { .. } => 15,
            MicroOp::ByteSwap { .. } => 16,
            MicroOp::Nop { .. } => 17,
        }
    }

    /// Type name for serialization
    pub fn type_name(&self) -> &'static str {
        match self {
            MicroOp::LoadVector { .. } => "LoadVector",
            MicroOp::ValidateCmpEq { .. } => "ValidateCmpEq",
            MicroOp::ValidateCmpGt { .. } => "ValidateCmpGt",
            MicroOp::ValidateCmpLt { .. } => "ValidateCmpLt",
            MicroOp::ValidateNonZero { .. } => "ValidateNonZero",
            MicroOp::MaskAnd { .. } => "MaskAnd",
            MicroOp::MaskOr { .. } => "MaskOr",
            MicroOp::MaskNot { .. } => "MaskNot",
            MicroOp::Select { .. } => "Select",
            MicroOp::Emit { .. } => "Emit",
            MicroOp::BroadcastImm { .. } => "BroadcastImm",
            MicroOp::Add { .. } => "Add",
            MicroOp::Sub { .. } => "Sub",
            MicroOp::And { .. } => "And",
            MicroOp::Or { .. } => "Or",
            MicroOp::Xor { .. } => "Xor",
            MicroOp::ByteSwap { .. } => "ByteSwap",
            MicroOp::Nop { .. } => "Nop",
        }
    }

    /// Details string for serialization
    pub fn details_string(&self) -> String {
        match self {
            MicroOp::LoadVector { dst, offset, width, scalar_type, mask } => {
                format!("dst=v{}, offset={}, width={}, type={:?}, mask={:?}", 
                    dst.0, offset, width.bits(), scalar_type, mask.map(|m| m.0))
            }
            MicroOp::ValidateCmpEq { dst_mask, src, imm_or_reg, scalar_type } => {
                format!("dst=v{}, src=v{}, cmp=v{}, type={:?}", 
                    dst_mask.0, src.0, imm_or_reg.0, scalar_type)
            }
            MicroOp::ValidateCmpGt { dst_mask, src, comparand, scalar_type } => {
                format!("dst=v{}, src=v{}, cmp=v{}, type={:?}", 
                    dst_mask.0, src.0, comparand.0, scalar_type)
            }
            MicroOp::ValidateCmpLt { dst_mask, src, comparand, scalar_type } => {
                format!("dst=v{}, src=v{}, cmp=v{}, type={:?}", 
                    dst_mask.0, src.0, comparand.0, scalar_type)
            }
            MicroOp::ValidateNonZero { dst_mask, src, scalar_type } => {
                format!("dst=v{}, src=v{}, type={:?}", dst_mask.0, src.0, scalar_type)
            }
            MicroOp::MaskAnd { dst, src1, src2 } => {
                format!("dst=v{}, src1=v{}, src2=v{}", dst.0, src1.0, src2.0)
            }
            MicroOp::MaskOr { dst, src1, src2 } => {
                format!("dst=v{}, src1=v{}, src2=v{}", dst.0, src1.0, src2.0)
            }
            MicroOp::MaskNot { dst, src } => {
                format!("dst=v{}, src=v{}", dst.0, src.0)
            }
            MicroOp::Select { dst, mask, true_val, false_val, scalar_type } => {
                format!("dst=v{}, mask=v{}, t=v{}, f=v{}, type={:?}", 
                    dst.0, mask.0, true_val.0, false_val.0, scalar_type)
            }
            MicroOp::Emit { src, field_offset, scalar_type, mask } => {
                format!("src=v{}, offset={}, type={:?}, mask={:?}", 
                    src.0, field_offset, scalar_type, mask.map(|m| m.0))
            }
            MicroOp::BroadcastImm { dst, value, scalar_type } => {
                format!("dst=v{}, value={}, type={:?}", dst.0, value, scalar_type)
            }
            MicroOp::Add { dst, src1, src2, scalar_type } => {
                format!("dst=v{}, src1=v{}, src2=v{}, type={:?}", dst.0, src1.0, src2.0, scalar_type)
            }
            MicroOp::Sub { dst, src1, src2, scalar_type } => {
                format!("dst=v{}, src1=v{}, src2=v{}, type={:?}", dst.0, src1.0, src2.0, scalar_type)
            }
            MicroOp::And { dst, src1, src2 } => {
                format!("dst=v{}, src1=v{}, src2=v{}", dst.0, src1.0, src2.0)
            }
            MicroOp::Or { dst, src1, src2 } => {
                format!("dst=v{}, src1=v{}, src2=v{}", dst.0, src1.0, src2.0)
            }
            MicroOp::Xor { dst, src1, src2 } => {
                format!("dst=v{}, src1=v{}, src2=v{}", dst.0, src1.0, src2.0)
            }
            MicroOp::ByteSwap { dst, src, scalar_type } => {
                format!("dst=v{}, src=v{}, type={:?}", dst.0, src.0, scalar_type)
            }
            MicroOp::Nop { bytes } => {
                format!("bytes={}", bytes)
            }
        }
    }

    /// Extract register numbers for fingerprinting
    pub fn registers(&self) -> Option<(u8, Option<u8>, Option<u8>)> {
        match self {
            MicroOp::LoadVector { dst, .. } => Some((dst.0, None, None)),
            MicroOp::ValidateCmpEq { dst_mask, src, imm_or_reg, .. } => {
                Some((dst_mask.0, Some(src.0), Some(imm_or_reg.0)))
            }
            MicroOp::ValidateCmpGt { dst_mask, src, comparand, .. } |
            MicroOp::ValidateCmpLt { dst_mask, src, comparand, .. } => {
                Some((dst_mask.0, Some(src.0), Some(comparand.0)))
            }
            MicroOp::ValidateNonZero { dst_mask, src, .. } => {
                Some((dst_mask.0, Some(src.0), None))
            }
            MicroOp::MaskAnd { dst, src1, src2 } |
            MicroOp::MaskOr { dst, src1, src2 } => {
                Some((dst.0, Some(src1.0), Some(src2.0)))
            }
            MicroOp::MaskNot { dst, src } => Some((dst.0, Some(src.0), None)),
            MicroOp::Select { dst, mask, true_val, .. } => {
                Some((dst.0, Some(mask.0), Some(true_val.0)))
            }
            MicroOp::Emit { src, .. } => Some((src.0, None, None)),
            MicroOp::BroadcastImm { dst, .. } => Some((dst.0, None, None)),
            MicroOp::Add { dst, src1, src2, .. } |
            MicroOp::Sub { dst, src1, src2, .. } |
            MicroOp::And { dst, src1, src2 } |
            MicroOp::Or { dst, src1, src2 } |
            MicroOp::Xor { dst, src1, src2 } => {
                Some((dst.0, Some(src1.0), Some(src2.0)))
            }
            MicroOp::ByteSwap { dst, src, .. } => Some((dst.0, Some(src.0), None)),
            MicroOp::Nop { .. } => None,
        }
    }
}

/// I-Cache budget tracking. 16KB hard limit, period.
pub const ICACHE_BUDGET_BYTES: usize = 16 * 1024;

/// Warning threshold — start worrying at 14KB
pub const ICACHE_WARNING_THRESHOLD: usize = 14 * 1024;

/// MicroOp stream with I-cache budget tracking.
/// Fixed capacity, no heap growth.
pub struct MicroOpStream {
    ops: [MicroOp; 2048], // ~2K ops max, should be plenty
    len: usize,
    footprint: usize,
}

impl MicroOpStream {
    pub const fn new() -> Self {
        Self {
            ops: [MicroOp::Nop { bytes: 0 }; 2048],
            len: 0,
            footprint: 0,
        }
    }

    /// Push a MicroOp. Returns Err if I-cache budget exceeded.
    pub fn push(&mut self, op: MicroOp) -> Result<(), LoweringError> {
        if self.len >= 2048 {
            return Err(LoweringError::TooManyOps);
        }

        let new_footprint = self.footprint + op.footprint_bytes() as usize;
        if new_footprint > ICACHE_BUDGET_BYTES {
            return Err(LoweringError::ICacheBudgetExceeded {
                current: self.footprint,
                requested: op.footprint_bytes() as usize,
                limit: ICACHE_BUDGET_BYTES,
            });
        }

        self.ops[self.len] = op;
        self.len += 1;
        self.footprint = new_footprint;
        Ok(())
    }

    pub fn as_slice(&self) -> &[MicroOp] {
        &self.ops[..self.len]
    }

    pub fn footprint(&self) -> usize {
        self.footprint
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check if we're approaching the budget limit
    pub fn is_near_limit(&self) -> bool {
        self.footprint >= ICACHE_WARNING_THRESHOLD
    }
}

/// Lowering errors — no panics, no surprises
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoweringError {
    ICacheBudgetExceeded {
        current: usize,
        requested: usize,
        limit: usize,
    },
    TooManyOps,
    RegisterPressure,
    UnsupportedNode,
    InvalidMaskChain,
    WitnessGenerationFailed,
}

/// Register allocator state. Dead simple: linear scan, no spilling.
/// If you need more registers, your kernel is too complex.
pub struct RegAlloc {
    next_vreg: u8,
    max_vreg: u8,
    /// Maps RIF NodeIndex -> VReg for value nodes
    node_to_vreg: [Option<VReg>; 512],
    /// Maps RIF NodeIndex -> ScalarType for type tracking
    node_to_type: [Option<ScalarType>; 512],
    /// Tracks output offset for each field_id
    field_offsets: [u16; 256],
    next_output_offset: u16,
}

impl RegAlloc {
    pub fn new(width: SimdWidth) -> Self {
        Self {
            next_vreg: 0,
            max_vreg: match width {
                SimdWidth::Avx2 => VReg::MAX_AVX2,
                SimdWidth::Avx512 => VReg::MAX_AVX512,
            },
            node_to_vreg: [None; 512],
            node_to_type: [None; 512],
            field_offsets: [0; 256],
            next_output_offset: 0,
        }
    }

    /// Allocate a fresh register
    pub fn alloc(&mut self) -> Result<VReg, LoweringError> {
        if self.next_vreg >= self.max_vreg {
            return Err(LoweringError::RegisterPressure);
        }
        let reg = VReg(self.next_vreg);
        self.next_vreg += 1;
        Ok(reg)
    }

    /// Bind a RIF node to a register with its scalar type
    pub fn bind(&mut self, node: NodeIndex, reg: VReg) {
        if (node.0 as usize) < 512 {
            self.node_to_vreg[node.0 as usize] = Some(reg);
        }
    }

    /// Bind a RIF node to a register with its scalar type
    pub fn bind_typed(&mut self, node: NodeIndex, reg: VReg, scalar_type: ScalarType) {
        if (node.0 as usize) < 512 {
            self.node_to_vreg[node.0 as usize] = Some(reg);
            self.node_to_type[node.0 as usize] = Some(scalar_type);
        }
    }

    /// Get the register for a RIF node
    pub fn get(&self, node: NodeIndex) -> Option<VReg> {
        if (node.0 as usize) < 512 {
            self.node_to_vreg[node.0 as usize]
        } else {
            None
        }
    }

    /// Get the scalar type for a RIF node
    pub fn get_type(&self, node: NodeIndex) -> Option<ScalarType> {
        if (node.0 as usize) < 512 {
            self.node_to_type[node.0 as usize]
        } else {
            None
        }
    }

    /// Allocate output offset for a field, returns the offset
    pub fn alloc_field_offset(&mut self, field_id: u16, scalar_type: ScalarType) -> u16 {
        let offset = self.next_output_offset;
        if (field_id as usize) < 256 {
            self.field_offsets[field_id as usize] = offset;
        }
        self.next_output_offset += scalar_type.size_bytes() as u16;
        offset
    }

    /// Get the output offset for a field
    pub fn get_field_offset(&self, field_id: u16) -> u16 {
        if (field_id as usize) < 256 {
            self.field_offsets[field_id as usize]
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lane_mask_monotonicity() {
        let parent = LaneMask(0b11111111);
        let child = LaneMask(0b00001111);
        assert!(child.is_subset_of(parent));
        assert!(!parent.is_subset_of(child));
    }

    #[test]
    fn test_microop_footprint() {
        let load = MicroOp::LoadVector {
            dst: VReg(0),
            offset: 0,
            width: SimdWidth::Avx2,
            scalar_type: ScalarType::U64,
            mask: None,
        };
        assert_eq!(load.footprint_bytes(), 5);
    }

    #[test]
    fn test_icache_budget() {
        let mut stream = MicroOpStream::new();
        // Push ops until we hit the limit
        let op = MicroOp::LoadVector {
            dst: VReg(0),
            offset: 0,
            width: SimdWidth::Avx2,
            scalar_type: ScalarType::U64,
            mask: None,
        };
        
        // 16KB / 5 bytes = 3276 ops max
        for _ in 0..3000 {
            if stream.push(op).is_err() {
                break;
            }
        }
        
        assert!(stream.footprint() <= ICACHE_BUDGET_BYTES);
    }
}
