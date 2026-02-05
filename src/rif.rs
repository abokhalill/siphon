//! RIF (Restricted Intermediate Form) Node Definitions

/// RIF format version. Bump this when hash format changes or face the wrath of divergent witnesses.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RifVersion {
    pub major: u16,
    pub minor: u16,
}

impl RifVersion {
    pub const CURRENT: RifVersion = RifVersion { major: 0, minor: 1 };

    /// Big-endian bytes for hashing. Don't get cute with endianness.
    #[inline]
    pub const fn to_bytes(self) -> [u8; 4] {
        [
            (self.major >> 8) as u8,
            self.major as u8,
            (self.minor >> 8) as u8,
            self.minor as u8,
        ]
    }
}

/// Scalar types. Discriminants are frozen—change them and you break every witness ever generated.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ScalarType {
    U8 = 0,
    U16 = 1,
    U32 = 2,
    U64 = 3,
    I32 = 4,
    I64 = 5,
    F32 = 6,
    F64 = 7,
}

impl ScalarType {
    #[inline]
    pub const fn size_bytes(self) -> u8 {
        match self {
            ScalarType::U8 => 1,
            ScalarType::U16 => 2,
            ScalarType::U32 => 4,
            ScalarType::U64 => 8,
            ScalarType::I32 => 4,
            ScalarType::I64 => 8,
            ScalarType::F32 => 4,
            ScalarType::F64 => 8,
        }
    }

    #[inline]
    pub const fn discriminant(self) -> u8 {
        self as u8
    }
}

/// Memory region. Discriminants frozen for hash stability.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MemoryRegion {
    /// Packet input (read-only, NIC-owned).
    PacketInput = 0,
    /// Per-core scratch (L1/L2-resident).
    Scratch = 1,
    /// Output buffer.
    Output = 2,
}

impl MemoryRegion {
    #[inline]
    pub const fn discriminant(self) -> u8 {
        self as u8
    }
}

/// Alignment spec.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    Natural,
    Unaligned,
    Explicit(u8),
}

impl Alignment {
    #[inline]
    pub const fn to_bytes(self) -> [u8; 2] {
        match self {
            Alignment::Natural => [0, 0],
            Alignment::Unaligned => [1, 0],
            Alignment::Explicit(n) => [2, n],
        }
    }
}

/// Memory access descriptor. The (region, offset, length, mask) tuple from ARCH_PIPELINE.
/// Invariants: offset+length can't overflow, length>0, mask refs valid node.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct MemoryAccess {
    pub region: MemoryRegion,
    pub offset: u32,
    pub length: u16,
    pub mask_node_idx: Option<NodeIndex>,
    pub alignment: Alignment,
}

impl MemoryAccess {
    /// Returns None if you violated the invariants. Check yourself.
    #[inline]
    pub const fn validate(self) -> Option<Self> {
        if self.length == 0 {
            return None;
        }
        if (self.offset as u64) + (self.length as u64) > (u32::MAX as u64) {
            return None;
        }
        Some(self)
    }
}

/// Node index. Dense from 0, forward refs forbidden (DAG property).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct NodeIndex(pub u32);

impl NodeIndex {
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn to_bytes(self) -> [u8; 4] {
        self.0.to_be_bytes()
    }
}

/// Validation constraint. Discriminants frozen.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Constraint {
    None = 0,
    NonZero = 1,
    Finite = 2,
    Range { lo: u64, hi: u64 } = 3,
}

impl Constraint {
    pub fn to_bytes(&self) -> [u8; 17] {
        let mut buf = [0u8; 17];
        match self {
            Constraint::None => buf[0] = 0,
            Constraint::NonZero => buf[0] = 1,
            Constraint::Finite => buf[0] = 2,
            Constraint::Range { lo, hi } => {
                buf[0] = 3;
                buf[1..9].copy_from_slice(&lo.to_be_bytes());
                buf[9..17].copy_from_slice(&hi.to_be_bytes());
            }
        }
        buf
    }
}

/// Binary ops. Discriminants frozen.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BinaryOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    And = 4,
    Or = 5,
    Xor = 6,
    Shl = 7,
    Shr = 8,
    Eq = 9,
    Ne = 10,
    Lt = 11,
    Le = 12,
    Gt = 13,
    Ge = 14,
}

impl BinaryOp {
    #[inline]
    pub const fn discriminant(self) -> u8 {
        self as u8
    }
}

/// Unary ops.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum UnaryOp {
    Not = 0,
    Neg = 1,
    ByteSwap = 2,
}

impl UnaryOp {
    #[inline]
    pub const fn discriminant(self) -> u8 {
        self as u8
    }
}

/// RIF node. Forward refs forbidden, masks are monotonic, discriminants frozen.
#[derive(Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum RifNode {
    Load {
        scalar_type: ScalarType,
        access: MemoryAccess,
    } = 0,

    Store {
        value_node: NodeIndex,
        access: MemoryAccess,
    } = 1,

    BinaryOp {
        op: BinaryOp,
        lhs: NodeIndex,
        rhs: NodeIndex,
        result_type: ScalarType,
    } = 2,

    UnaryOp {
        op: UnaryOp,
        operand: NodeIndex,
        result_type: ScalarType,
    } = 3,

    /// Big-endian, zero-padded to 8 bytes.
    Const {
        scalar_type: ScalarType,
        value: [u8; 8],
    } = 4,

    /// Produces mask: 1=valid, 0=invalid.
    Validate {
        value_node: NodeIndex,
        constraint: Constraint,
    } = 5,

    /// Mask refinement. Masks only narrow, never widen.
    Guard {
        parent_mask: Option<NodeIndex>,
        condition: NodeIndex,
    } = 6,

    /// Branchless ternary: mask ? true_val : false_val
    Select {
        mask: NodeIndex,
        true_val: NodeIndex,
        false_val: NodeIndex,
        result_type: ScalarType,
    } = 7,

    /// Terminal node: write parsed value to output.
    Emit {
        field_id: u16,
        value_node: NodeIndex,
        mask: Option<NodeIndex>,
    } = 8,

    /// Ordering constraint (max 8 predecessors).
    Sequence {
        predecessors: [Option<NodeIndex>; 8],
    } = 9,
}

impl RifNode {
    #[inline]
    pub const fn discriminant(&self) -> u8 {
        match self {
            RifNode::Load { .. } => 0,
            RifNode::Store { .. } => 1,
            RifNode::BinaryOp { .. } => 2,
            RifNode::UnaryOp { .. } => 3,
            RifNode::Const { .. } => 4,
            RifNode::Validate { .. } => 5,
            RifNode::Guard { .. } => 6,
            RifNode::Select { .. } => 7,
            RifNode::Emit { .. } => 8,
            RifNode::Sequence { .. } => 9,
        }
    }
}

/// Complete RIF graph. DAG of nodes, must have ≥1 Emit, no forward refs.
pub struct RifGraph<'a> {
    pub version: RifVersion,
    pub protocol_version: u32,
    pub nodes: &'a [RifNode],
    pub max_packet_length: u16,
    pub version_discriminator_node: NodeIndex,
}

impl<'a> RifGraph<'a> {
    /// Validates invariants. Returns Err with the invariant you violated.
    pub fn validate(&self) -> Result<(), &'static str> {
        let node_count = self.nodes.len() as u32;

        if self.version_discriminator_node.0 >= node_count {
            return Err("INV-GRAPH-002: version_discriminator_node out of bounds");
        }

        let mut has_emit = false;

        for (idx, node) in self.nodes.iter().enumerate() {
            let current_idx = idx as u32;

            match node {
                RifNode::Load { access, .. } => {
                    if access.validate().is_none() {
                        return Err("INV-NODE-002: invalid memory access in Load");
                    }
                    if let Some(mask_idx) = access.mask_node_idx {
                        if mask_idx.0 >= current_idx {
                            return Err("INV-NODE-001: forward reference in Load mask");
                        }
                    }
                }
                RifNode::Store { value_node, access } => {
                    if access.validate().is_none() {
                        return Err("INV-NODE-002: invalid memory access in Store");
                    }
                    if value_node.0 >= current_idx {
                        return Err("INV-NODE-001: forward reference in Store value");
                    }
                    if let Some(mask_idx) = access.mask_node_idx {
                        if mask_idx.0 >= current_idx {
                            return Err("INV-NODE-001: forward reference in Store mask");
                        }
                    }
                }
                RifNode::BinaryOp { lhs, rhs, .. } => {
                    if lhs.0 >= current_idx || rhs.0 >= current_idx {
                        return Err("INV-NODE-001: forward reference in BinaryOp");
                    }
                }
                RifNode::UnaryOp { operand, .. } => {
                    if operand.0 >= current_idx {
                        return Err("INV-NODE-001: forward reference in UnaryOp");
                    }
                }
                RifNode::Const { .. } => {}
                RifNode::Validate { value_node, .. } => {
                    if value_node.0 >= current_idx {
                        return Err("INV-NODE-001: forward reference in Validate");
                    }
                }
                RifNode::Guard { parent_mask, condition } => {
                    if let Some(parent) = parent_mask {
                        if parent.0 >= current_idx {
                            return Err("INV-NODE-001: forward reference in Guard parent");
                        }
                    }
                    if condition.0 >= current_idx {
                        return Err("INV-NODE-001: forward reference in Guard condition");
                    }
                }
                RifNode::Select { mask, true_val, false_val, .. } => {
                    if mask.0 >= current_idx || true_val.0 >= current_idx || false_val.0 >= current_idx {
                        return Err("INV-NODE-001: forward reference in Select");
                    }
                }
                RifNode::Emit { value_node, mask, .. } => {
                    has_emit = true;
                    if value_node.0 >= current_idx {
                        return Err("INV-NODE-001: forward reference in Emit value");
                    }
                    if let Some(m) = mask {
                        if m.0 >= current_idx {
                            return Err("INV-NODE-001: forward reference in Emit mask");
                        }
                    }
                }
                RifNode::Sequence { predecessors } => {
                    for pred in predecessors.iter().flatten() {
                        if pred.0 >= current_idx {
                            return Err("INV-NODE-001: forward reference in Sequence");
                        }
                    }
                }
            }
        }

        if !has_emit {
            return Err("INV-GRAPH-003: no Emit node in graph");
        }

        Ok(())
    }
}
