//! Canonical Normalization for RIF

use crate::rif::{BinaryOp, NodeIndex, RifNode};

/// Commutative ops get operands sorted by index for canonical form.
#[inline]
pub const fn is_commutative(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add | BinaryOp::Mul | BinaryOp::And | BinaryOp::Or | BinaryOp::Xor | BinaryOp::Eq | BinaryOp::Ne
    )
}

/// Sort operands for commutative ops. Pure function.
#[inline]
pub const fn normalize_binary_operands(
    op: BinaryOp,
    lhs: NodeIndex,
    rhs: NodeIndex,
) -> (NodeIndex, NodeIndex) {
    if is_commutative(op) && lhs.0 > rhs.0 {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

/// Serialize node to canonical bytes for hashing. Big-endian, deterministic, no uninitialized padding.
pub fn canonicalize_node(node: &RifNode) -> CanonicalNode {
    let mut buf = [0u8; CANONICAL_NODE_SIZE];
    let mut pos = 0;

    buf[pos] = node.discriminant();
    pos += 1;

    match node {
        RifNode::Load { scalar_type, access } => {
            buf[pos] = scalar_type.discriminant();
            pos += 1;
            pos = write_memory_access(&mut buf, pos, access);
        }
        RifNode::Store { value_node, access } => {
            buf[pos..pos + 4].copy_from_slice(&value_node.to_bytes());
            pos += 4;
            pos = write_memory_access(&mut buf, pos, access);
        }
        RifNode::BinaryOp { op, lhs, rhs, result_type } => {
            let (norm_lhs, norm_rhs) = normalize_binary_operands(*op, *lhs, *rhs);
            buf[pos] = op.discriminant();
            pos += 1;
            buf[pos..pos + 4].copy_from_slice(&norm_lhs.to_bytes());
            pos += 4;
            buf[pos..pos + 4].copy_from_slice(&norm_rhs.to_bytes());
            pos += 4;
            buf[pos] = result_type.discriminant();
            pos += 1;
        }
        RifNode::UnaryOp { op, operand, result_type } => {
            buf[pos] = op.discriminant();
            pos += 1;
            buf[pos..pos + 4].copy_from_slice(&operand.to_bytes());
            pos += 4;
            buf[pos] = result_type.discriminant();
            pos += 1;
        }
        RifNode::Const { scalar_type, value } => {
            buf[pos] = scalar_type.discriminant();
            pos += 1;
            buf[pos..pos + 8].copy_from_slice(value);
            pos += 8;
        }
        RifNode::Validate { value_node, constraint } => {
            buf[pos..pos + 4].copy_from_slice(&value_node.to_bytes());
            pos += 4;
            let constraint_bytes = constraint.to_bytes();
            buf[pos..pos + 17].copy_from_slice(&constraint_bytes);
            pos += 17;
        }
        RifNode::Guard { parent_mask, condition } => {
            match parent_mask {
                Some(idx) => {
                    buf[pos] = 1;
                    pos += 1;
                    buf[pos..pos + 4].copy_from_slice(&idx.to_bytes());
                    pos += 4;
                }
                None => {
                    buf[pos] = 0;
                    pos += 1;
                    pos += 4;
                }
            }
            buf[pos..pos + 4].copy_from_slice(&condition.to_bytes());
            pos += 4;
        }
        RifNode::Select { mask, true_val, false_val, result_type } => {
            buf[pos..pos + 4].copy_from_slice(&mask.to_bytes());
            pos += 4;
            buf[pos..pos + 4].copy_from_slice(&true_val.to_bytes());
            pos += 4;
            buf[pos..pos + 4].copy_from_slice(&false_val.to_bytes());
            pos += 4;
            buf[pos] = result_type.discriminant();
            pos += 1;
        }
        RifNode::Emit { field_id, value_node, mask } => {
            buf[pos..pos + 2].copy_from_slice(&field_id.to_be_bytes());
            pos += 2;
            buf[pos..pos + 4].copy_from_slice(&value_node.to_bytes());
            pos += 4;
            match mask {
                Some(idx) => {
                    buf[pos] = 1;
                    pos += 1;
                    buf[pos..pos + 4].copy_from_slice(&idx.to_bytes());
                    pos += 4;
                }
                None => {
                    buf[pos] = 0;
                    pos += 1;
                    pos += 4;
                }
            }
        }
        RifNode::Sequence { predecessors } => {
            for pred in predecessors.iter() {
                match pred {
                    Some(idx) => {
                        buf[pos] = 1;
                        pos += 1;
                        buf[pos..pos + 4].copy_from_slice(&idx.to_bytes());
                        pos += 4;
                    }
                    None => {
                        buf[pos] = 0;
                        pos += 1;
                        pos += 4;
                    }
                }
            }
        }
    }

    CanonicalNode { data: buf, len: pos }
}

/// Max canonical node size. 64 bytes for cache alignment.
pub const CANONICAL_NODE_SIZE: usize = 64;

#[derive(Clone)]
pub struct CanonicalNode {
    data: [u8; CANONICAL_NODE_SIZE],
    len: usize,
}

impl CanonicalNode {
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data[..self.len]
    }
}

fn write_memory_access(buf: &mut [u8; CANONICAL_NODE_SIZE], mut pos: usize, access: &crate::rif::MemoryAccess) -> usize {
    buf[pos] = access.region.discriminant();
    pos += 1;
    buf[pos..pos + 4].copy_from_slice(&access.offset.to_be_bytes());
    pos += 4;
    buf[pos..pos + 2].copy_from_slice(&access.length.to_be_bytes());
    pos += 2;
    match access.mask_node_idx {
        Some(idx) => {
            buf[pos] = 1;
            pos += 1;
            buf[pos..pos + 4].copy_from_slice(&idx.to_bytes());
            pos += 4;
        }
        None => {
            buf[pos] = 0;
            pos += 1;
            pos += 4;
        }
    }
    let align_bytes = access.alignment.to_bytes();
    buf[pos..pos + 2].copy_from_slice(&align_bytes);
    pos += 2;
    pos
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rif::*;

    #[test]
    fn test_commutative_normalization() {
        let a = NodeIndex(5);
        let b = NodeIndex(3);

        // Add is commutative: should normalize to (3, 5)
        let (lhs, rhs) = normalize_binary_operands(BinaryOp::Add, a, b);
        assert_eq!(lhs.0, 3);
        assert_eq!(rhs.0, 5);

        // Sub is not commutative: should preserve order
        let (lhs, rhs) = normalize_binary_operands(BinaryOp::Sub, a, b);
        assert_eq!(lhs.0, 5);
        assert_eq!(rhs.0, 3);
    }

    #[test]
    fn test_canonicalize_const() {
        let node = RifNode::Const {
            scalar_type: ScalarType::U64,
            value: [0, 0, 0, 0, 0, 0, 0, 42],
        };
        let canonical = canonicalize_node(&node);
        let bytes = canonical.as_bytes();

        // Tag (4 for Const) + type (3 for U64) + 8 value bytes
        assert_eq!(bytes[0], 4); // Const discriminant
        assert_eq!(bytes[1], 3); // U64 discriminant
        assert_eq!(bytes[9], 42); // Last byte of value
    }
}
