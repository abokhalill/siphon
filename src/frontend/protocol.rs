//! Protocol data structures for Phase-A

use crate::rif::{
    Alignment, Constraint, MemoryAccess, MemoryRegion, NodeIndex, RifGraph, RifNode,
    RifVersion, ScalarType,
};

#[derive(Clone, Debug)]
pub struct Protocol {
    pub name: String,
    pub version: u8,
    pub fields: Vec<ProtocolField>,
    pub max_size: u16,
}

#[derive(Clone, Debug)]
pub struct ProtocolField {
    pub name: String,
    pub offset: u32,
    pub scalar_type: ScalarType,
    pub constraint: Option<FieldConstraint>,
}

#[derive(Clone, Debug)]
pub enum FieldConstraint {
    Range { lo: u64, hi: u64 },
    Equals(u64),
    NonZero,
}

impl Protocol {
    pub fn new(name: &str, version: u8, max_size: u16) -> Self {
        Self {
            name: name.to_string(),
            version,
            fields: Vec::new(),
            max_size,
        }
    }

    pub fn add_field(&mut self, field: ProtocolField) {
        self.fields.push(field);
    }

    pub fn to_rif_graph(&self) -> RifGraph<'static> {
        let nodes: &'static [RifNode] = Box::leak(self.build_rif_nodes().into_boxed_slice());

        RifGraph {
            version: RifVersion::CURRENT,
            protocol_version: self.version as u32,
            nodes,
            max_packet_length: self.max_size,
            version_discriminator_node: NodeIndex(0),
        }
    }

    fn build_rif_nodes(&self) -> Vec<RifNode> {
        let mut nodes = Vec::new();
        let mut field_to_node: Vec<usize> = Vec::new();

        for field in &self.fields {
            let node_idx = nodes.len();
            field_to_node.push(node_idx);

            nodes.push(RifNode::Load {
                scalar_type: field.scalar_type,
                access: MemoryAccess {
                    region: MemoryRegion::PacketInput,
                    offset: field.offset,
                    length: field.scalar_type.size_bytes() as u16,
                    mask_node_idx: None,
                    alignment: Alignment::Natural,
                },
            });
        }

        let mut last_guard: Option<NodeIndex> = None;

        for (i, field) in self.fields.iter().enumerate() {
            if let Some(ref constraint) = field.constraint {
                let load_node = NodeIndex(field_to_node[i] as u32);

                let validate_node_idx = nodes.len();
                match constraint {
                    FieldConstraint::Range { lo, hi } => {
                        nodes.push(RifNode::Validate {
                            value_node: load_node,
                            constraint: Constraint::Range { lo: *lo, hi: *hi },
                        });
                    }
                    FieldConstraint::Equals(val) => {
                        nodes.push(RifNode::Validate {
                            value_node: load_node,
                            constraint: Constraint::Range { lo: *val, hi: *val },
                        });
                    }
                    FieldConstraint::NonZero => {
                        nodes.push(RifNode::Validate {
                            value_node: load_node,
                            constraint: Constraint::NonZero,
                        });
                    }
                }

                let guard_idx = nodes.len();
                nodes.push(RifNode::Guard {
                    parent_mask: last_guard,
                    condition: NodeIndex(validate_node_idx as u32),
                });
                last_guard = Some(NodeIndex(guard_idx as u32));
            }
        }

        for (i, _field) in self.fields.iter().enumerate() {
            let load_node = NodeIndex(field_to_node[i] as u32);
            nodes.push(RifNode::Emit {
                field_id: i as u16,
                value_node: load_node,
                mask: last_guard,
            });
        }

        nodes
    }
}

impl ProtocolField {
    pub fn new(name: &str, offset: u32, scalar_type: ScalarType) -> Self {
        Self {
            name: name.to_string(),
            offset,
            scalar_type,
            constraint: None,
        }
    }

    pub fn with_constraint(mut self, constraint: FieldConstraint) -> Self {
        self.constraint = Some(constraint);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_to_rif() {
        let mut proto = Protocol::new("test", 1, 64);
        proto.add_field(ProtocolField::new("version", 0, ScalarType::U8));
        proto.add_field(
            ProtocolField::new("length", 1, ScalarType::U16)
                .with_constraint(FieldConstraint::Range { lo: 1, hi: 1500 }),
        );

        let graph = proto.to_rif_graph();
        assert!(graph.validate().is_ok());
        assert!(graph.nodes.len() >= 2);
    }

    #[test]
    fn test_protocol_with_guards() {
        let mut proto = Protocol::new("guarded", 1, 128);
        proto.add_field(
            ProtocolField::new("magic", 0, ScalarType::U32)
                .with_constraint(FieldConstraint::Equals(0xDEADBEEF)),
        );
        proto.add_field(
            ProtocolField::new("flags", 4, ScalarType::U8)
                .with_constraint(FieldConstraint::NonZero),
        );

        let graph = proto.to_rif_graph();
        assert!(graph.validate().is_ok());

        let guard_count = graph.nodes.iter().filter(|n| matches!(n, RifNode::Guard { .. })).count();
        assert_eq!(guard_count, 2);
    }
}
