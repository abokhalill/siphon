//! Minimal Deterministic Example: Stable SH_A Generation
//!
//! This example demonstrates that the Semantic Hash (SH_A) is:
//! 1. Deterministic across invocations
//! 2. Stable for identical RIF graphs
//! 3. Different for semantically different graphs
//!
//! Run with: cargo run --example stable_hash_demo

use siphon_tcb::{
    Alignment, MemoryAccess, MemoryRegion, NodeIndex, RifGraph, RifNode, RifVersion,
    ScalarType, compute_semantic_hash, Constraint,
};

fn main() {
    println!("=== Siphon Phase A: Semantic Hash Stability Demo ===\n");

    // Example 1: Minimal protocol kernel (version byte + emit)
    demo_minimal_kernel();

    // Example 2: Protocol with validation constraint
    demo_validated_kernel();

    // Example 3: Demonstrate hash stability across multiple computations
    demo_hash_stability();

    // Example 4: Demonstrate hash sensitivity to changes
    demo_hash_sensitivity();

    println!("\n=== All demonstrations complete ===");
}

fn demo_minimal_kernel() {
    println!("--- Demo 1: Minimal Protocol Kernel ---");

    let nodes = [
        // Node 0: Load version discriminator byte at offset 0
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
        // Node 1: Emit the version as field 0
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

    match compute_semantic_hash(&graph) {
        Ok(hash) => {
            let hex = hash.to_hex();
            let hex_str = core::str::from_utf8(&hex).unwrap();
            println!("  Graph: 2 nodes (Load + Emit)");
            println!("  SH_A:  {}", hex_str);
        }
        Err(e) => {
            println!("  ERROR: {}", e);
        }
    }
    println!();
}

fn demo_validated_kernel() {
    println!("--- Demo 2: Protocol with Validation ---");

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
        // Node 1: Load payload length (u16 at offset 2)
        RifNode::Load {
            scalar_type: ScalarType::U16,
            access: MemoryAccess {
                region: MemoryRegion::PacketInput,
                offset: 2,
                length: 2,
                mask_node_idx: None,
                alignment: Alignment::Natural,
            },
        },
        // Node 2: Validate length is in range [1, 1400]
        RifNode::Validate {
            value_node: NodeIndex(1),
            constraint: Constraint::Range { lo: 1, hi: 1400 },
        },
        // Node 3: Guard on validation result
        RifNode::Guard {
            parent_mask: None,
            condition: NodeIndex(2),
        },
        // Node 4: Emit version (under guard)
        RifNode::Emit {
            field_id: 0,
            value_node: NodeIndex(0),
            mask: Some(NodeIndex(3)),
        },
        // Node 5: Emit length (under guard)
        RifNode::Emit {
            field_id: 1,
            value_node: NodeIndex(1),
            mask: Some(NodeIndex(3)),
        },
    ];

    let graph = RifGraph {
        version: RifVersion::CURRENT,
        protocol_version: 1,
        nodes: &nodes,
        max_packet_length: 1500,
        version_discriminator_node: NodeIndex(0),
    };

    match compute_semantic_hash(&graph) {
        Ok(hash) => {
            let hex = hash.to_hex();
            let hex_str = core::str::from_utf8(&hex).unwrap();
            println!("  Graph: 6 nodes (Load, Load, Validate, Guard, Emit, Emit)");
            println!("  SH_A:  {}", hex_str);
        }
        Err(e) => {
            println!("  ERROR: {}", e);
        }
    }
    println!();
}

fn demo_hash_stability() {
    println!("--- Demo 3: Hash Stability (10 iterations) ---");

    let nodes_with_version = [
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
        RifNode::Const {
            scalar_type: ScalarType::U64,
            value: [0, 0, 0, 0, 0, 0, 0, 42],
        },
        RifNode::Emit {
            field_id: 0,
            value_node: NodeIndex(1),
            mask: None,
        },
    ];

    let graph = RifGraph {
        version: RifVersion::CURRENT,
        protocol_version: 1,
        nodes: &nodes_with_version,
        max_packet_length: 64,
        version_discriminator_node: NodeIndex(0),
    };

    let mut hashes: [[u8; 32]; 10] = [[0u8; 32]; 10];

    for i in 0..10 {
        match compute_semantic_hash(&graph) {
            Ok(hash) => {
                hashes[i] = *hash.as_bytes();
            }
            Err(e) => {
                println!("  Iteration {}: ERROR - {}", i, e);
                return;
            }
        }
    }

    // Verify all hashes are identical
    let all_equal = hashes.windows(2).all(|w| w[0] == w[1]);

    let hex = siphon_tcb::SemanticHash::from_bytes(hashes[0]).to_hex();
    let hex_str = core::str::from_utf8(&hex).unwrap();

    println!("  Computed hash 10 times");
    println!("  All identical: {}", all_equal);
    println!("  SH_A:  {}", hex_str);
    println!();
}

fn demo_hash_sensitivity() {
    println!("--- Demo 4: Hash Sensitivity to Changes ---");

    // Base graph
    let nodes_base = [
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

    // Modified: different offset
    let nodes_offset = [
        RifNode::Load {
            scalar_type: ScalarType::U8,
            access: MemoryAccess {
                region: MemoryRegion::PacketInput,
                offset: 1, // Changed from 0 to 1
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

    // Modified: different type
    let nodes_type = [
        RifNode::Load {
            scalar_type: ScalarType::U16, // Changed from U8
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

    let graph_base = RifGraph {
        version: RifVersion::CURRENT,
        protocol_version: 1,
        nodes: &nodes_base,
        max_packet_length: 64,
        version_discriminator_node: NodeIndex(0),
    };

    let graph_offset = RifGraph {
        version: RifVersion::CURRENT,
        protocol_version: 1,
        nodes: &nodes_offset,
        max_packet_length: 64,
        version_discriminator_node: NodeIndex(0),
    };

    let graph_type = RifGraph {
        version: RifVersion::CURRENT,
        protocol_version: 1,
        nodes: &nodes_type,
        max_packet_length: 64,
        version_discriminator_node: NodeIndex(0),
    };

    let hash_base = compute_semantic_hash(&graph_base).unwrap();
    let hash_offset = compute_semantic_hash(&graph_offset).unwrap();
    let hash_type = compute_semantic_hash(&graph_type).unwrap();

    let hex_base = hash_base.to_hex();
    let hex_offset = hash_offset.to_hex();
    let hex_type = hash_type.to_hex();

    println!("  Base (offset=0, U8):    {}", core::str::from_utf8(&hex_base).unwrap());
    println!("  Offset change (1):      {}", core::str::from_utf8(&hex_offset).unwrap());
    println!("  Type change (U16):      {}", core::str::from_utf8(&hex_type).unwrap());
    println!();
    println!("  Base != Offset: {}", hash_base.as_bytes() != hash_offset.as_bytes());
    println!("  Base != Type:   {}", hash_base.as_bytes() != hash_type.as_bytes());
    println!("  Offset != Type: {}", hash_offset.as_bytes() != hash_type.as_bytes());
}
