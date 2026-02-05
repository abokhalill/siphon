//! Phase B Lowering Demo — From RIF to Executable SIMD Kernel
//!
//! Demonstrates:
//! 1. RIF graph construction
//! 2. Lowering to MicroOps
//! 3. Witness generation
//! 4. Divergence checking (reference vs fast path)

use siphon_tcb::{
    Alignment, MemoryAccess, MemoryRegion, NodeIndex, RifGraph, RifNode, RifVersion,
    ScalarType, compute_semantic_hash, Constraint,
};
use siphon_tcb::lowering::{
    LoweringEngine, SimdWidth, DivergenceChecker,
};

fn main() {
    println!("=== Siphon Phase B: Lowering Engine Demo ===\n");

    demo_minimal_lowering();
    demo_validated_kernel();
    demo_divergence_check();

    println!("\n=== All Phase B demonstrations complete ===");
}

fn demo_minimal_lowering() {
    println!("--- Demo 1: Minimal Kernel Lowering ---");

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
        // Node 1: Emit version as field 0
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

    let phase_a_hash = compute_semantic_hash(&graph).unwrap();
    let engine = LoweringEngine::new(SimdWidth::Avx2, 64);

    match engine.lower(&graph, phase_a_hash) {
        Ok(kernel) => {
            println!("  RIF nodes:     {}", graph.nodes.len());
            println!("  MicroOps:      {}", kernel.ops().len());
            println!("  Code size:     {} bytes", kernel.code_size());
            println!("  Witness size:  {} entries", kernel.witness().len());
            
            let hex = kernel.phase_b_hash().to_hex();
            let hex_str = core::str::from_utf8(&hex).unwrap();
            println!("  SH_B:          {}", hex_str);
        }
        Err(e) => {
            println!("  ERROR: {:?}", e);
        }
    }
    println!();
}

fn demo_validated_kernel() {
    println!("--- Demo 2: Kernel with Validation ---");

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
        // Node 2: Validate length in range [1, 1400]
        RifNode::Validate {
            value_node: NodeIndex(1),
            constraint: Constraint::Range { lo: 1, hi: 1400 },
        },
        // Node 3: Guard on validation
        RifNode::Guard {
            parent_mask: None,
            condition: NodeIndex(2),
        },
        // Node 4: Emit version (guarded)
        RifNode::Emit {
            field_id: 0,
            value_node: NodeIndex(0),
            mask: Some(NodeIndex(3)),
        },
        // Node 5: Emit length (guarded)
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

    let phase_a_hash = compute_semantic_hash(&graph).unwrap();
    let engine = LoweringEngine::new(SimdWidth::Avx2, 1500);

    match engine.lower(&graph, phase_a_hash) {
        Ok(kernel) => {
            println!("  RIF nodes:     {}", graph.nodes.len());
            println!("  MicroOps:      {}", kernel.ops().len());
            println!("  Code size:     {} bytes", kernel.code_size());
            println!("  Witness size:  {} entries", kernel.witness().len());
            
            // Show MicroOp breakdown
            println!("  MicroOp types:");
            let mut load_count = 0;
            let mut validate_count = 0;
            let mut mask_count = 0;
            let mut emit_count = 0;
            let mut other_count = 0;
            
            for op in kernel.ops() {
                match op {
                    siphon_tcb::lowering::MicroOp::LoadVector { .. } => load_count += 1,
                    siphon_tcb::lowering::MicroOp::ValidateCmpEq { .. } |
                    siphon_tcb::lowering::MicroOp::ValidateCmpGt { .. } |
                    siphon_tcb::lowering::MicroOp::ValidateCmpLt { .. } |
                    siphon_tcb::lowering::MicroOp::ValidateNonZero { .. } => validate_count += 1,
                    siphon_tcb::lowering::MicroOp::MaskAnd { .. } |
                    siphon_tcb::lowering::MicroOp::MaskOr { .. } |
                    siphon_tcb::lowering::MicroOp::MaskNot { .. } => mask_count += 1,
                    siphon_tcb::lowering::MicroOp::Emit { .. } => emit_count += 1,
                    _ => other_count += 1,
                }
            }
            
            println!("    - Loads:      {}", load_count);
            println!("    - Validates:  {}", validate_count);
            println!("    - Masks:      {}", mask_count);
            println!("    - Emits:      {}", emit_count);
            println!("    - Other:      {}", other_count);
        }
        Err(e) => {
            println!("  ERROR: {:?}", e);
        }
    }
    println!();
}

fn demo_divergence_check() {
    println!("--- Demo 3: Divergence Checking ---");

    let nodes = [
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

    // Run reference path
    let checker = DivergenceChecker::new(&graph);
    let packet = [0x42u8, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut output = [0u8; 64];

    match checker.reference_execute(&packet, &mut output) {
        Ok(sig) => {
            println!("  Reference path executed successfully");
            println!("  Loads recorded:      {}", sig.loads.iter().take_while(|&&(o, _)| o != 0 || sig.loads[0].0 == 0).count().max(1));
            println!("  Emits recorded:      {}", sig.emits.iter().take_while(|&&(f, _)| f != 0 || sig.emits[0].0 == 0).count().max(1));
            
            // Show emitted value
            let emitted = u64::from_le_bytes([
                output[0], output[1], output[2], output[3],
                output[4], output[5], output[6], output[7],
            ]);
            println!("  Emitted value:       0x{:016X}", emitted);
        }
        Err(_) => {
            println!("  Reference path failed");
        }
    }
    println!();
}
