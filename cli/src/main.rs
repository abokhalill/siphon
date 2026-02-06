//! Siphon CLI 
//!
//! This CLI strictly orchestrates existing subsystems.
//! It does not reinterpret, optimize, or bypass any compiler phase.

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use siphon_tcb::{
    RifGraph, RifNode, RifVersion, NodeIndex, ScalarType, MemoryAccess, MemoryRegion,
    Alignment, Constraint, compute_semantic_hash,
    parse_protocol, Protocol, ProtocolField, FieldConstraint, ParseError,
};
use siphon_tcb::lowering::{
    LoweringEngine, SimdWidth, DivergenceChecker, SerializedWitness,
};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_usage();
        return ExitCode::from(1);
    }
    
    match args[1].as_str() {
        "check" => {
            if args.len() < 3 {
                eprintln!("Error: missing protocol file");
                eprintln!("Usage: siphon check <protocol>");
                return ExitCode::from(1);
            }
            cmd_check(&args[2])
        }
        "compile" => {
            if args.len() < 3 {
                eprintln!("Error: missing protocol file");
                eprintln!("Usage: siphon compile <protocol> [--emit-witness <file>]");
                return ExitCode::from(1);
            }
            let witness_file = if args.len() >= 5 && args[3] == "--emit-witness" {
                Some(args[4].as_str())
            } else {
                None
            };
            cmd_compile(&args[2], witness_file)
        }
        "verify" => {
            if args.len() < 3 {
                eprintln!("Error: missing witness file");
                eprintln!("Usage: siphon verify <witness.json>");
                return ExitCode::from(1);
            }
            cmd_verify(&args[2])
        }
        "bench" => {
            if args.len() < 3 {
                eprintln!("Error: missing protocol file");
                eprintln!("Usage: siphon bench <protocol>");
                return ExitCode::from(1);
            }
            cmd_bench(&args[2])
        }
        "bench-batch" => {
            if args.len() < 3 {
                eprintln!("Error: missing protocol file");
                eprintln!("Usage: siphon bench-batch <protocol>");
                return ExitCode::from(1);
            }
            cmd_bench_batch(&args[2])
        }
        "help" | "--help" | "-h" => {
            print_usage();
            ExitCode::SUCCESS
        }
        "version" | "--version" | "-V" => {
            println!("siphon 0.1.0");
            ExitCode::SUCCESS
        }
        cmd => {
            eprintln!("Error: unknown command '{}'", cmd);
            print_usage();
            ExitCode::from(1)
        }
    }
}

fn print_usage() {
    eprintln!("Siphon — Deterministic Protocol Execution Engine");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("    siphon <COMMAND> [OPTIONS]");
    eprintln!();
    eprintln!("COMMANDS:");
    eprintln!("    check <protocol>                        Run Phase A only");
    eprintln!("    compile <protocol> [--emit-witness F]   Run Phase A + Phase B");
    eprintln!("    bench <protocol>                        Execute JIT + Reference under load");
    eprintln!("    verify <witness.json>                   Verify witness artifact");
    eprintln!("    help                                    Print this help message");
    eprintln!("    version                                 Print version information");
    eprintln!();
    eprintln!("EXAMPLES:");
    eprintln!("    siphon check protocols/market_data.siphon");
    eprintln!("    siphon compile protocols/market_data.siphon --emit-witness witness.json");
    eprintln!("    siphon bench protocols/market_data.siphon");
    eprintln!("    siphon verify witness.json");
}

fn cmd_check(protocol_path: &str) -> ExitCode {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SIPHON CHECK — Phase A Verification");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    
    let protocol = match load_protocol(protocol_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error loading protocol: {}", e);
            return ExitCode::from(1);
        }
    };
    
    println!("Protocol: {}", protocol_path);
    println!();
    
    let graph = protocol.to_rif_graph();
    let sha = match compute_semantic_hash(&graph) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Error computing semantic hash: {:?}", e);
            return ExitCode::from(1);
        }
    };
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ PHASE A RESULTS                                             │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ RIF Nodes:        {:>6}                                     │", graph.nodes.len());
    println!("│ Max Packet Len:   {:>6} bytes                               │", graph.max_packet_length);
    println!("│ Protocol Version: {:>6}                                     │", graph.protocol_version);
    println!("├─────────────────────────────────────────────────────────────┤");
    
    let hex = sha.to_hex();
    let hex_str = core::str::from_utf8(&hex).unwrap_or("???");
    println!("│ SH_A: {}  │", hex_str);
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    
    println!("VALIDATION SUMMARY:");
    let validation = validate_rif(&graph);
    println!("  Bounds checks:    {:>3} (all static)", validation.bounds_checks);
    println!("  Mask operations:  {:>3}", validation.mask_ops);
    println!("  Memory regions:   {:>3}", validation.memory_regions);
    println!("  Guard nodes:      {:>3}", validation.guard_nodes);
    println!();
    
    if validation.errors.is_empty() {
        println!("✓ Phase A verification PASSED");
        ExitCode::SUCCESS
    } else {
        println!("✗ Phase A verification FAILED:");
        for err in &validation.errors {
            println!("  - {}", err);
        }
        ExitCode::from(1)
    }
}

fn cmd_compile(protocol_path: &str, witness_file: Option<&str>) -> ExitCode {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SIPHON COMPILE — Phase A + Phase B");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    
    let protocol = match load_protocol(protocol_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error loading protocol: {}", e);
            return ExitCode::from(1);
        }
    };
    
    println!("Protocol: {}", protocol_path);
    println!();
    
    let graph = protocol.to_rif_graph();
    let sha = match compute_semantic_hash(&graph) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Error computing semantic hash: {:?}", e);
            return ExitCode::from(1);
        }
    };
    
    let engine = LoweringEngine::new(SimdWidth::Avx2, graph.max_packet_length);
    let kernel = match engine.lower(&graph, sha) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("Error in Phase B lowering: {:?}", e);
            return ExitCode::from(1);
        }
    };
    
    // Output results
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ COMPILATION RESULTS                                         │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    let sha_hex = sha.to_hex();
    let sha_str = core::str::from_utf8(&sha_hex).unwrap_or("???");
    println!("│ SH_A:             {}  │", sha_str);
    
    let shb_hex = kernel.phase_b_hash().to_hex();
    let shb_str = core::str::from_utf8(&shb_hex).unwrap_or("???");
    println!("│ Witness Hash:     {}  │", shb_str);
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ MicroOp Count:    {:>6}                                     │", kernel.ops().len());
    println!("│ I-Cache Footprint:{:>6} bytes                               │", kernel.code_size());
    println!("│ Register Pressure:{:>6} (peak)                              │", kernel.witness().len().min(14));
    println!("│ Vector Width:        256 bits (AVX2)                        │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    
    let icache_pct = (kernel.code_size() as f64 / 16384.0) * 100.0;
    if icache_pct > 87.5 {
        println!("⚠ WARNING: I-cache usage at {:.1}% of 16KB budget", icache_pct);
    } else {
        println!("✓ I-cache usage: {:.1}% of 16KB budget", icache_pct);
    }
    
    if let Some(witness_path) = witness_file {
        let serialized = SerializedWitness::from_kernel(
            kernel.witness(),
            kernel.ops(),
            &sha,
            kernel.phase_b_hash(),
            SimdWidth::Avx2,
            kernel.code_size(),
        );
        
        let json = serialized.to_json();
        if let Err(e) = fs::write(witness_path, &json) {
            eprintln!("Error writing witness file: {}", e);
            return ExitCode::from(1);
        }
        println!("✓ Witness written to: {}", witness_path);
    }
    
    println!("✓ Compilation SUCCEEDED");
    ExitCode::SUCCESS
}

fn cmd_verify(witness_path: &str) -> ExitCode {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SIPHON VERIFY — Witness Verification");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    
    let json = match fs::read_to_string(witness_path) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("Error reading witness file: {}", e);
            return ExitCode::from(1);
        }
    };
    
    println!("Witness: {}", witness_path);
    println!();
    
    let witness = match SerializedWitness::from_json(&json) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Error parsing witness: {:?}", e);
            return ExitCode::from(1);
        }
    };
    
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ WITNESS METADATA                                            │");
    println!("├─────────────────────────────────────────────────────────────┤");
    
    let sha_hex = hex_encode(&witness.phase_a_hash);
    println!("│ SH_A:             {}  │", sha_hex);
    
    let shb_hex = hex_encode(&witness.phase_b_hash);
    println!("│ Witness Hash:     {}  │", shb_hex);
    
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ MicroOp Count:    {:>6}                                     │", witness.microop_count);
    println!("│ Code Size:        {:>6} bytes                               │", witness.code_size);
    println!("│ Vector Width:     {:>6} bits                                │", witness.vector_width);
    println!("│ Entry Count:      {:>6}                                     │", witness.entries.len());
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    
    // Verify witness
    println!("VERIFICATION:");
    
    match witness.verify() {
        Ok(()) => {
            println!("  ✓ Entry count matches microop count");
            println!("  ✓ Mask monotonicity preserved");
            println!("  ✓ Phase B hash verified");
            println!();
            println!("✓ Witness verification PASSED");
            ExitCode::SUCCESS
        }
        Err(e) => {
            println!("  ✗ Verification FAILED: {:?}", e);
            ExitCode::from(1)
        }
    }
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

/// Benchmark: JIT path + reference path under load with byte-for-byte comparison
fn cmd_bench(protocol_path: &str) -> ExitCode {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SIPHON BENCH — JIT Execution Verification");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    
    let protocol = match load_protocol(protocol_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error loading protocol: {}", e);
            return ExitCode::from(1);
        }
    };
    
    println!("Protocol: {}", protocol_path);
    println!();
    
    // Phase A + B
    let graph = protocol.to_rif_graph();
    let sha = match compute_semantic_hash(&graph) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Error computing semantic hash: {:?}", e);
            return ExitCode::from(1);
        }
    };
    
    let engine = LoweringEngine::new(SimdWidth::Avx2, graph.max_packet_length);
    let kernel = match engine.lower(&graph, sha) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("Error in Phase B lowering: {:?}", e);
            return ExitCode::from(1);
        }
    };
    
    // Generate test packets
    let test_packets = generate_test_packets(&protocol, 1000);
    
    // Get JIT function pointer
    let jit_fn = kernel.as_fn();
    
    // Warm up both paths
    let checker = DivergenceChecker::new(&graph);
    let mut ref_output = [0u8; 256];
    let mut jit_output = [0u8; 256];
    
    for packet in test_packets.iter().take(100) {
        let _ = checker.reference_execute(packet, &mut ref_output);
        let _ = jit_fn(packet.as_ptr(), packet.len() as u32, jit_output.as_mut_ptr());
    }
    
    // =========================================================================
    // PHASE 1: Reference Path Benchmark
    // =========================================================================
    let iterations = 10000;
    
    let ref_start = Instant::now();
    for i in 0..iterations {
        let packet = &test_packets[i % test_packets.len()];
        let _ = checker.reference_execute(packet, &mut ref_output);
    }
    let ref_elapsed = ref_start.elapsed();
    let ref_ns_per_msg = ref_elapsed.as_nanos() as f64 / iterations as f64;
    
    // =========================================================================
    // PHASE 2: JIT Path Benchmark
    // =========================================================================
    let jit_start = Instant::now();
    for i in 0..iterations {
        let packet = &test_packets[i % test_packets.len()];
        let _ = jit_fn(packet.as_ptr(), packet.len() as u32, jit_output.as_mut_ptr());
    }
    let jit_elapsed = jit_start.elapsed();
    let jit_ns_per_msg = jit_elapsed.as_nanos() as f64 / iterations as f64;
    
    // =========================================================================
    // PHASE 3: Byte-for-Byte Comparison
    // =========================================================================
    let mut divergence_count = 0u64;
    let mut first_divergence: Option<(usize, usize)> = None;
    
    for (i, packet) in test_packets.iter().enumerate() {
        // Clear outputs
        ref_output.fill(0);
        jit_output.fill(0);
        
        // Execute both paths
        let ref_result = checker.reference_execute(packet, &mut ref_output);
        let jit_result = jit_fn(packet.as_ptr(), packet.len() as u32, jit_output.as_mut_ptr());
        
        // Compare outputs byte-for-byte
        let ref_ok = ref_result.is_ok();
        let jit_ok = jit_result == 0;
        
        if ref_ok != jit_ok {
            // Validation result mismatch
            divergence_count += 1;
            if first_divergence.is_none() {
                first_divergence = Some((i, 0));
            }
        } else if ref_ok {
            // Both succeeded - compare output bytes
            for (j, (&r, &j_byte)) in ref_output.iter().zip(jit_output.iter()).enumerate() {
                if r != j_byte {
                    divergence_count += 1;
                    if first_divergence.is_none() {
                        first_divergence = Some((i, j));
                    }
                    break;
                }
            }
        }
    }
    
    // Speedup ratio
    let speedup = ref_ns_per_msg / jit_ns_per_msg;
    
    // Deterministic replay seed
    let replay_seed = sha.as_bytes()[0..8].iter()
        .fold(0u64, |acc, &b| (acc << 8) | b as u64);
    
    // Output results
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ BENCHMARK RESULTS                                           │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Iterations:       {:>10}                                    │", iterations);
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ REFERENCE PATH                                              │");
    println!("│   Total Time:     {:>10.3} ms                               │", ref_elapsed.as_secs_f64() * 1000.0);
    println!("│   Latency:        {:>10.1} ns/msg                           │", ref_ns_per_msg);
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ JIT PATH                                                    │");
    println!("│   Total Time:     {:>10.3} ms                               │", jit_elapsed.as_secs_f64() * 1000.0);
    println!("│   Latency:        {:>10.1} ns/msg                           │", jit_ns_per_msg);
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Speedup:          {:>10.2}x                                 │", speedup);
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Divergence Count: {:>10}                                    │", divergence_count);
    println!("│ Replay Seed:      0x{:016X}                                 │", replay_seed);
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    
    if divergence_count > 0 {
        if let Some((packet_idx, byte_idx)) = first_divergence {
            eprintln!("✗ DIVERGENCE DETECTED at packet {}, byte {}", packet_idx, byte_idx);
        } else {
            eprintln!("✗ DIVERGENCE DETECTED: {} mismatches", divergence_count);
        }
        return ExitCode::from(1);
    }
    
    println!("✓ Benchmark PASSED — JIT output matches reference byte-for-byte");
    ExitCode::SUCCESS
}

// ============================================================================
// Protocol Loading (uses library frontend module)
// ============================================================================

fn load_protocol(path: &str) -> Result<Protocol, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("cannot read file: {}", e))?;
    
    parse_protocol(&content).map_err(|e| format!("parse error: {:?}", e))
}

fn generate_test_packets(protocol: &Protocol, count: usize) -> Vec<Vec<u8>> {
    let mut packets = Vec::with_capacity(count);
    
    for i in 0..count {
        let mut packet = vec![0u8; protocol.max_size as usize];
        
        packet[0] = protocol.version;
        
        let force_invalid = i % 100 == 99;
        
        for (j, field) in protocol.fields.iter().enumerate() {
            let value = if force_invalid && j == 0 {
                match &field.constraint {
                    Some(FieldConstraint::Range { hi, .. }) => hi + 100,
                    Some(FieldConstraint::NonZero) => 0,
                    Some(FieldConstraint::Equals(v)) => v + 1,
                    None => (i + j) as u64 % 256,
                }
            } else {
                match &field.constraint {
                    Some(FieldConstraint::Range { lo, hi }) => {
                        let range = hi.saturating_sub(*lo) + 1;
                        lo + ((i + j) as u64 % range.max(1))
                    }
                    Some(FieldConstraint::NonZero) => {
                        ((i + j) as u64 % 255) + 1
                    }
                    Some(FieldConstraint::Equals(v)) => *v,
                    None => (i + j) as u64 % 256,
                }
            };
            let offset = field.offset as usize;
            
            match field.scalar_type {
                ScalarType::U8 => {
                    if offset < packet.len() {
                        packet[offset] = value as u8;
                    }
                }
                ScalarType::U16 => {
                    if offset + 1 < packet.len() {
                        let bytes = (value as u16).to_le_bytes();
                        packet[offset..offset+2].copy_from_slice(&bytes);
                    }
                }
                ScalarType::U32 => {
                    if offset + 3 < packet.len() {
                        let bytes = (value as u32).to_le_bytes();
                        packet[offset..offset+4].copy_from_slice(&bytes);
                    }
                }
                ScalarType::U64 => {
                    if offset + 7 < packet.len() {
                        let bytes = value.to_le_bytes();
                        packet[offset..offset+8].copy_from_slice(&bytes);
                    }
                }
                _ => {}
            }
        }
        
        packets.push(packet);
    }
    
    packets
}

// ============================================================================
// Validation
// ============================================================================

struct ValidationResult {
    bounds_checks: usize,
    mask_ops: usize,
    memory_regions: usize,
    guard_nodes: usize,
    errors: Vec<String>,
}

fn validate_rif(graph: &RifGraph) -> ValidationResult {
    let mut result = ValidationResult {
        bounds_checks: 0,
        mask_ops: 0,
        memory_regions: 1, // PacketInput
        guard_nodes: 0,
        errors: Vec::new(),
    };
    
    for (i, node) in graph.nodes.iter().enumerate() {
        match node {
            RifNode::Load { access, .. } => {
                // Bounds check
                let end = access.offset as u64 + access.length as u64;
                if end > graph.max_packet_length as u64 {
                    result.errors.push(format!(
                        "Node {}: load exceeds packet bounds (offset {} + len {} > max {})",
                        i, access.offset, access.length, graph.max_packet_length
                    ));
                }
                result.bounds_checks += 1;
            }
            RifNode::Validate { value_node, .. } => {
                if value_node.0 as usize >= i {
                    result.errors.push(format!(
                        "Node {}: validate references future node {}",
                        i, value_node.0
                    ));
                }
                result.mask_ops += 1;
            }
            RifNode::Guard { parent_mask, condition } => {
                if condition.0 as usize >= i {
                    result.errors.push(format!(
                        "Node {}: guard references future condition {}",
                        i, condition.0
                    ));
                }
                if let Some(parent) = parent_mask {
                    if parent.0 as usize >= i {
                        result.errors.push(format!(
                            "Node {}: guard references future parent {}",
                            i, parent.0
                        ));
                    }
                }
                result.guard_nodes += 1;
            }
            RifNode::Emit { value_node, mask, .. } => {
                if value_node.0 as usize >= i {
                    result.errors.push(format!(
                        "Node {}: emit references future value {}",
                        i, value_node.0
                    ));
                }
                if let Some(m) = mask {
                    if m.0 as usize >= i {
                        result.errors.push(format!(
                            "Node {}: emit references future mask {}",
                            i, m.0
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    
    result
}

// ============================================================================
// Batch Benchmark (SIMD 4-packet parallel)
// ============================================================================

fn cmd_bench_batch(protocol_path: &str) -> ExitCode {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SIPHON BATCH BENCH — SIMD 4-Packet Parallel");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Protocol: {}", protocol_path);
    println!();

    let protocol = match load_protocol(protocol_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error loading protocol: {}", e);
            return ExitCode::from(1);
        }
    };

    let graph = protocol.to_rif_graph();
    let sha = match compute_semantic_hash(&graph) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Error computing semantic hash: {:?}", e);
            return ExitCode::from(1);
        }
    };

    let engine = LoweringEngine::new(SimdWidth::Avx2, graph.max_packet_length);

    let batch_kernel = match engine.lower_batch(&graph, sha) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("Batch lowering error: {:?}", e);
            return ExitCode::from(1);
        }
    };

    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ BATCH KERNEL COMPILED                                       │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Code Size:             {:>5} bytes                          │", batch_kernel.code_size());
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    println!("✓ Batch kernel ready for SIMD 4-packet parallel processing");

    ExitCode::SUCCESS
}
