# Siphon

**Deterministic Protocol Execution Engine**

Siphon compiles fixed-layout protocol specifications into JIT'd x86-64 machine code for parsing binary messages. Currently a research prototype.

## Quick Start

```bash
# Build
cd cli && cargo build --release

# Define a protocol, compile it, and benchmark
./target/release/siphon check protocols/market_data.siphon
./target/release/siphon compile protocols/market_data.siphon
./target/release/siphon bench protocols/market_data.siphon
```

## Performance (Preliminary)

Microbenchmark results on the Golden Demo protocol (64-byte fixed layout, 6 fields). These numbers are **not production-grade measurements** — no core pinning, no NUMA control, no perf counters, no percentile reporting. They compare JIT output against a deliberately simple scalar reference interpreter.

| Metric | Value | Caveat |
|--------|-------|--------|
| JIT latency | ~17–21 ns/msg | `Instant`-based, mean only |
| Speedup vs reference interpreter | ~37–57x | Baseline is intentionally simple |
| Divergence | 0 | On generated test corpus only |

---

## The Problem

Every distributed system pays a serialization tax. For each message:

1. **Parse** — Decode bytes into structured data
2. **Validate** — Check invariants, bounds, required fields
3. **Transform** — Convert to application representation

Traditional approaches fall short:

| Approach | Problem |
|----------|---------|
| Hand-written parsers | Error-prone, drift from spec |
| Code generators (protobuf) | Runtime overhead, branches, cache pollution |
| JIT compilers | Non-deterministic, hard to verify |

**The fundamental issue**: parsing logic derives from specifications but executes as opaque code. The connection between "what the protocol says" and "what the CPU does" is severed.

---

## The Solution

Siphon treats protocol definitions as the source of truth—not just for documentation, but for execution. Rather than *generating* code that *interprets* a specification, Siphon *compiles* the specification directly into machine instructions.

**Current properties:**

- **Deterministic** — Same input → same output (single-implementation; cross-machine reproducibility not yet validated)
- **Witness-traced** — Every codegen decision is recorded in a witness artifact mapping MicroOps to RIF nodes. Independent verification is not yet implemented.
- **Scalar fast path** — Current emitter produces scalar x86-64 code with stack-backed virtual registers. SIMD/branchless backend is architectural target, not current reality.
- **Cache-friendly** — Predictable memory access, no pointer chasing

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PROTOCOL SPECIFICATION                          │
│                      (market_data.siphon)                           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE A: Trusted Computing Base                                    │
│  ─────────────────────────────────                                  │
│  • Parse specification into RIF (Restricted Intermediate Form)      │
│  • Validate all constraints statically                              │
│  • Compute Semantic Hash (SH_A) — content-addressed identity        │
│                                                                     │
│  Output: RIF Graph + SH_A                                           │
│  Properties: Pure, safe Rust. No unsafe. Deterministic.             │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE B: Lowering Engine                                            │
│  ────────────────────────                                            │
│  • Lower RIF to MicroOps (instruction templates)                     │
│  • Allocate registers (linear scan, no spilling)                    │
│  • Emit x86-64 scalar machine code (SIMD backend planned)           │
│  • Generate Witness (codegen trace, not independent proof)          │
│                                                                     │
│  Output: Executable kernel + Witness + SH_B                         │
│  Status: Self-attested witness; independent verifier not yet built   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RUNTIME: Physical Execution (Partial)                               │
│  ─────────────────────────────────────                               │
│  • io_uring packet ingress (scaffold; incomplete lifecycle)          │
│  • Version dispatcher (bounded jump table)                          │
│  • JIT kernel execution (scalar; branches in prologue/stores)       │
│                                                                     │
│  Status: Runtime is experimental. No production ingress lifecycle.   │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase A: Trusted Computing Base

The security boundary. Pure, safe Rust that can be formally audited.

- **RIF (Restricted Intermediate Form)** — DAG of typed operations with explicit memory regions and bounds. No pointers, recursion, or dynamic allocation.
- **Semantic Hash (SH_A)** — Cryptographic fingerprint of protocol *meaning*, not syntax.

### Phase B: Lowering Engine

Translates RIF into machine code. Outside the TCB. Witness traces codegen decisions but does not constitute independent proof of correctness.

- **MicroOps** — Closed set of instruction templates with fixed footprints
- **Witness** — Trace artifact mapping each MicroOp to a RIF node with mask state
- **Register Allocation** — Linear scan with hard failure on pressure (no spilling)

### Runtime (Experimental)

Partial runtime scaffold. Not a complete production data plane.

- **io_uring** — Partial scaffold for packet ingress (no submit/recycle lifecycle)
- **Version Dispatcher** — Bounded jump table
- **Slow path** — Currently a no-op; must be replaced with reference interpreter fallback

---

## Usage

### 1. Define Protocol

```
# protocols/market_data.siphon

protocol MarketData {
    version: 1;
    max_size: 64;
    
    msg_type:     u8  @offset(0)  @range(1, 10);
    sequence:     u32 @offset(2)  @nonzero;
    timestamp_ns: u64 @offset(6)  @range(0, 86400000000000);
    symbol_id:    u64 @offset(16) @nonzero;
    bid_price:    u64 @offset(32) @range(1, 999999999999);
    ask_price:    u64 @offset(40) @range(1, 999999999999);
}
```

### 2. Check (Phase A)

```bash
$ siphon check protocols/market_data.siphon
✓ Phase A verification PASSED
  SH_A: 280a01f8913630d0...
```

### 3. Compile (Phase A + B)

```bash
$ siphon compile protocols/market_data.siphon
✓ Compilation SUCCEEDED
  MicroOps: 26 | Code: 272 bytes | I-cache: 1.7%
```

### 4. Benchmark

```bash
$ siphon bench protocols/market_data.siphon
✓ Benchmark PASSED — JIT matches reference byte-for-byte
  Latency: 21 ns/msg | Speedup: 44x | Divergence: 0
```

---

## Witness Artifacts

Compilation produces a witness JSON artifact recording all codegen decisions. The current `verify` command checks internal witness consistency (entry count, monotonicity, hash integrity). It does **not** independently verify machine code semantics.

```bash
# Emit witness for inspection
$ siphon compile protocols/market_data.siphon --emit-witness witness.json

# Check witness internal consistency
$ siphon verify witness.json
✓ Witness verification PASSED
```

**Current limitations:**
- Replay seed is derived but not used to drive codegen decisions
- Cross-machine determinism is not validated
- Independent verification (disassembly-based) is not yet implemented

---

## Design Goals (Not All Achieved)

| Technique | Status |
|-----------|--------|
| **Branchless hot path** | **Not yet.** Current scalar emitter has conditional branches in prologue and masked stores. |
| **No allocation** | Kernel hot path has no heap allocation. Runtime init path does allocate. |
| **I-cache resident** | Kernel fits in L1i. 16KB hard budget enforced. |
| **Predictable** | Deterministic codegen within single implementation. Cross-machine not validated. |

---

## Limitations

Siphon is designed for fixed-layout binary protocols with high-throughput requirements. It does **not** support:

- Variable-length fields
- Optional fields
- Recursive structures
- Dynamic schemas
- Non-x86 architectures (currently x86-64 only)

---

## Building

```bash
cd cli && cargo build --release
cargo test
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
