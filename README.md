# Siphon

**Deterministic Protocol Execution Engine**

Siphon compiles protocol specifications into verified, branchless machine code that parses binary messages at memory bandwidth.

## Quick Start

```bash
# Build
cd cli && cargo build --release

# Define a protocol, compile it, and benchmark
./target/release/siphon check protocols/market_data.siphon
./target/release/siphon compile protocols/market_data.siphon
./target/release/siphon bench protocols/market_data.siphon
```

## Performance

| Metric | Value |
|--------|-------|
| JIT latency | **21 ns/msg** |
| Throughput | **47M msg/s** |
| Speedup vs interpreter | **44x** |
| Divergence | **0** (byte-for-byte verified) |

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

**Key properties:**

- **Deterministic** — Same input → same output, always
- **Verifiable** — Mathematical proof that generated code matches specification
- **Branchless** — No conditional jumps in the hot path
- **Cache-friendly** — Predictable memory access, no pointer chasing

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PROTOCOL SPECIFICATION                      │
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
│  PHASE B: Verified Lowering Engine                                  │
│  ─────────────────────────────────                                  │
│  • Lower RIF to MicroOps (SIMD instruction templates)               │
│  • Allocate registers (linear scan, no spilling)                    │
│  • Emit x86-64 AVX2 machine code                                    │
│  • Generate Witness (proof of semantic equivalence)                 │
│                                                                     │
│  Output: Executable kernel + Witness + SH_B                         │
│  Invariant: SH_B derived from SH_A proves equivalence               │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RUNTIME: Physical Execution                                        │
│  ────────────────────────────                                       │
│  • io_uring zero-copy packet ingress                                │
│  • Version dispatcher (bounded jump table)                          │
│  • JIT kernel execution (branchless, straight-line)                 │
│  • Non-temporal stores (cache-bypassing output)                     │
│                                                                     │
│  Properties: No heap allocation. No syscalls in hot path.           │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase A: Trusted Computing Base

The security boundary. Pure, safe Rust that can be formally audited.

- **RIF (Restricted Intermediate Form)** — DAG of typed operations with explicit memory regions and bounds. No pointers, recursion, or dynamic allocation.
- **Semantic Hash (SH_A)** — Cryptographic fingerprint of protocol *meaning*, not syntax.

### Phase B: Lowering Engine

Translates RIF into machine code. Outside the TCB but fully verifiable via the Witness.

- **MicroOps** — Closed set of SIMD instruction templates with fixed footprints
- **Witness** — Proof that every MicroOp corresponds to a RIF node
- **Register Allocation** — Linear scan with hard failure on pressure (no spilling)

### Runtime

Executes compiled kernels with zero heap allocation and no syscalls in the hot path.

- **io_uring** — Zero-copy packet ingress
- **Version Dispatcher** — Bounded jump table with Spectre mitigation
- **Non-Temporal Stores** — Cache-bypassing output

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

## Deterministic Replay

Every compilation produces a **Replay Seed** derived from the protocol specification and compilation decisions. Given the same seed, Siphon produces *identical* machine code on any machine.

```bash
# Emit witness for CI verification
$ siphon compile protocols/market_data.siphon --emit-witness witness.json

# Verify artifact matches witness
$ siphon verify witness.json
✓ Witness verification PASSED
```

**Use cases:**
- Reproduce production issues locally with exact same code
- CI artifact verification before deployment
- Audit trail for compliance

---

## Why It's Fast

| Technique | Benefit |
|-----------|---------|
| **Branchless** | No mispredictions—validation uses masks, not jumps |
| **No allocation** | Pre-allocated buffers, no malloc in hot path |
| **I-cache resident** | Entire kernel fits in L1i (<1KB) |
| **Non-temporal stores** | Output bypasses cache, preserves working set |
| **Predictable** | Same code path for every message |

### Comparison

| Metric | Siphon | Protobuf | FlatBuffers |
|--------|--------|----------|-------------|
| Parse latency | **21 ns** | 500-2000 ns | 50-200 ns |
| Branches | 0 | Many | Few |
| Heap allocation | None | Per-message | None |
| Verification | Witness proof | None | None |

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
