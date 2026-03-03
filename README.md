# Siphon

**Protocol Compilation Engine (Research Prototype)**

Siphon compiles fixed-layout protocol specifications into JIT-emitted x86-64 machine code with a witness artifact linking generated code back to the specification.

> **Status:** Research prototype. Not production-hardened. See [Current Limitations](#current-limitations) and [Implementation Status](#implementation-status).

## Quick Start

```bash
# Build
cd cli && cargo build --release

# Define a protocol, compile it, and benchmark
./target/release/siphon check protocols/market_data.siphon
./target/release/siphon compile protocols/market_data.siphon
./target/release/siphon bench protocols/market_data.siphon
```

## Observed Performance (Uncontrolled)

Measured on a single machine, no core pinning, no NUMA control, no perf counters.
These numbers are **indicative only** and not reproducible benchmarks.

| Metric | Observed |
|--------|----------|
| JIT latency | ~17–21 ns/msg |
| Speedup vs reference interpreter | ~37–57x |
| Output divergence (JIT vs reference) | 0 on test corpus |

Methodology limitations: `Instant`-based timing over 10k iterations, no confidence intervals, no percentile reporting, compared against built-in scalar reference interpreter only.

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

## The Approach

Siphon treats protocol definitions as the source of truth—not just for documentation, but for execution. Rather than *generating* code that *interprets* a specification, Siphon *compiles* the specification directly into machine instructions.

**Design goals:**

- **Deterministic** — Same protocol definition produces the same kernel within a single toolchain version
- **Auditable** — Witness artifact records every lowering decision from RIF node to MicroOp
- **Cache-friendly** — Predictable memory access, no pointer chasing
- **Compact** — Generated kernels target L1i residency

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
│  Properties: Safe Rust. No unsafe. Deterministic.                   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE B: Lowering Engine                                           │
│  ─────────────────────────────────                                  │
│  • Lower RIF to MicroOps (instruction templates)                    │
│  • Allocate registers (linear scan, no spilling)                    │
│  • Emit x86-64 scalar machine code (JIT)                            │
│  • Generate Witness (per-MicroOp audit trail)                       │
│                                                                     │
│  Output: Executable kernel + Witness + SH_B                         │
│  Note: Witness records lowering decisions but does not              │
│        independently verify machine-code semantics.                 │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RUNTIME: Execution (Experimental)                                  │
│  ────────────────────────────                                       │
│  • io_uring packet ingress (partial, Linux only)                    │
│  • Version dispatcher (bounded jump table)                          │
│  • JIT kernel execution                                             │
│                                                                     │
│  Status: Scaffold. Not a complete ingress lifecycle.                │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase A: Trusted Computing Base

Safe Rust that can be audited.

- **RIF (Restricted Intermediate Form)** — DAG of typed operations with explicit memory regions and bounds. No pointers, recursion, or dynamic allocation.
- **Semantic Hash (SH_A)** — Hash of protocol semantics (currently uses a built-in hash implementation; audited crate replacement planned).

### Phase B: Lowering Engine

Translates RIF into machine code. Generates a witness artifact for auditability.

- **MicroOps** — Closed set of instruction templates with declared footprints
- **Witness** — Records which RIF node produced each MicroOp, with mask state snapshots. Checks internal consistency (entry count, mask monotonicity, hash integrity). Does not independently verify emitted machine code.
- **Register Allocation** — Linear scan with hard failure on pressure (no spilling)

### Runtime (Experimental)

Partial scaffold for packet processing. Not a complete data-plane.

- **io_uring** — Ring setup and CQ polling implemented; SQ submission and buffer recycling not complete
- **Version Dispatcher** — Bounded jump table routing packets to registered kernels
- **Slow Path** — Stub only; does not preserve packet semantics

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
✓ Benchmark PASSED — JIT output matches reference byte-for-byte
  Latency: ~21 ns/msg | Speedup: ~44x | Divergence: 0
```

---

## Witness & Replay

Compilation produces a witness artifact recording all lowering decisions.

```bash
# Emit witness
$ siphon compile protocols/market_data.siphon --emit-witness witness.json

# Verify witness internal consistency
$ siphon verify witness.json
✓ Witness verification PASSED
```

Current `verify` checks: entry count matches MicroOp count, mask monotonicity, hash integrity.
Not yet implemented: independent verification against protocol graph, machine-code disassembly verification.

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Protocol parser (Phase A) | Working. Field overlap detection not implemented. |
| RIF construction & validation | Working. |
| Semantic hash (SH_A) | Working. Uses built-in hash; audited crate planned. |
| Scalar JIT codegen (Phase B) | Working. Emits scalar x86-64 with stack-based vregs. Contains conditional branches for masked stores and prologue bounds check. |
| AVX2 SIMD backend | Encoder exists (`backend/x86_64.rs`) but is not integrated into the primary lowering path. |
| Batch (4-packet) kernel | Compiles but no correctness verification against reference. |
| Witness generation | Working. Self-attested; no independent verifier. |
| Witness verification (CLI) | Checks internal consistency only. Does not bind to protocol graph or disassemble code. |
| Deterministic replay | Replay seed derived and printed. Not used to drive codegen decisions. |
| io_uring runtime | Partial scaffold. CQ polling works; no complete RX/process/recycle lifecycle. |
| Version dispatcher | Working for registered kernels. Slow path is a no-op stub. |
| NUMA / core pinning | Documented as requirement. Not enforced in code. |
| Benchmark harness | Basic `Instant`-based timing. No core pinning, perf counters, or statistical rigor. |

---

## Current Limitations

Siphon is designed for fixed-layout binary protocols. It does **not** support:

- Variable-length fields
- Optional fields
- Recursive structures
- Dynamic schemas
- Non-x86 architectures (x86-64 only)

Additionally:

- No independent machine-code verification (witness is self-attested by the emitter)
- No cross-machine determinism validation
- No register spilling (hard failure on pressure)
- 16KB hard I-cache budget with no graceful degradation
- 512 RIF node cap, 4096 witness entry cap
- No production-grade benchmark methodology

---

## Building

```bash
cd cli && cargo build --release
cargo test
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
