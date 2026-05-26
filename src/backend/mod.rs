pub mod x86_64;

pub use x86_64::*;

use crate::lowering::target::{LaneMask, LoweringError, MicroOp};

/// Abstraction over code emission backends.
///
/// Implementations translate MicroOps into machine code bytes written to
/// an `ExecutableBuffer`. The lowering engine calls `emit_prologue`,
/// then `emit_microop` for each op, then `emit_epilogue`.
pub trait BackendEmitter {
    /// Human-readable backend name (e.g. "x86_64-scalar", "x86_64-avx2").
    fn name(&self) -> &'static str;

    /// Emit function prologue (bounds check, frame setup).
    fn emit_prologue(&self, code: &mut crate::lowering::jit::ExecutableBuffer, min_packet_len: u16) -> Result<(), LoweringError>;

    /// Emit machine code for a single MicroOp.
    fn emit_microop(&self, op: &MicroOp, code: &mut crate::lowering::jit::ExecutableBuffer) -> Result<(), LoweringError>;

    /// Emit function epilogue (frame teardown, return).
    fn emit_epilogue(&self, code: &mut crate::lowering::jit::ExecutableBuffer, final_mask: LaneMask) -> Result<(), LoweringError>;
}
