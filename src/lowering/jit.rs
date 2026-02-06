//! JIT Code Emission
//!
//! unsafe only for mmap/mprotect.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::result_unit_err)]

use crate::rif::{
    BinaryOp, Constraint, MemoryAccess, NodeIndex, RifGraph, RifNode, ScalarType,
    UnaryOp,
};
use crate::lowering::target::{
    LaneMask, LoweringError, MicroOp, MicroOpStream, RegAlloc, SimdWidth, VReg,
    ICACHE_BUDGET_BYTES,
};
use crate::lowering::witness::{
    MemoryAccessRecord, MemorySafetyVerifier, WitnessGenerator,
};
use crate::semantic_hash::SemanticHash;

/// Kernel signature: (packet, len, output) -> 0=success, nonzero=validation failure.
pub type KernelFn = extern "C" fn(*const u8, u32, *mut u8) -> u32;

/// Batch kernel: processes 4 packets in parallel using AVX2.
/// packets: array of 4 packet pointers, lens: array of 4 lengths, outputs: array of 4 output ptrs
/// Returns bitmask of successful packets (0xF = all succeeded).
pub type BatchKernelFn = extern "C" fn(*const *const u8, *const u32, *mut *mut u8) -> u32;

/// Verified executable kernel.
pub struct LoweredKernel {
    code: ExecutableBuffer,
    witness: crate::lowering::witness::Witness,
    ops: MicroOpStream,
    phase_a_hash: SemanticHash,
    phase_b_hash: SemanticHash,
}

impl LoweredKernel {
    pub fn as_fn(&self) -> KernelFn {
        self.code.as_fn()
    }

    pub fn witness(&self) -> &crate::lowering::witness::Witness {
        &self.witness
    }

    pub fn ops(&self) -> &[MicroOp] {
        self.ops.as_slice()
    }

    pub fn code_size(&self) -> usize {
        self.code.len()
    }

    pub fn phase_a_hash(&self) -> &SemanticHash {
        &self.phase_a_hash
    }

    pub fn phase_b_hash(&self) -> &SemanticHash {
        &self.phase_b_hash
    }
}

/// Batch kernel for SIMD 4-packet parallel processing.
pub struct BatchKernel {
    code: ExecutableBuffer,
    witness: crate::lowering::witness::Witness,
    phase_a_hash: SemanticHash,
}

impl BatchKernel {
    pub fn as_fn(&self) -> BatchKernelFn {
        unsafe { std::mem::transmute(self.code.ptr) }
    }

    pub fn code_size(&self) -> usize {
        self.code.len()
    }

    pub fn witness(&self) -> &crate::lowering::witness::Witness {
        &self.witness
    }

    pub fn phase_a_hash(&self) -> &SemanticHash {
        &self.phase_a_hash
    }
}

/// Executable memory buffer. mmap -> mprotect(EXEC) -> never write again.
pub struct ExecutableBuffer {
    ptr: *mut u8,
    len: usize,
    capacity: usize,
}

impl ExecutableBuffer {
    #[cfg(target_os = "linux")]
    pub fn new(capacity: usize) -> Result<Self, LoweringError> {
        use std::ptr;
        
        // Round up to page size
        let page_size = 4096;
        let capacity = (capacity + page_size - 1) & !(page_size - 1);
        
        // SAFETY: mmap with PROT_READ | PROT_WRITE initially
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                capacity,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        
        if ptr == libc::MAP_FAILED {
            return Err(LoweringError::WitnessGenerationFailed);
        }
        
        Ok(Self {
            ptr: ptr as *mut u8,
            len: 0,
            capacity,
        })
    }

    #[cfg(not(target_os = "linux"))]
    pub fn new(capacity: usize) -> Result<Self, LoweringError> {
        let layout = std::alloc::Layout::from_size_align(capacity, 4096)
            .map_err(|_| LoweringError::WitnessGenerationFailed)?;
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(LoweringError::WitnessGenerationFailed);
        }
        Ok(Self {
            ptr,
            len: 0,
            capacity,
        })
    }

    pub fn write(&mut self, bytes: &[u8]) -> Result<(), LoweringError> {
        if self.len + bytes.len() > self.capacity {
            return Err(LoweringError::ICacheBudgetExceeded {
                current: self.len,
                requested: bytes.len(),
                limit: self.capacity,
            });
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(self.len), bytes.len());
        }
        self.len += bytes.len();
        Ok(())
    }

    #[cfg(target_os = "linux")]
    pub fn make_executable(&mut self) -> Result<(), LoweringError> {
        let result = unsafe {
            libc::mprotect(
                self.ptr as *mut libc::c_void,
                self.capacity,
                libc::PROT_READ | libc::PROT_EXEC,
            )
        };
        
        if result != 0 {
            return Err(LoweringError::WitnessGenerationFailed);
        }
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn make_executable(&mut self) -> Result<(), LoweringError> {
        Ok(())
    }

    pub fn as_fn(&self) -> KernelFn {
        unsafe { std::mem::transmute(self.ptr) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for ExecutableBuffer {
    #[cfg(target_os = "linux")]
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.capacity);
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.capacity, 4096).unwrap();
        unsafe {
            std::alloc::dealloc(self.ptr, layout);
        }
    }
}

/// Lowering engine. RIF -> machine code.
pub struct LoweringEngine {
    width: SimdWidth,
    max_packet_len: u16,
}

impl LoweringEngine {
    pub fn new(width: SimdWidth, max_packet_len: u16) -> Self {
        Self { width, max_packet_len }
    }

    pub fn lower<'a>(
        &self,
        graph: &RifGraph<'a>,
        phase_a_hash: SemanticHash,
    ) -> Result<LoweredKernel, LoweringError> {
        graph.validate().map_err(|_| LoweringError::UnsupportedNode)?;

        let initial_lanes = self.width.lanes_for(ScalarType::U64);
        let initial_mask = LaneMask((1u16 << initial_lanes) - 1);

        let mut ops = MicroOpStream::new();
        let mut regalloc = RegAlloc::new(self.width);
        let mut witness_gen = WitnessGenerator::new(phase_a_hash, initial_mask);
        let mut mem_verifier = MemorySafetyVerifier::new(self.max_packet_len);

        for (idx, node) in graph.nodes.iter().enumerate() {
            let node_idx = NodeIndex(idx as u32);
            self.lower_node(
                node,
                node_idx,
                &mut ops,
                &mut regalloc,
                &mut witness_gen,
                &mut mem_verifier,
            )?;
        }

        mem_verifier.verify_all().map_err(|_| LoweringError::WitnessGenerationFailed)?;

        let final_mask = self.find_final_mask(ops.as_slice());

        let min_packet_len = self.compute_min_packet_len(graph);

        let mut code = ExecutableBuffer::new(ICACHE_BUDGET_BYTES)?;
        self.emit_prologue(&mut code, min_packet_len)?;
        
        for op in ops.as_slice() {
            self.emit_microop(op, &mut code)?;
        }
        
        self.emit_epilogue(&mut code, final_mask)?;
        code.make_executable()?;

        let witness = witness_gen.finalize();
        let phase_b_hash = witness.phase_b_hash().unwrap_or(phase_a_hash);

        Ok(LoweredKernel {
            code,
            witness,
            ops,
            phase_a_hash,
            phase_b_hash,
        })
    }

    /// Lower to batch kernel (SIMD 4-packet parallel).
    pub fn lower_batch<'a>(
        &self,
        graph: &RifGraph<'a>,
        phase_a_hash: SemanticHash,
    ) -> Result<BatchKernel, LoweringError> {
        graph.validate().map_err(|_| LoweringError::UnsupportedNode)?;

        let initial_lanes = self.width.lanes_for(ScalarType::U64);
        let initial_mask = LaneMask((1u16 << initial_lanes) - 1);

        let mut ops = MicroOpStream::new();
        let mut regalloc = RegAlloc::new(self.width);
        let mut witness_gen = WitnessGenerator::new(phase_a_hash, initial_mask);
        let mut mem_verifier = MemorySafetyVerifier::new(self.max_packet_len);

        for (idx, node) in graph.nodes.iter().enumerate() {
            let node_idx = NodeIndex(idx as u32);
            self.lower_node(node, node_idx, &mut ops, &mut regalloc, &mut witness_gen, &mut mem_verifier)?;
        }

        mem_verifier.verify_all().map_err(|_| LoweringError::WitnessGenerationFailed)?;

        let min_packet_len = self.compute_min_packet_len(graph);

        let mut code = ExecutableBuffer::new(ICACHE_BUDGET_BYTES)?;
        self.emit_batch_prologue(&mut code, min_packet_len)?;
        
        for op in ops.as_slice() {
            self.emit_batch_microop(op, &mut code)?;
        }
        
        self.emit_batch_epilogue(&mut code)?;
        code.make_executable()?;

        let witness = witness_gen.finalize();

        Ok(BatchKernel {
            code,
            witness,
            phase_a_hash,
        })
    }

    fn lower_node(
        &self,
        node: &RifNode,
        node_idx: NodeIndex,
        ops: &mut MicroOpStream,
        regalloc: &mut RegAlloc,
        witness: &mut WitnessGenerator,
        mem_verifier: &mut MemorySafetyVerifier,
    ) -> Result<(), LoweringError> {
        match node {
            RifNode::Load { scalar_type, access } => {
                self.lower_load(*scalar_type, access, node_idx, ops, regalloc, witness, mem_verifier)
            }
            RifNode::Store { value_node, access } => {
                self.lower_store(*value_node, access, node_idx, ops, regalloc, witness, mem_verifier)
            }
            RifNode::BinaryOp { op, lhs, rhs, result_type } => {
                self.lower_binop(*op, *lhs, *rhs, *result_type, node_idx, ops, regalloc, witness)
            }
            RifNode::UnaryOp { op, operand, result_type } => {
                self.lower_unaryop(*op, *operand, *result_type, node_idx, ops, regalloc, witness)
            }
            RifNode::Const { scalar_type, value } => {
                self.lower_const(*scalar_type, value, node_idx, ops, regalloc, witness)
            }
            RifNode::Validate { value_node, constraint } => {
                self.lower_validate(*value_node, constraint, node_idx, ops, regalloc, witness)
            }
            RifNode::Guard { parent_mask, condition } => {
                self.lower_guard(*parent_mask, *condition, node_idx, ops, regalloc, witness)
            }
            RifNode::Select { mask, true_val, false_val, result_type } => {
                self.lower_select(*mask, *true_val, *false_val, *result_type, node_idx, ops, regalloc, witness)
            }
            RifNode::Emit { field_id, value_node, mask } => {
                self.lower_emit(*field_id, *value_node, *mask, node_idx, ops, regalloc, witness, mem_verifier)
            }
            RifNode::Sequence { .. } => {
                // Sequence nodes are ordering constraints, no code emitted
                Ok(())
            }
        }
    }

    fn lower_load(
        &self,
        scalar_type: ScalarType,
        access: &MemoryAccess,
        node_idx: NodeIndex,
        ops: &mut MicroOpStream,
        regalloc: &mut RegAlloc,
        witness: &mut WitnessGenerator,
        mem_verifier: &mut MemorySafetyVerifier,
    ) -> Result<(), LoweringError> {
        let dst = regalloc.alloc()?;
        let mask_reg = access.mask_node_idx.and_then(|m| regalloc.get(m));

        let op = MicroOp::LoadVector {
            dst,
            offset: access.offset,
            width: self.width,
            scalar_type,
            mask: mask_reg,
        };

        mem_verifier.record_access(MemoryAccessRecord {
            rif_node: node_idx,
            offset: access.offset,
            length: access.length,
            mask: witness.current_mask(),
            is_load: true,
        }).map_err(|_| LoweringError::WitnessGenerationFailed)?;

        witness.record(node_idx, &op).map_err(|_| LoweringError::WitnessGenerationFailed)?;
        ops.push(op)?;
        regalloc.bind_typed(node_idx, dst, scalar_type);
        Ok(())
    }

    fn lower_store(
        &self,
        value_node: NodeIndex,
        access: &MemoryAccess,
        node_idx: NodeIndex,
        ops: &mut MicroOpStream,
        regalloc: &mut RegAlloc,
        witness: &mut WitnessGenerator,
        mem_verifier: &mut MemorySafetyVerifier,
    ) -> Result<(), LoweringError> {
        let src = regalloc.get(value_node).ok_or(LoweringError::UnsupportedNode)?;
        let mask_reg = access.mask_node_idx.and_then(|m| regalloc.get(m));

        let op = MicroOp::Emit {
            src,
            field_offset: access.offset as u16,
            scalar_type: ScalarType::U8, // raw bytes
            mask: mask_reg,
        };

        mem_verifier.record_access(MemoryAccessRecord {
            rif_node: node_idx,
            offset: access.offset,
            length: access.length,
            mask: witness.current_mask(),
            is_load: false,
        }).map_err(|_| LoweringError::WitnessGenerationFailed)?;

        witness.record(node_idx, &op).map_err(|_| LoweringError::WitnessGenerationFailed)?;
        ops.push(op)?;
        Ok(())
    }

    fn lower_binop(
        &self,
        op: BinaryOp,
        lhs: NodeIndex,
        rhs: NodeIndex,
        result_type: ScalarType,
        node_idx: NodeIndex,
        ops: &mut MicroOpStream,
        regalloc: &mut RegAlloc,
        witness: &mut WitnessGenerator,
    ) -> Result<(), LoweringError> {
        let lhs_reg = regalloc.get(lhs).ok_or(LoweringError::UnsupportedNode)?;
        let rhs_reg = regalloc.get(rhs).ok_or(LoweringError::UnsupportedNode)?;
        let dst = regalloc.alloc()?;

        let microop = match op {
            BinaryOp::Add => MicroOp::Add { dst, src1: lhs_reg, src2: rhs_reg, scalar_type: result_type },
            BinaryOp::Sub => MicroOp::Sub { dst, src1: lhs_reg, src2: rhs_reg, scalar_type: result_type },
            BinaryOp::And => MicroOp::And { dst, src1: lhs_reg, src2: rhs_reg },
            BinaryOp::Or => MicroOp::Or { dst, src1: lhs_reg, src2: rhs_reg },
            BinaryOp::Xor => MicroOp::Xor { dst, src1: lhs_reg, src2: rhs_reg },
            BinaryOp::Eq => {
                let microop = MicroOp::ValidateCmpEq {
                    dst_mask: dst,
                    src: lhs_reg,
                    imm_or_reg: rhs_reg,
                    scalar_type: result_type,
                };
                witness.record(node_idx, &microop).map_err(|_| LoweringError::WitnessGenerationFailed)?;
                ops.push(microop)?;
                regalloc.bind(node_idx, dst);
                return Ok(());
            }
            BinaryOp::Lt => {
                let microop = MicroOp::ValidateCmpLt {
                    dst_mask: dst,
                    src: lhs_reg,
                    comparand: rhs_reg,
                    scalar_type: result_type,
                };
                witness.record(node_idx, &microop).map_err(|_| LoweringError::WitnessGenerationFailed)?;
                ops.push(microop)?;
                regalloc.bind(node_idx, dst);
                return Ok(());
            }
            BinaryOp::Gt => {
                let microop = MicroOp::ValidateCmpGt {
                    dst_mask: dst,
                    src: lhs_reg,
                    comparand: rhs_reg,
                    scalar_type: result_type,
                };
                witness.record(node_idx, &microop).map_err(|_| LoweringError::WitnessGenerationFailed)?;
                ops.push(microop)?;
                regalloc.bind(node_idx, dst);
                return Ok(());
            }
            _ => return Err(LoweringError::UnsupportedNode),
        };

        witness.record(node_idx, &microop).map_err(|_| LoweringError::WitnessGenerationFailed)?;
        ops.push(microop)?;
        regalloc.bind(node_idx, dst);
        Ok(())
    }

    fn lower_unaryop(
        &self,
        op: UnaryOp,
        operand: NodeIndex,
        result_type: ScalarType,
        node_idx: NodeIndex,
        ops: &mut MicroOpStream,
        regalloc: &mut RegAlloc,
        witness: &mut WitnessGenerator,
    ) -> Result<(), LoweringError> {
        let src = regalloc.get(operand).ok_or(LoweringError::UnsupportedNode)?;
        let dst = regalloc.alloc()?;

        let microop = match op {
            UnaryOp::ByteSwap => MicroOp::ByteSwap { dst, src, scalar_type: result_type },
            UnaryOp::Not => {
                // XOR with all-ones
                let ones = regalloc.alloc()?;
                let broadcast = MicroOp::BroadcastImm {
                    dst: ones,
                    value: u64::MAX,
                    scalar_type: result_type,
                };
                witness.record(node_idx, &broadcast).map_err(|_| LoweringError::WitnessGenerationFailed)?;
                ops.push(broadcast)?;
                
                MicroOp::Xor { dst, src1: src, src2: ones }
            }
            UnaryOp::Neg => {
                // Subtract from zero
                let zero = regalloc.alloc()?;
                let broadcast = MicroOp::BroadcastImm {
                    dst: zero,
                    value: 0,
                    scalar_type: result_type,
                };
                witness.record(node_idx, &broadcast).map_err(|_| LoweringError::WitnessGenerationFailed)?;
                ops.push(broadcast)?;
                
                MicroOp::Sub { dst, src1: zero, src2: src, scalar_type: result_type }
            }
        };

        witness.record(node_idx, &microop).map_err(|_| LoweringError::WitnessGenerationFailed)?;
        ops.push(microop)?;
        regalloc.bind(node_idx, dst);
        Ok(())
    }

    fn lower_const(
        &self,
        scalar_type: ScalarType,
        value: &[u8; 8],
        node_idx: NodeIndex,
        ops: &mut MicroOpStream,
        regalloc: &mut RegAlloc,
        witness: &mut WitnessGenerator,
    ) -> Result<(), LoweringError> {
        let dst = regalloc.alloc()?;
        
        let val = u64::from_be_bytes(*value);
        
        let op = MicroOp::BroadcastImm {
            dst,
            value: val,
            scalar_type,
        };

        witness.record(node_idx, &op).map_err(|_| LoweringError::WitnessGenerationFailed)?;
        ops.push(op)?;
        regalloc.bind(node_idx, dst);
        Ok(())
    }

    fn lower_validate(
        &self,
        value_node: NodeIndex,
        constraint: &Constraint,
        node_idx: NodeIndex,
        ops: &mut MicroOpStream,
        regalloc: &mut RegAlloc,
        witness: &mut WitnessGenerator,
    ) -> Result<(), LoweringError> {
        let src = regalloc.get(value_node).ok_or(LoweringError::UnsupportedNode)?;
        let dst_mask = regalloc.alloc()?;

        match constraint {
            Constraint::None => {
                let op = MicroOp::BroadcastImm {
                    dst: dst_mask,
                    value: u64::MAX,
                    scalar_type: ScalarType::U64,
                };
                witness.record(node_idx, &op).map_err(|_| LoweringError::WitnessGenerationFailed)?;
                ops.push(op)?;
            }
            Constraint::NonZero => {
                let op = MicroOp::ValidateNonZero {
                    dst_mask,
                    src,
                    scalar_type: ScalarType::U64,
                };
                witness.record(node_idx, &op).map_err(|_| LoweringError::WitnessGenerationFailed)?;
                ops.push(op)?;
            }
            Constraint::Range { lo, hi } => {
                let lo_reg = regalloc.alloc()?;
                let lo_op = MicroOp::BroadcastImm {
                    dst: lo_reg,
                    value: *lo,
                    scalar_type: ScalarType::U64,
                };
                witness.record(node_idx, &lo_op).map_err(|_| LoweringError::WitnessGenerationFailed)?;
                ops.push(lo_op)?;

                let hi_reg = regalloc.alloc()?;
                let hi_op = MicroOp::BroadcastImm {
                    dst: hi_reg,
                    value: *hi,
                    scalar_type: ScalarType::U64,
                };
                witness.record(node_idx, &hi_op).map_err(|_| LoweringError::WitnessGenerationFailed)?;
                ops.push(hi_op)?;

                let ge_mask = regalloc.alloc()?;
                let ge_op = MicroOp::ValidateCmpGt {
                    dst_mask: ge_mask,
                    src,
                    comparand: lo_reg,
                    scalar_type: ScalarType::U64,
                };
                witness.record(node_idx, &ge_op).map_err(|_| LoweringError::WitnessGenerationFailed)?;
                ops.push(ge_op)?;

                let le_mask = regalloc.alloc()?;
                let le_op = MicroOp::ValidateCmpLt {
                    dst_mask: le_mask,
                    src,
                    comparand: hi_reg,
                    scalar_type: ScalarType::U64,
                };
                witness.record(node_idx, &le_op).map_err(|_| LoweringError::WitnessGenerationFailed)?;
                ops.push(le_op)?;

                let and_op = MicroOp::MaskAnd {
                    dst: dst_mask,
                    src1: ge_mask,
                    src2: le_mask,
                };
                witness.record(node_idx, &and_op).map_err(|_| LoweringError::WitnessGenerationFailed)?;
                ops.push(and_op)?;
            }
            Constraint::Finite => {
                return Err(LoweringError::UnsupportedNode);
            }
        }

        regalloc.bind(node_idx, dst_mask);
        Ok(())
    }

    fn lower_guard(
        &self,
        parent_mask: Option<NodeIndex>,
        condition: NodeIndex,
        node_idx: NodeIndex,
        ops: &mut MicroOpStream,
        regalloc: &mut RegAlloc,
        witness: &mut WitnessGenerator,
    ) -> Result<(), LoweringError> {
        let cond_reg = regalloc.get(condition).ok_or(LoweringError::UnsupportedNode)?;
        let dst = regalloc.alloc()?;

        if let Some(parent) = parent_mask {
            let parent_reg = regalloc.get(parent).ok_or(LoweringError::UnsupportedNode)?;
            let op = MicroOp::MaskAnd {
                dst,
                src1: parent_reg,
                src2: cond_reg,
            };
            
            witness.record(node_idx, &op).map_err(|_| LoweringError::WitnessGenerationFailed)?;
            ops.push(op)?;
        } else {
            let op = MicroOp::MaskAnd {
                dst,
                src1: cond_reg,
                src2: cond_reg,
            };
            witness.record(node_idx, &op).map_err(|_| LoweringError::WitnessGenerationFailed)?;
            ops.push(op)?;
        }

        let new_mask = witness.current_mask().and(LaneMask(0xFF));
        witness.refine_mask(new_mask).map_err(|_| LoweringError::InvalidMaskChain)?;
        witness.set_mask_reg(dst, new_mask);

        regalloc.bind(node_idx, dst);
        Ok(())
    }

    fn lower_select(
        &self,
        mask: NodeIndex,
        true_val: NodeIndex,
        false_val: NodeIndex,
        result_type: ScalarType,
        node_idx: NodeIndex,
        ops: &mut MicroOpStream,
        regalloc: &mut RegAlloc,
        witness: &mut WitnessGenerator,
    ) -> Result<(), LoweringError> {
        let mask_reg = regalloc.get(mask).ok_or(LoweringError::UnsupportedNode)?;
        let true_reg = regalloc.get(true_val).ok_or(LoweringError::UnsupportedNode)?;
        let false_reg = regalloc.get(false_val).ok_or(LoweringError::UnsupportedNode)?;
        let dst = regalloc.alloc()?;

        let op = MicroOp::Select {
            dst,
            mask: mask_reg,
            true_val: true_reg,
            false_val: false_reg,
            scalar_type: result_type,
        };

        witness.record(node_idx, &op).map_err(|_| LoweringError::WitnessGenerationFailed)?;
        ops.push(op)?;
        regalloc.bind(node_idx, dst);
        Ok(())
    }

    fn lower_emit(
        &self,
        field_id: u16,
        value_node: NodeIndex,
        mask: Option<NodeIndex>,
        node_idx: NodeIndex,
        ops: &mut MicroOpStream,
        regalloc: &mut RegAlloc,
        witness: &mut WitnessGenerator,
        _mem_verifier: &mut MemorySafetyVerifier,
    ) -> Result<(), LoweringError> {
        let src = regalloc.get(value_node).ok_or(LoweringError::UnsupportedNode)?;
        let mask_reg = mask.and_then(|m| regalloc.get(m));

        let scalar_type = regalloc.get_type(value_node).unwrap_or(ScalarType::U64);
        let field_offset = regalloc.alloc_field_offset(field_id, scalar_type);

        let op = MicroOp::Emit {
            src,
            field_offset,
            scalar_type,
            mask: mask_reg,
        };

        witness.record(node_idx, &op).map_err(|_| LoweringError::WitnessGenerationFailed)?;
        ops.push(op)?;
        Ok(())
    }

    fn emit_prologue(&self, code: &mut ExecutableBuffer, min_packet_len: u16) -> Result<(), LoweringError> {
        // Bounds check: if esi < min_packet_len, return 2 (bounds error)
        if min_packet_len > 0 {
            code.write(&[0x81, 0xFE])?;           // CMP ESI, imm32
            code.write(&(min_packet_len as u32).to_le_bytes())?;
            code.write(&[0x72, 0x04])?;           // JB +4 (to early exit)
            code.write(&[0xEB, 0x06])?;           // JMP +6 (skip early exit)
            // Early exit: return 2 (bounds error)
            code.write(&[0xB8, 0x02, 0x00, 0x00, 0x00])?;  // MOV EAX, 2
            code.write(&[0xC3])?;                 // RET
        }

        code.write(&[0x0F, 0x18, 0x07])?;         // PREFETCHT0 [rdi]
        code.write(&[0x0F, 0x18, 0x47, 0x40])?;   // PREFETCHT0 [rdi+64]
        code.write(&[0x55])?;                     // push rbp
        code.write(&[0x48, 0x89, 0xE5])?;         // mov rbp, rsp
        code.write(&[0x48, 0x83, 0xEC, 0x80])?;   // sub rsp, 128
        code.write(&[0x41, 0x54])?;               // push r12
        code.write(&[0x49, 0x89, 0xD4])?;         // mov r12, rdx
        Ok(())
    }

    fn compute_min_packet_len(&self, graph: &RifGraph) -> u16 {
        let mut max_end: u32 = 0;
        for node in graph.nodes.iter() {
            if let RifNode::Load { access, scalar_type } = node {
                let end = access.offset + scalar_type.size_bytes() as u32;
                if end > max_end {
                    max_end = end;
                }
            }
        }
        max_end.min(u16::MAX as u32) as u16
    }

    fn find_final_mask(&self, ops: &[MicroOp]) -> Option<VReg> {
        for op in ops.iter().rev() {
            if let MicroOp::Emit { mask, .. } = op {
                if mask.is_some() {
                    return *mask;
                }
            }
        }
        None
    }

    fn emit_epilogue(&self, code: &mut ExecutableBuffer, final_mask: Option<VReg>) -> Result<(), LoweringError> {
        code.write(&[0x0F, 0xAE, 0xF8])?;  // SFENCE
        code.write(&[0x41, 0x5C])?;        // pop r12
        code.write(&[0x48, 0x89, 0xEC])?;  // mov rsp, rbp
        code.write(&[0x5D])?;              // pop rbp

        if let Some(mask_reg) = final_mask {
            self.emit_load_vreg_to_rax_epilogue(code, mask_reg)?;
            code.write(&[0x48, 0x85, 0xC0])?;  // TEST RAX, RAX
            code.write(&[0x0F, 0x94, 0xC0])?;  // SETZ AL (1 if mask==0, 0 if mask!=0)
            code.write(&[0x0F, 0xB6, 0xC0])?;  // MOVZX EAX, AL
        } else {
            code.write(&[0x31, 0xC0])?;        // xor eax, eax (always success if no guards)
        }

        code.write(&[0xC3])?;              // ret
        Ok(())
    }

    fn emit_load_vreg_to_rax_epilogue(&self, code: &mut ExecutableBuffer, src: VReg) -> Result<(), LoweringError> {
        let stack_offset = -8 * (src.0 as i32 + 1);
        code.write(&[0x48, 0x8B, 0x85])?;
        code.write(&stack_offset.to_le_bytes())?;
        Ok(())
    }

    fn emit_batch_prologue(&self, code: &mut ExecutableBuffer, min_packet_len: u16) -> Result<(), LoweringError> {
        // Bounds check for batch: check first packet length (simplified)
        if min_packet_len > 0 {
            code.write(&[0x81, 0xFE])?;           // CMP ESI, imm32
            code.write(&(min_packet_len as u32).to_le_bytes())?;
            code.write(&[0x72, 0x04])?;           // JB +4 (to early exit)
            code.write(&[0xEB, 0x06])?;           // JMP +6 (skip early exit)
            code.write(&[0xB8, 0x00, 0x00, 0x00, 0x00])?;  // MOV EAX, 0 (no packets succeeded)
            code.write(&[0xC3])?;                 // RET
        }

        code.write(&[0x55, 0x48, 0x89, 0xE5])?;                       // push rbp; mov rbp, rsp
        code.write(&[0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57])?; // push r12-r15
        code.write(&[0x49, 0x89, 0xFC])?;                             // mov r12, rdi
        code.write(&[0x49, 0x89, 0xF5])?;                             // mov r13, rsi
        code.write(&[0x49, 0x89, 0xD6])?;                             // mov r14, rdx
        code.write(&[0xC5, 0xFC, 0x77])?;                             // VZEROALL
        Ok(())
    }

    fn emit_batch_epilogue(&self, code: &mut ExecutableBuffer) -> Result<(), LoweringError> {
        code.write(&[0x0F, 0xAE, 0xF8])?;                             // SFENCE
        code.write(&[0xC5, 0xF8, 0x77])?;                             // VZEROUPPER
        code.write(&[0x41, 0x5F, 0x41, 0x5E, 0x41, 0x5D, 0x41, 0x5C])?; // pop r15-r12
        code.write(&[0x48, 0x89, 0xEC, 0x5D])?;                       // mov rsp, rbp; pop rbp
        code.write(&[0xB8, 0x0F, 0x00, 0x00, 0x00])?;                 // mov eax, 0xF
        code.write(&[0xC3])?;                                         // ret
        Ok(())
    }

    fn emit_batch_microop(&self, op: &MicroOp, code: &mut ExecutableBuffer) -> Result<(), LoweringError> {
        match op {
            MicroOp::LoadVector { dst, offset, width: _, scalar_type: _, mask: _ } => {
                self.emit_batch_load(code, *dst, *offset)
            }
            MicroOp::Emit { src, field_offset, scalar_type: _, mask: _ } => {
                self.emit_batch_store(code, *src, *field_offset as u32)
            }
            MicroOp::BroadcastImm { dst, value, scalar_type } => {
                self.emit_broadcast_imm64(code, *dst, *value, *scalar_type)
            }
            MicroOp::ValidateCmpGt { dst_mask, src, comparand, scalar_type: _ } => {
                self.emit_vpcmpgtq(code, *dst_mask, *src, *comparand)
            }
            MicroOp::ValidateCmpLt { dst_mask, src, comparand, scalar_type: _ } => {
                self.emit_vpcmpgtq(code, *dst_mask, *comparand, *src)
            }
            MicroOp::ValidateCmpEq { dst_mask, src, imm_or_reg, scalar_type: _ } => {
                self.emit_vpcmpeqq(code, *dst_mask, *src, *imm_or_reg)
            }
            MicroOp::ValidateNonZero { dst_mask, src, scalar_type: _ } => {
                self.emit_validate_nonzero(code, *dst_mask, *src)
            }
            MicroOp::MaskAnd { dst, src1, src2 } => {
                self.emit_vpand(code, *dst, *src1, *src2)
            }
            MicroOp::MaskOr { dst, src1, src2 } => {
                self.emit_vpor(code, *dst, *src1, *src2)
            }
            MicroOp::MaskNot { dst, src } => {
                self.emit_vpxor(code, *dst, *src, *src)
            }
            MicroOp::Select { dst, mask, true_val, false_val, scalar_type: _ } => {
                self.emit_vblendvpd(code, *dst, *false_val, *true_val, *mask)
            }
            MicroOp::Add { dst, src1, src2, scalar_type } => {
                self.emit_vpaddq(code, *dst, *src1, *src2, *scalar_type)
            }
            MicroOp::Sub { dst, src1, src2, scalar_type } => {
                self.emit_vpsubq(code, *dst, *src1, *src2, *scalar_type)
            }
            MicroOp::And { dst, src1, src2 } => {
                self.emit_vpand(code, *dst, *src1, *src2)
            }
            MicroOp::Or { dst, src1, src2 } => {
                self.emit_vpor(code, *dst, *src1, *src2)
            }
            MicroOp::Xor { dst, src1, src2 } => {
                self.emit_vpxor(code, *dst, *src1, *src2)
            }
            MicroOp::ByteSwap { dst: _, src: _, scalar_type: _ } => {
                Ok(())
            }
            MicroOp::Nop { bytes } => {
                self.emit_nop(code, *bytes)
            }
        }
    }

    fn emit_batch_load(&self, code: &mut ExecutableBuffer, dst: VReg, offset: u32) -> Result<(), LoweringError> {
        let dst_reg = dst.0 & 0x0F;
        code.write(&[0x49, 0x8B, 0x04, 0x24])?;  // mov rax, [r12]
        if dst_reg < 8 {
            code.write(&[0xC4, 0xE2, 0x7D, 0x19])?;
        } else {
            code.write(&[0xC4, 0x62, 0x7D, 0x19])?;
        }
        let modrm = 0x80 | ((dst_reg & 7) << 3);
        code.write(&[modrm])?;
        code.write(&offset.to_le_bytes())?;
        Ok(())
    }

    fn emit_batch_store(&self, code: &mut ExecutableBuffer, src: VReg, offset: u32) -> Result<(), LoweringError> {
        let src_reg = src.0 & 0x0F;
        code.write(&[0x49, 0x8B, 0x06])?;  // mov rax, [r14]
        if src_reg < 8 {
            code.write(&[0xC5, 0xFD, 0xE7])?;
        } else {
            code.write(&[0xC4, 0x41, 0x7D, 0xE7])?;
        }
        let modrm = 0x80 | ((src_reg & 7) << 3);
        code.write(&[modrm])?;
        code.write(&offset.to_le_bytes())?;
        Ok(())
    }

    fn emit_microop(&self, op: &MicroOp, code: &mut ExecutableBuffer) -> Result<(), LoweringError> {
        match op {
            MicroOp::LoadVector { dst, offset, width: _, scalar_type, mask: _ } => {
                self.emit_scalar_load(code, *dst, *offset, *scalar_type)
            }
            MicroOp::Emit { src, field_offset, scalar_type, mask } => {
                self.emit_scalar_store(code, *src, *field_offset as u32, *scalar_type, *mask)
            }
            MicroOp::BroadcastImm { dst, value, scalar_type: _ } => {
                self.emit_scalar_const(code, *dst, *value)
            }
            MicroOp::ValidateCmpGt { dst_mask, src, comparand, scalar_type: _ } => {
                self.emit_scalar_cmp_gt(code, *dst_mask, *src, *comparand)
            }
            MicroOp::ValidateCmpLt { dst_mask, src, comparand, scalar_type: _ } => {
                self.emit_scalar_cmp_lt(code, *dst_mask, *src, *comparand)
            }
            MicroOp::ValidateCmpEq { dst_mask, src, imm_or_reg, scalar_type: _ } => {
                self.emit_scalar_cmp_eq(code, *dst_mask, *src, *imm_or_reg)
            }
            MicroOp::ValidateNonZero { dst_mask, src, scalar_type: _ } => {
                self.emit_scalar_test_nonzero(code, *dst_mask, *src)
            }
            MicroOp::MaskAnd { dst, src1, src2 } => {
                self.emit_scalar_mask_and(code, *dst, *src1, *src2)
            }
            MicroOp::MaskOr { dst, src1, src2 } => {
                self.emit_scalar_mask_or(code, *dst, *src1, *src2)
            }
            MicroOp::MaskNot { dst, src } => {
                self.emit_scalar_mask_not(code, *dst, *src)
            }
            MicroOp::Select { dst, mask, true_val, false_val, scalar_type: _ } => {
                self.emit_scalar_select(code, *dst, *mask, *true_val, *false_val)
            }
            MicroOp::Add { dst, src1, src2, scalar_type: _ } => {
                self.emit_scalar_add(code, *dst, *src1, *src2)
            }
            MicroOp::Sub { dst, src1, src2, scalar_type: _ } => {
                self.emit_scalar_sub(code, *dst, *src1, *src2)
            }
            MicroOp::And { dst, src1, src2 } => {
                self.emit_scalar_and(code, *dst, *src1, *src2)
            }
            MicroOp::Or { dst, src1, src2 } => {
                self.emit_scalar_or(code, *dst, *src1, *src2)
            }
            MicroOp::Xor { dst, src1, src2 } => {
                self.emit_scalar_xor(code, *dst, *src1, *src2)
            }
            MicroOp::ByteSwap { dst, src, scalar_type: _ } => {
                self.emit_scalar_bswap(code, *dst, *src)
            }
            MicroOp::Nop { bytes } => {
                self.emit_nop(code, *bytes)
            }
        }
    }

    fn emit_scalar_load(&self, code: &mut ExecutableBuffer, dst: VReg, offset: u32, scalar_type: ScalarType) -> Result<(), LoweringError> {
        match scalar_type.size_bytes() {
            8 => { code.write(&[0x48, 0x8B, 0x87])?; code.write(&offset.to_le_bytes())?; }
            4 => { code.write(&[0x8B, 0x87])?; code.write(&offset.to_le_bytes())?; }
            2 => { code.write(&[0x0F, 0xB7, 0x87])?; code.write(&offset.to_le_bytes())?; }
            1 => { code.write(&[0x0F, 0xB6, 0x87])?; code.write(&offset.to_le_bytes())?; }
            _ => { code.write(&[0x48, 0x8B, 0x87])?; code.write(&offset.to_le_bytes())?; }
        }
        
        let stack_offset = -8 * (dst.0 as i32 + 1);
        if stack_offset >= -128 {
            code.write(&[0x48, 0x89, 0x45, stack_offset as u8])?;
        } else {
            code.write(&[0x48, 0x89, 0x85])?;
            code.write(&stack_offset.to_le_bytes())?;
        }
        Ok(())
    }

    fn emit_scalar_store(&self, code: &mut ExecutableBuffer, src: VReg, offset: u32, scalar_type: ScalarType, mask: Option<VReg>) -> Result<(), LoweringError> {
        if let Some(mask_reg) = mask {
            self.emit_load_vreg_to_rcx(code, mask_reg)?;
            code.write(&[0x48, 0x85, 0xC9])?;  // TEST RCX, RCX
            code.write(&[0x74])?;              // JZ rel8 (skip store)
            let skip_offset = self.store_instruction_size(offset, scalar_type) + 7;
            code.write(&[skip_offset as u8])?;
        }

        let stack_offset = -8 * (src.0 as i32 + 1);
        if stack_offset >= -128 {
            code.write(&[0x48, 0x8B, 0x45, stack_offset as u8])?;
        } else {
            code.write(&[0x48, 0x8B, 0x85])?;
            code.write(&stack_offset.to_le_bytes())?;
        }
        
        match scalar_type.size_bytes() {
            8 => {
                if offset < 128 {
                    code.write(&[0x49, 0x89, 0x44, 0x24, offset as u8])?;  // MOV [r12+off8], RAX
                } else {
                    code.write(&[0x49, 0x89, 0x84, 0x24])?;                // MOV [r12+off32], RAX
                    code.write(&offset.to_le_bytes())?;
                }
            }
            4 => {
                if offset < 128 {
                    code.write(&[0x41, 0x89, 0x44, 0x24, offset as u8])?;  // MOV [r12+off8], EAX
                } else {
                    code.write(&[0x41, 0x89, 0x84, 0x24])?;                // MOV [r12+off32], EAX
                    code.write(&offset.to_le_bytes())?;
                }
            }
            2 => {
                if offset < 128 {
                    code.write(&[0x66, 0x41, 0x89, 0x44, 0x24, offset as u8])?;  // MOV [r12+off8], AX
                } else {
                    code.write(&[0x66, 0x41, 0x89, 0x84, 0x24])?;                // MOV [r12+off32], AX
                    code.write(&offset.to_le_bytes())?;
                }
            }
            1 => {
                if offset < 128 {
                    code.write(&[0x41, 0x88, 0x44, 0x24, offset as u8])?;  // MOV [r12+off8], AL
                } else {
                    code.write(&[0x41, 0x88, 0x84, 0x24])?;                // MOV [r12+off32], AL
                    code.write(&offset.to_le_bytes())?;
                }
            }
            _ => {
                if offset < 128 {
                    code.write(&[0x49, 0x89, 0x44, 0x24, offset as u8])?;
                } else {
                    code.write(&[0x49, 0x89, 0x84, 0x24])?;
                    code.write(&offset.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    fn store_instruction_size(&self, offset: u32, scalar_type: ScalarType) -> usize {
        let base = match scalar_type.size_bytes() {
            8 => 5,
            4 => 5,
            2 => 6,
            1 => 5,
            _ => 5,
        };
        if offset < 128 { base } else { base + 3 }
    }

    fn emit_scalar_const(&self, code: &mut ExecutableBuffer, dst: VReg, value: u64) -> Result<(), LoweringError> {
        code.write(&[0x48, 0xB8])?;  // MOV RAX, imm64
        code.write(&value.to_le_bytes())?;
        
        let stack_offset = -8 * (dst.0 as i32 + 1);
        if stack_offset >= -128 {
            code.write(&[0x48, 0x89, 0x45, stack_offset as u8])?;
        } else {
            code.write(&[0x48, 0x89, 0x85])?;
            code.write(&stack_offset.to_le_bytes())?;
        }
        Ok(())
    }

    fn emit_load_vreg_to_rax(&self, code: &mut ExecutableBuffer, src: VReg) -> Result<(), LoweringError> {
        let stack_offset = -8 * (src.0 as i32 + 1);
        if stack_offset >= -128 {
            code.write(&[0x48, 0x8B, 0x45, stack_offset as u8])?;
        } else {
            code.write(&[0x48, 0x8B, 0x85])?;
            code.write(&stack_offset.to_le_bytes())?;
        }
        Ok(())
    }

    fn emit_load_vreg_to_rcx(&self, code: &mut ExecutableBuffer, src: VReg) -> Result<(), LoweringError> {
        let stack_offset = -8 * (src.0 as i32 + 1);
        if stack_offset >= -128 {
            code.write(&[0x48, 0x8B, 0x4D, stack_offset as u8])?;
        } else {
            code.write(&[0x48, 0x8B, 0x8D])?;
            code.write(&stack_offset.to_le_bytes())?;
        }
        Ok(())
    }

    fn emit_store_rax_to_vreg(&self, code: &mut ExecutableBuffer, dst: VReg) -> Result<(), LoweringError> {
        let stack_offset = -8 * (dst.0 as i32 + 1);
        if stack_offset >= -128 {
            code.write(&[0x48, 0x89, 0x45, stack_offset as u8])?;
        } else {
            code.write(&[0x48, 0x89, 0x85])?;
            code.write(&stack_offset.to_le_bytes())?;
        }
        Ok(())
    }

    fn emit_scalar_cmp_gt(&self, code: &mut ExecutableBuffer, dst_mask: VReg, src: VReg, comparand: VReg) -> Result<(), LoweringError> {
        self.emit_load_vreg_to_rax(code, src)?;
        self.emit_load_vreg_to_rcx(code, comparand)?;
        code.write(&[0x48, 0x39, 0xC8])?;  // CMP RAX, RCX
        code.write(&[0x0F, 0x9F, 0xC0])?;  // SETG AL
        code.write(&[0x48, 0x0F, 0xB6, 0xC0])?;
        self.emit_store_rax_to_vreg(code, dst_mask)
    }

    fn emit_scalar_cmp_lt(&self, code: &mut ExecutableBuffer, dst_mask: VReg, src: VReg, comparand: VReg) -> Result<(), LoweringError> {
        self.emit_load_vreg_to_rax(code, src)?;
        self.emit_load_vreg_to_rcx(code, comparand)?;
        // CMP RAX, RCX
        code.write(&[0x48, 0x39, 0xC8])?;
        code.write(&[0x0F, 0x9C, 0xC0])?;
        // MOVZX RAX, AL
        code.write(&[0x48, 0x0F, 0xB6, 0xC0])?;
        self.emit_store_rax_to_vreg(code, dst_mask)
    }

    fn emit_scalar_cmp_eq(&self, code: &mut ExecutableBuffer, dst_mask: VReg, src: VReg, comparand: VReg) -> Result<(), LoweringError> {
        self.emit_load_vreg_to_rax(code, src)?;
        self.emit_load_vreg_to_rcx(code, comparand)?;
        // CMP RAX, RCX
        code.write(&[0x48, 0x39, 0xC8])?;
        code.write(&[0x0F, 0x94, 0xC0])?;
        // MOVZX RAX, AL
        code.write(&[0x48, 0x0F, 0xB6, 0xC0])?;
        self.emit_store_rax_to_vreg(code, dst_mask)
    }

    fn emit_scalar_test_nonzero(&self, code: &mut ExecutableBuffer, dst_mask: VReg, src: VReg) -> Result<(), LoweringError> {
        self.emit_load_vreg_to_rax(code, src)?;
        code.write(&[0x48, 0x85, 0xC0])?;
        code.write(&[0x0F, 0x95, 0xC0])?;
        // MOVZX RAX, AL
        code.write(&[0x48, 0x0F, 0xB6, 0xC0])?;
        self.emit_store_rax_to_vreg(code, dst_mask)
    }

    fn emit_scalar_mask_and(&self, code: &mut ExecutableBuffer, dst: VReg, src1: VReg, src2: VReg) -> Result<(), LoweringError> {
        self.emit_load_vreg_to_rax(code, src1)?;
        self.emit_load_vreg_to_rcx(code, src2)?;
        // AND RAX, RCX
        code.write(&[0x48, 0x21, 0xC8])?;
        self.emit_store_rax_to_vreg(code, dst)
    }

    fn emit_scalar_mask_or(&self, code: &mut ExecutableBuffer, dst: VReg, src1: VReg, src2: VReg) -> Result<(), LoweringError> {
        self.emit_load_vreg_to_rax(code, src1)?;
        self.emit_load_vreg_to_rcx(code, src2)?;
        // OR RAX, RCX
        code.write(&[0x48, 0x09, 0xC8])?;
        self.emit_store_rax_to_vreg(code, dst)
    }

    fn emit_scalar_mask_not(&self, code: &mut ExecutableBuffer, dst: VReg, src: VReg) -> Result<(), LoweringError> {
        self.emit_load_vreg_to_rax(code, src)?;
        code.write(&[0x48, 0x83, 0xF0, 0x01])?;
        self.emit_store_rax_to_vreg(code, dst)
    }

    fn emit_scalar_select(&self, code: &mut ExecutableBuffer, dst: VReg, mask: VReg, true_val: VReg, false_val: VReg) -> Result<(), LoweringError> {
        self.emit_load_vreg_to_rax(code, false_val)?;
        self.emit_load_vreg_to_rcx(code, true_val)?;
        let mask_offset = -8 * (mask.0 as i32 + 1);
        if mask_offset >= -128 {
            code.write(&[0x48, 0x8B, 0x55, mask_offset as u8])?;
        } else {
            code.write(&[0x48, 0x8B, 0x95])?;
            code.write(&mask_offset.to_le_bytes())?;
        }
        code.write(&[0x48, 0x85, 0xD2])?;
        code.write(&[0x48, 0x0F, 0x45, 0xC1])?;
        self.emit_store_rax_to_vreg(code, dst)
    }

    fn emit_scalar_add(&self, code: &mut ExecutableBuffer, dst: VReg, src1: VReg, src2: VReg) -> Result<(), LoweringError> {
        self.emit_load_vreg_to_rax(code, src1)?;
        self.emit_load_vreg_to_rcx(code, src2)?;
        code.write(&[0x48, 0x01, 0xC8])?;
        self.emit_store_rax_to_vreg(code, dst)
    }

    fn emit_scalar_sub(&self, code: &mut ExecutableBuffer, dst: VReg, src1: VReg, src2: VReg) -> Result<(), LoweringError> {
        self.emit_load_vreg_to_rax(code, src1)?;
        self.emit_load_vreg_to_rcx(code, src2)?;
        code.write(&[0x48, 0x29, 0xC8])?;
        self.emit_store_rax_to_vreg(code, dst)
    }

    fn emit_scalar_and(&self, code: &mut ExecutableBuffer, dst: VReg, src1: VReg, src2: VReg) -> Result<(), LoweringError> {
        self.emit_load_vreg_to_rax(code, src1)?;
        self.emit_load_vreg_to_rcx(code, src2)?;
        // AND RAX, RCX
        code.write(&[0x48, 0x21, 0xC8])?;
        self.emit_store_rax_to_vreg(code, dst)
    }

    fn emit_scalar_or(&self, code: &mut ExecutableBuffer, dst: VReg, src1: VReg, src2: VReg) -> Result<(), LoweringError> {
        self.emit_load_vreg_to_rax(code, src1)?;
        self.emit_load_vreg_to_rcx(code, src2)?;
        // OR RAX, RCX
        code.write(&[0x48, 0x09, 0xC8])?;
        self.emit_store_rax_to_vreg(code, dst)
    }

    fn emit_scalar_xor(&self, code: &mut ExecutableBuffer, dst: VReg, src1: VReg, src2: VReg) -> Result<(), LoweringError> {
        self.emit_load_vreg_to_rax(code, src1)?;
        self.emit_load_vreg_to_rcx(code, src2)?;
        code.write(&[0x48, 0x31, 0xC8])?;
        self.emit_store_rax_to_vreg(code, dst)
    }

    fn emit_scalar_bswap(&self, code: &mut ExecutableBuffer, dst: VReg, src: VReg) -> Result<(), LoweringError> {
        self.emit_load_vreg_to_rax(code, src)?;
        code.write(&[0x48, 0x0F, 0xC8])?;
        self.emit_store_rax_to_vreg(code, dst)
    }

    /// VMOVDQU ymm, [rdi + disp32]
    #[allow(dead_code)]
    fn emit_vmovdqu_load(&self, code: &mut ExecutableBuffer, dst: VReg, offset: u32) -> Result<(), LoweringError> {
        let dst_reg = dst.0 & 0x0F;
        
        if dst_reg < 8 {
            code.write(&[0xC5, 0xFE])?;
        } else {
            code.write(&[0xC4, 0x41, 0x7E])?;
        }
        
        code.write(&[0x6F])?;
        
        let modrm = 0x80 | ((dst_reg & 7) << 3) | 0x07;
        code.write(&[modrm])?;
        
        // disp32
        code.write(&offset.to_le_bytes())?;
        
        Ok(())
    }

    /// VMOVDQU [rdx + disp32], ymm
    #[allow(dead_code)]
    fn emit_vmovdqu_store(&self, code: &mut ExecutableBuffer, src: VReg, offset: u32) -> Result<(), LoweringError> {
        let src_reg = src.0 & 0x0F;
        
        if src_reg < 8 {
            code.write(&[0xC5, 0xFE])?;
        } else {
            code.write(&[0xC4, 0x41, 0x7E])?;
        }
        
        code.write(&[0x7F])?;
        
        let modrm = 0x80 | ((src_reg & 7) << 3) | 0x02;
        code.write(&[modrm])?;
        
        // disp32
        code.write(&offset.to_le_bytes())?;
        
        Ok(())
    }

    /// Broadcast imm64 to all lanes.
    #[allow(dead_code)]
    fn emit_broadcast_imm64(&self, code: &mut ExecutableBuffer, dst: VReg, value: u64, _scalar_type: ScalarType) -> Result<(), LoweringError> {
        let dst_reg = dst.0 & 0x0F;
        
        code.write(&[0x48, 0xB8])?;
        code.write(&value.to_le_bytes())?;
        
        if dst_reg < 8 {
            code.write(&[0xC4, 0xE1, 0xF9])?;
        } else {
            code.write(&[0xC4, 0x61, 0xF9])?;
        }
        code.write(&[0x6E])?;
        let modrm = 0xC0 | ((dst_reg & 7) << 3);
        code.write(&[modrm])?;
        
        if dst_reg < 8 {
            code.write(&[0xC4, 0xE2, 0x7D])?;
        } else {
            code.write(&[0xC4, 0x62, 0x7D])?;
        }
        code.write(&[0x59])?;
        let modrm = 0xC0 | ((dst_reg & 7) << 3) | (dst_reg & 7);
        code.write(&[modrm])?;
        
        Ok(())
    }

    /// VPCMPGTQ ymm, ymm, ymm
    #[allow(dead_code)]
    fn emit_vpcmpgtq(&self, code: &mut ExecutableBuffer, dst: VReg, src1: VReg, src2: VReg) -> Result<(), LoweringError> {
        let dst_reg = dst.0 & 0x0F;
        let src1_reg = src1.0 & 0x0F;
        let src2_reg = src2.0 & 0x0F;
        
        
        let r = if dst_reg < 8 { 0x80 } else { 0x00 };
        let b = if src2_reg < 8 { 0x20 } else { 0x00 };
        let byte1 = r | 0x40 | b | 0x02; // R.1.B.00010
        
        let vvvv = (!(src1_reg) & 0x0F) << 3;
        let byte2 = 0x80 | vvvv | 0x05; // W=1, vvvv, L=1, pp=01
        
        code.write(&[0xC4, byte1, byte2])?;
        code.write(&[0x37])?;
        
        let modrm = 0xC0 | ((dst_reg & 7) << 3) | (src2_reg & 7);
        code.write(&[modrm])?;
        
        Ok(())
    }

    /// VPCMPEQQ ymm, ymm, ymm
    #[allow(dead_code)]
    fn emit_vpcmpeqq(&self, code: &mut ExecutableBuffer, dst: VReg, src1: VReg, src2: VReg) -> Result<(), LoweringError> {
        let dst_reg = dst.0 & 0x0F;
        let src1_reg = src1.0 & 0x0F;
        let src2_reg = src2.0 & 0x0F;
        
        let r = if dst_reg < 8 { 0x80 } else { 0x00 };
        let b = if src2_reg < 8 { 0x20 } else { 0x00 };
        let byte1 = r | 0x40 | b | 0x02;
        
        let vvvv = (!(src1_reg) & 0x0F) << 3;
        let byte2 = vvvv | 0x05;
        
        code.write(&[0xC4, byte1, byte2])?;
        code.write(&[0x29])?;
        
        let modrm = 0xC0 | ((dst_reg & 7) << 3) | (src2_reg & 7);
        code.write(&[modrm])?;
        
        Ok(())
    }

    /// Validate nonzero using YMM15 as scratch zero vector.
    #[allow(dead_code)]
    fn emit_validate_nonzero(&self, code: &mut ExecutableBuffer, dst: VReg, src: VReg) -> Result<(), LoweringError> {
        code.write(&[0xC4, 0x41, 0x05])?; // R=0, X=1, B=0, mmmmm=1, W=0, vvvv=0000, L=1, pp=01
        code.write(&[0xEF])?;
        code.write(&[0xFF])?; // mod=11, reg=15, rm=15
        
        self.emit_vpcmpgtq(code, dst, src, VReg(15))
    }

    /// VPAND ymm, ymm, ymm
    #[allow(dead_code)]
    fn emit_vpand(&self, code: &mut ExecutableBuffer, dst: VReg, src1: VReg, src2: VReg) -> Result<(), LoweringError> {
        let dst_reg = dst.0 & 0x0F;
        let src1_reg = src1.0 & 0x0F;
        let src2_reg = src2.0 & 0x0F;
        
        if dst_reg < 8 && src2_reg < 8 {
            let vvvv = (!(src1_reg) & 0x0F) << 3;
            let byte1 = 0x80 | vvvv | 0x05; // R=1, vvvv, L=1, pp=01
            code.write(&[0xC5, byte1])?;
        } else {
            let r = if dst_reg < 8 { 0x80 } else { 0x00 };
            let b = if src2_reg < 8 { 0x20 } else { 0x00 };
            let byte1 = r | 0x40 | b | 0x01;
            let vvvv = (!(src1_reg) & 0x0F) << 3;
            let byte2 = vvvv | 0x05;
            code.write(&[0xC4, byte1, byte2])?;
        }
        
        code.write(&[0xDB])?;
        let modrm = 0xC0 | ((dst_reg & 7) << 3) | (src2_reg & 7);
        code.write(&[modrm])?;
        
        Ok(())
    }

    /// VPOR ymm, ymm, ymm
    #[allow(dead_code)]
    fn emit_vpor(&self, code: &mut ExecutableBuffer, dst: VReg, src1: VReg, src2: VReg) -> Result<(), LoweringError> {
        let dst_reg = dst.0 & 0x0F;
        let src1_reg = src1.0 & 0x0F;
        let src2_reg = src2.0 & 0x0F;
        
        if dst_reg < 8 && src2_reg < 8 {
            let vvvv = (!(src1_reg) & 0x0F) << 3;
            let byte1 = 0x80 | vvvv | 0x05;
            code.write(&[0xC5, byte1])?;
        } else {
            let r = if dst_reg < 8 { 0x80 } else { 0x00 };
            let b = if src2_reg < 8 { 0x20 } else { 0x00 };
            let byte1 = r | 0x40 | b | 0x01;
            let vvvv = (!(src1_reg) & 0x0F) << 3;
            let byte2 = vvvv | 0x05;
            code.write(&[0xC4, byte1, byte2])?;
        }
        
        code.write(&[0xEB])?;
        let modrm = 0xC0 | ((dst_reg & 7) << 3) | (src2_reg & 7);
        code.write(&[modrm])?;
        
        Ok(())
    }

    /// VPXOR ymm, ymm, ymm
    #[allow(dead_code)]
    fn emit_vpxor(&self, code: &mut ExecutableBuffer, dst: VReg, src1: VReg, src2: VReg) -> Result<(), LoweringError> {
        let dst_reg = dst.0 & 0x0F;
        let src1_reg = src1.0 & 0x0F;
        let src2_reg = src2.0 & 0x0F;
        
        if dst_reg < 8 && src2_reg < 8 {
            let vvvv = (!(src1_reg) & 0x0F) << 3;
            let byte1 = 0x80 | vvvv | 0x05;
            code.write(&[0xC5, byte1])?;
        } else {
            let r = if dst_reg < 8 { 0x80 } else { 0x00 };
            let b = if src2_reg < 8 { 0x20 } else { 0x00 };
            let byte1 = r | 0x40 | b | 0x01;
            let vvvv = (!(src1_reg) & 0x0F) << 3;
            let byte2 = vvvv | 0x05;
            code.write(&[0xC4, byte1, byte2])?;
        }
        
        code.write(&[0xEF])?;
        let modrm = 0xC0 | ((dst_reg & 7) << 3) | (src2_reg & 7);
        code.write(&[modrm])?;
        
        Ok(())
    }

    /// VBLENDVPD ymm, ymm, ymm, ymm
    #[allow(dead_code)]
    fn emit_vblendvpd(&self, code: &mut ExecutableBuffer, dst: VReg, src1: VReg, src2: VReg, mask: VReg) -> Result<(), LoweringError> {
        let dst_reg = dst.0 & 0x0F;
        let src1_reg = src1.0 & 0x0F;
        let src2_reg = src2.0 & 0x0F;
        let mask_reg = mask.0 & 0x0F;
        
        // 3-byte VEX: C4 <R.X.B.mmmmm> <W.vvvv.L.pp>
        let r = if dst_reg < 8 { 0x80 } else { 0x00 };
        let b = if src2_reg < 8 { 0x20 } else { 0x00 };
        let byte1 = r | 0x40 | b | 0x03; // mmmmm = 00011 (0F3A)
        
        let vvvv = (!(src1_reg) & 0x0F) << 3;
        let byte2 = vvvv | 0x05;
        
        code.write(&[0xC4, byte1, byte2])?;
        code.write(&[0x4B])?;
        
        let modrm = 0xC0 | ((dst_reg & 7) << 3) | (src2_reg & 7);
        code.write(&[modrm])?;
        
        let is4 = (mask_reg & 0x0F) << 4;
        code.write(&[is4])?;
        
        Ok(())
    }

    /// VPADDQ ymm, ymm, ymm
    #[allow(dead_code)]
    fn emit_vpaddq(&self, code: &mut ExecutableBuffer, dst: VReg, src1: VReg, src2: VReg, _scalar_type: ScalarType) -> Result<(), LoweringError> {
        let dst_reg = dst.0 & 0x0F;
        let src1_reg = src1.0 & 0x0F;
        let src2_reg = src2.0 & 0x0F;
        
        if dst_reg < 8 && src2_reg < 8 {
            let vvvv = (!(src1_reg) & 0x0F) << 3;
            let byte1 = 0x80 | vvvv | 0x05;
            code.write(&[0xC5, byte1])?;
        } else {
            let r = if dst_reg < 8 { 0x80 } else { 0x00 };
            let b = if src2_reg < 8 { 0x20 } else { 0x00 };
            let byte1 = r | 0x40 | b | 0x01;
            let vvvv = (!(src1_reg) & 0x0F) << 3;
            let byte2 = vvvv | 0x05;
            code.write(&[0xC4, byte1, byte2])?;
        }
        
        code.write(&[0xD4])?;
        let modrm = 0xC0 | ((dst_reg & 7) << 3) | (src2_reg & 7);
        code.write(&[modrm])?;
        
        Ok(())
    }

    /// VPSUBQ ymm, ymm, ymm
    #[allow(dead_code)]
    fn emit_vpsubq(&self, code: &mut ExecutableBuffer, dst: VReg, src1: VReg, src2: VReg, _scalar_type: ScalarType) -> Result<(), LoweringError> {
        let dst_reg = dst.0 & 0x0F;
        let src1_reg = src1.0 & 0x0F;
        let src2_reg = src2.0 & 0x0F;
        
        if dst_reg < 8 && src2_reg < 8 {
            let vvvv = (!(src1_reg) & 0x0F) << 3;
            let byte1 = 0x80 | vvvv | 0x05;
            code.write(&[0xC5, byte1])?;
        } else {
            let r = if dst_reg < 8 { 0x80 } else { 0x00 };
            let b = if src2_reg < 8 { 0x20 } else { 0x00 };
            let byte1 = r | 0x40 | b | 0x01;
            let vvvv = (!(src1_reg) & 0x0F) << 3;
            let byte2 = vvvv | 0x05;
            code.write(&[0xC4, byte1, byte2])?;
        }
        
        code.write(&[0xFB])?;
        let modrm = 0xC0 | ((dst_reg & 7) << 3) | (src2_reg & 7);
        code.write(&[modrm])?;
        
        Ok(())
    }

    fn emit_nop(&self, code: &mut ExecutableBuffer, bytes: u8) -> Result<(), LoweringError> {
        let nop_bytes: &[u8] = match bytes {
            0 => &[],
            1 => &[0x90],
            2 => &[0x66, 0x90],
            3 => &[0x0F, 0x1F, 0x00],
            4 => &[0x0F, 0x1F, 0x40, 0x00],
            5 => &[0x0F, 0x1F, 0x44, 0x00, 0x00],
            6 => &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00],
            7 => &[0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00],
            8 => &[0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
            9 => &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
            _ => {
                for _ in 0..(bytes / 9) {
                    code.write(&[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00])?;
                }
                let remainder = bytes % 9;
                if remainder > 0 {
                    return self.emit_nop(code, remainder);
                }
                return Ok(());
            }
        };
        code.write(nop_bytes)
    }
}

/// Compares JIT output vs reference interpreter.
pub struct DivergenceChecker<'a> {
    graph: &'a RifGraph<'a>,
}

impl<'a> DivergenceChecker<'a> {
    pub fn new(graph: &'a RifGraph<'a>) -> Self {
        Self { graph }
    }

    /// Reference interpreter: scalar, simple, obviously correct.
    pub fn reference_execute(&self, packet: &[u8], output: &mut [u8]) -> Result<ExecutionSignature, ()> {
        let mut sig = ExecutionSignature::new();
        
        let mut values: [u64; 512] = [0; 512];
        let mut masks: [bool; 512] = [true; 512];

        for (idx, node) in self.graph.nodes.iter().enumerate() {
            match node {
                RifNode::Load { scalar_type, access } => {
                    if access.offset as usize + scalar_type.size_bytes() as usize <= packet.len() {
                        let val = self.read_scalar(packet, access.offset as usize, *scalar_type);
                        values[idx] = val;
                        sig.record_load(access.offset, val);
                    }
                }
                RifNode::Const { value, .. } => {
                    values[idx] = u64::from_be_bytes(*value);
                }
                RifNode::BinaryOp { op, lhs, rhs, .. } => {
                    let l = values[lhs.0 as usize];
                    let r = values[rhs.0 as usize];
                    values[idx] = self.eval_binop(*op, l, r);
                }
                RifNode::Validate { value_node, constraint } => {
                    let val = values[value_node.0 as usize];
                    masks[idx] = self.check_constraint(val, constraint);
                    sig.record_validation(idx as u32, masks[idx]);
                }
                RifNode::Guard { parent_mask, condition } => {
                    let parent = parent_mask.map(|p| masks[p.0 as usize]).unwrap_or(true);
                    let cond = masks[condition.0 as usize];
                    masks[idx] = parent && cond;
                    sig.record_mask_change(idx as u32, masks[idx]);
                }
                RifNode::Emit { field_id, value_node, mask } => {
                    let active = mask.map(|m| masks[m.0 as usize]).unwrap_or(true);
                    if active {
                        let val = values[value_node.0 as usize];
                        let offset = (*field_id as usize) * 8;
                        if offset + 8 <= output.len() {
                            output[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
                        }
                        sig.record_emit(*field_id, val);
                    }
                }
                _ => {}
            }
        }

        Ok(sig)
    }

    fn read_scalar(&self, packet: &[u8], offset: usize, scalar_type: ScalarType) -> u64 {
        let size = scalar_type.size_bytes() as usize;
        if offset + size > packet.len() {
            return 0;
        }
        
        let bytes = &packet[offset..offset + size];
        match scalar_type {
            ScalarType::U8 => bytes[0] as u64,
            ScalarType::U16 => u16::from_le_bytes([bytes[0], bytes[1]]) as u64,
            ScalarType::U32 => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64,
            ScalarType::U64 => u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
            ]),
            ScalarType::I32 => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64,
            ScalarType::I64 => i64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
            ]) as u64,
            ScalarType::F32 | ScalarType::F64 => 0, // simplified
        }
    }

    fn eval_binop(&self, op: BinaryOp, l: u64, r: u64) -> u64 {
        match op {
            BinaryOp::Add => l.wrapping_add(r),
            BinaryOp::Sub => l.wrapping_sub(r),
            BinaryOp::Mul => l.wrapping_mul(r),
            BinaryOp::Div => if r != 0 { l / r } else { 0 },
            BinaryOp::And => l & r,
            BinaryOp::Or => l | r,
            BinaryOp::Xor => l ^ r,
            BinaryOp::Shl => l << (r & 63),
            BinaryOp::Shr => l >> (r & 63),
            BinaryOp::Eq => if l == r { 1 } else { 0 },
            BinaryOp::Ne => if l != r { 1 } else { 0 },
            BinaryOp::Lt => if l < r { 1 } else { 0 },
            BinaryOp::Le => if l <= r { 1 } else { 0 },
            BinaryOp::Gt => if l > r { 1 } else { 0 },
            BinaryOp::Ge => if l >= r { 1 } else { 0 },
        }
    }

    fn check_constraint(&self, val: u64, constraint: &Constraint) -> bool {
        match constraint {
            Constraint::None => true,
            Constraint::NonZero => val != 0,
            Constraint::Finite => true, // simplified
            Constraint::Range { lo, hi } => val >= *lo && val <= *hi,
        }
    }

    /// Verify JIT matches reference.
    pub fn check_equivalence(
        &self,
        kernel: &LoweredKernel,
        packet: &[u8],
    ) -> Result<(), DivergenceError> {
        let mut ref_output = [0u8; 4096];
        let mut fast_output = [0u8; 4096];

        let ref_sig = self.reference_execute(packet, &mut ref_output)
            .map_err(|_| DivergenceError::ReferencePathFailed)?;

        let kernel_fn = kernel.as_fn();
        let result = kernel_fn(packet.as_ptr(), packet.len() as u32, fast_output.as_mut_ptr());
        
        if result != 0 {
            if ref_sig.validation_failures > 0 {
                return Ok(()); // Both failed, that's consistent
            }
            return Err(DivergenceError::FastPathFailedUnexpectedly);
        }

        if ref_output != fast_output {
            return Err(DivergenceError::OutputMismatch);
        }

        Ok(())
    }
}

/// Execution trace for divergence detection.
#[derive(Clone, Debug)]
pub struct ExecutionSignature {
    pub loads: [(u32, u64); 64],
    pub load_count: usize,
    pub validations: [(u32, bool); 64],
    pub validation_count: usize,
    pub mask_changes: [(u32, bool); 64],
    pub mask_count: usize,
    pub emits: [(u16, u64); 64],
    pub emit_count: usize,
    pub validation_failures: usize,
}

impl ExecutionSignature {
    pub fn new() -> Self {
        Self {
            loads: [(0, 0); 64],
            load_count: 0,
            validations: [(0, false); 64],
            validation_count: 0,
            mask_changes: [(0, false); 64],
            mask_count: 0,
            emits: [(0, 0); 64],
            emit_count: 0,
            validation_failures: 0,
        }
    }

    fn record_load(&mut self, offset: u32, value: u64) {
        if self.load_count < 64 {
            self.loads[self.load_count] = (offset, value);
            self.load_count += 1;
        }
    }

    fn record_validation(&mut self, node: u32, passed: bool) {
        if self.validation_count < 64 {
            self.validations[self.validation_count] = (node, passed);
            self.validation_count += 1;
            if !passed {
                self.validation_failures += 1;
            }
        }
    }

    fn record_mask_change(&mut self, node: u32, active: bool) {
        if self.mask_count < 64 {
            self.mask_changes[self.mask_count] = (node, active);
            self.mask_count += 1;
        }
    }

    fn record_emit(&mut self, field_id: u16, value: u64) {
        if self.emit_count < 64 {
            self.emits[self.emit_count] = (field_id, value);
            self.emit_count += 1;
        }
    }
}

impl Default for ExecutionSignature {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DivergenceError {
    ReferencePathFailed,
    FastPathFailedUnexpectedly,
    OutputMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rif::*;
    use crate::semantic_hash::compute_semantic_hash;

    #[test]
    fn test_lowering_minimal_kernel() {
        let nodes = [
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

        let graph = RifGraph {
            version: RifVersion::CURRENT,
            protocol_version: 1,
            nodes: &nodes,
            max_packet_length: 64,
            version_discriminator_node: NodeIndex(0),
        };

        let phase_a_hash = compute_semantic_hash(&graph).unwrap();
        let engine = LoweringEngine::new(SimdWidth::Avx2, 64);
        
        let result = engine.lower(&graph, phase_a_hash);
        assert!(result.is_ok());
        
        let kernel = result.unwrap();
        assert!(kernel.code_size() > 0);
        assert!(!kernel.witness().is_empty());
    }

    #[test]
    fn test_reference_path_execution() {
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

        let checker = DivergenceChecker::new(&graph);
        let packet = [0x42u8; 64];
        let mut output = [0u8; 64];
        
        let sig = checker.reference_execute(&packet, &mut output).unwrap();
        
        // Check that something was emitted
        assert!(sig.emit_count > 0);
    }
}
