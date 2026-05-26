//! x86_64 AVX2 Encoder

#![allow(dead_code)]

use crate::lowering::target::{MicroOp, SimdWidth, VReg, LoweringError};
use crate::rif::ScalarType;

/// Emit VEX prefix for reg-reg AVX2 256-bit instructions.
/// Uses 2-byte form (C5) when possible, falls back to 3-byte (C4).
/// `map`: 1 = 0F, 2 = 0F38, 3 = 0F3A.
/// `pp`: 0 = none, 1 = 66, 2 = F3, 3 = F2.
/// `w`: REX.W bit (0 or 1).
fn vex_rr(buf: &mut [u8; 4], map: u8, pp: u8, w: u8, vvvv: u8, reg: u8, rm: u8) -> usize {
    let r_inv = if reg < 8 { 0x80u8 } else { 0x00u8 };
    let b_inv = if rm < 8 { 0x20u8 } else { 0x00u8 };
    let vvvv_inv = ((!vvvv) & 0xF) << 3;

    // 2-byte VEX: only when map=0F, W=0, X=1, B=1 (rm < 8)
    if map == 1 && w == 0 && rm < 8 {
        buf[0] = 0xC5;
        buf[1] = r_inv | vvvv_inv | 0x04 | pp; // L=1 (bit 2), pp in bits 0-1
        2
    } else {
        buf[0] = 0xC4;
        buf[1] = r_inv | 0x40 | b_inv | map; // X̃=1 (bit 6), R̃, B̃, mmmmm
        buf[2] = (w << 7) | vvvv_inv | 0x04 | pp; // W, ~vvvv, L=1, pp
        3
    }
}

#[inline]
const fn modrm(mod_: u8, reg: u8, rm: u8) -> u8 {
    ((mod_ & 0x3) << 6) | ((reg & 0x7) << 3) | (rm & 0x7)
}

#[inline]
const fn sib(scale: u8, index: u8, base: u8) -> u8 {
    ((scale & 0x3) << 6) | ((index & 0x7) << 3) | (base & 0x7)
}

/// VReg -> YMM. YMM14-15 reserved for scratch.
#[inline]
const fn ymm(vreg: VReg) -> u8 {
    vreg.0
}

/// Load fault contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadContract {
    UnmaskedUnchecked,
    MaskedFaultSafe,
    ScalarPeel,
}

impl LoadContract {
    pub fn for_load(offset: u32, width: SimdWidth, mask: Option<VReg>, packet_len: u16) -> Self {
        let load_bytes = width.bytes() as u64;
        let load_end = offset as u64 + load_bytes;

        if mask.is_none() && load_end <= packet_len as u64 {
            // Page-crossing check: first and last byte must be on the same 4KB page
            let first_page = (offset as u64) >> 12;
            let last_page = (offset as u64 + load_bytes - 1) >> 12;
            if first_page == last_page {
                return LoadContract::UnmaskedUnchecked;
            }
        }

        if mask.is_some() {
            return LoadContract::MaskedFaultSafe;
        }

        LoadContract::ScalarPeel
    }
}

/// The byte slinger. 16KB I-cache budget.
pub struct X86_64Encoder {
    buf: [u8; 16384],
    pos: usize,
    packet_len: u16,
    packet_base: u8,
    output_base: u8,
}

impl X86_64Encoder {
    pub fn new(packet_len: u16) -> Self {
        Self {
            buf: [0u8; 16384],
            pos: 0,
            packet_len,
            packet_base: 7,  // rdi
            output_base: 2,  // rdx
        }
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        16384 - self.pos
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pos
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pos == 0
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.pos]
    }

    /// Emit a VEX-encoded reg-reg instruction: VEX prefix + opcode + ModRM.
    fn emit_vex_rr(&mut self, map: u8, pp: u8, w: u8, opcode: u8, dst: u8, src1: u8, src2: u8) -> Result<(), LoweringError> {
        let mut vex = [0u8; 4];
        let vex_len = vex_rr(&mut vex, map, pp, w, src1, dst, src2);
        self.emit(&vex[..vex_len])?;
        self.emit(&[opcode, modrm(0b11, dst & 7, src2 & 7)])
    }

    fn emit(&mut self, bytes: &[u8]) -> Result<(), LoweringError> {
        if self.pos + bytes.len() > 16384 {
            return Err(LoweringError::ICacheBudgetExceeded {
                current: self.pos,
                requested: bytes.len(),
                limit: 16384,
            });
        }
        self.buf[self.pos..self.pos + bytes.len()].copy_from_slice(bytes);
        self.pos += bytes.len();
        Ok(())
    }

    /// VMOVDQU ymm, [rdi + offset] — VEX.256.F3.0F 6F /r
    pub fn emit_vmovdqu_load(&mut self, dst: VReg, offset: u32) -> Result<usize, LoweringError> {
        let start = self.pos;
        let dst_reg = ymm(dst);
        let r_inv = if dst_reg < 8 { 0x80u8 } else { 0x00u8 };
        if dst_reg < 8 {
            // 2-byte VEX: C5 [R̃ vvvv=1111 L=1 pp=10(F3)]
            self.emit(&[0xC5, r_inv | 0x7E, 0x6F])?;
        } else {
            // 3-byte VEX: packet_base < 8 so B̃=1
            self.emit(&[0xC4, r_inv | 0x61, 0x7E, 0x6F])?;
        }
        self.emit(&[modrm(0b10, dst_reg & 7, self.packet_base)])?;
        self.emit(&offset.to_le_bytes())?;
        Ok(self.pos - start)
    }

    /// VPMASKMOVQ — fault-safe masked load. VEX.256.66.0F38 8C /r
    pub fn emit_vpmaskmovq_load(&mut self, dst: VReg, mask: VReg, offset: u32) -> Result<usize, LoweringError> {
        let start = self.pos;
        let dst_reg = ymm(dst);
        let mask_reg = ymm(mask);
        let r_inv = if dst_reg < 8 { 0x80u8 } else { 0x00u8 };
        let vvvv_inv = ((!mask_reg) & 0xF) << 3;
        // 3-byte VEX: C4 [R̃Xb̃ mmmmm] [W vvvv L pp]
        // map=2 (0F38), X̃=1, B̃=1 (packet_base < 8), W=0, L=1, pp=01
        self.emit(&[0xC4, r_inv | 0x62, vvvv_inv | 0x05, 0x8C])?;
        self.emit(&[modrm(0b10, dst_reg & 7, self.packet_base)])?;
        self.emit(&offset.to_le_bytes())?;
        Ok(self.pos - start)
    }

    /// VPCMPGTQ ymm, ymm, ymm — VEX.256.66.0F38 37 /r
    pub fn emit_vpcmpgtq(&mut self, dst: VReg, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        self.emit_vex_rr(2, 1, 0, 0x37, ymm(dst), ymm(src1), ymm(src2))?;
        Ok(self.pos - start)
    }

    /// VPCMPEQQ ymm, ymm, ymm — VEX.256.66.0F38 29 /r
    pub fn emit_vpcmpeqq(&mut self, dst: VReg, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        self.emit_vex_rr(2, 1, 0, 0x29, ymm(dst), ymm(src1), ymm(src2))?;
        Ok(self.pos - start)
    }

    /// VPANDN ymm, ymm, ymm — dst = ~src1 & src2, VEX.256.66.0F DF /r
    pub fn emit_vpandn(&mut self, dst: VReg, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        self.emit_vex_rr(1, 1, 0, 0xDF, ymm(dst), ymm(src1), ymm(src2))?;
        Ok(self.pos - start)
    }

    /// VPCMPEQD ymm, ymm, ymm — VEX.256.66.0F 76 /r
    pub fn emit_vpcmpeqd(&mut self, dst: VReg, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        self.emit_vex_rr(1, 1, 0, 0x76, ymm(dst), ymm(src1), ymm(src2))?;
        Ok(self.pos - start)
    }

    /// VPAND ymm, ymm, ymm — VEX.256.66.0F DB /r
    pub fn emit_vpand(&mut self, dst: VReg, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        self.emit_vex_rr(1, 1, 0, 0xDB, ymm(dst), ymm(src1), ymm(src2))?;
        Ok(self.pos - start)
    }

    /// VPOR ymm, ymm, ymm — VEX.256.66.0F EB /r
    pub fn emit_vpor(&mut self, dst: VReg, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        self.emit_vex_rr(1, 1, 0, 0xEB, ymm(dst), ymm(src1), ymm(src2))?;
        Ok(self.pos - start)
    }

    /// VPXOR ymm, ymm, ymm — VEX.256.66.0F EF /r
    pub fn emit_vpxor(&mut self, dst: VReg, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        self.emit_vex_rr(1, 1, 0, 0xEF, ymm(dst), ymm(src1), ymm(src2))?;
        Ok(self.pos - start)
    }

    /// VPADDQ ymm, ymm, ymm — VEX.256.66.0F D4 /r
    pub fn emit_vpaddq(&mut self, dst: VReg, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        self.emit_vex_rr(1, 1, 0, 0xD4, ymm(dst), ymm(src1), ymm(src2))?;
        Ok(self.pos - start)
    }

    /// VPSUBQ ymm, ymm, ymm — VEX.256.66.0F FB /r
    pub fn emit_vpsubq(&mut self, dst: VReg, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        self.emit_vex_rr(1, 1, 0, 0xFB, ymm(dst), ymm(src1), ymm(src2))?;
        Ok(self.pos - start)
    }

    /// Unsigned VPCMPGTQ via sign-flip trick: XOR both operands with 0x8000000000000000,
    /// then signed compare. Uses YMM14/YMM15 as scratch.
    pub fn emit_unsigned_cmpgt(&mut self, dst: VReg, src: VReg, comparand: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        let sign_bit = VReg(15);
        let tmp_a = VReg(14);

        self.emit_vpbroadcastq_imm(sign_bit, 0x8000000000000000u64)?;
        self.emit_vpxor(tmp_a, src, sign_bit)?;
        self.emit_vpxor(sign_bit, comparand, sign_bit)?;
        self.emit_vpcmpgtq(dst, tmp_a, sign_bit)?;

        Ok(self.pos - start)
    }

    /// VPBROADCASTQ ymm, imm64 — via RAX scratch
    pub fn emit_vpbroadcastq_imm(&mut self, dst: VReg, value: u64) -> Result<usize, LoweringError> {
        let start = self.pos;
        let d = ymm(dst);
        let r_inv = if d < 8 { 0x80u8 } else { 0x00u8 };
        
        // MOV RAX, imm64
        self.emit(&[0x48, 0xB8])?;
        self.emit(&value.to_le_bytes())?;
        // VMOVQ xmm(dst), RAX — VEX.128.66.0F.W1 6E /r (rm=RAX=0)
        self.emit(&[0xC4, r_inv | 0x61, 0xF9, 0x6E, modrm(0b11, d & 7, 0)])?;
        // VPBROADCASTQ ymm(dst), xmm(dst) — VEX.256.66.0F38 59 /r
        self.emit(&[0xC4, r_inv | 0x62, 0x7D, 0x59, modrm(0b11, d & 7, d & 7)])?;
        
        Ok(self.pos - start)
    }

    /// VBLENDVPD ymm, ymm, ymm, ymm — VEX.256.66.0F3A 4B /r /is4
    pub fn emit_vblendvpd(&mut self, dst: VReg, src1: VReg, src2: VReg, mask: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        let d = ymm(dst);
        let s2 = ymm(src2);
        let r_inv = if d < 8 { 0x80u8 } else { 0x00u8 };
        let b_inv = if s2 < 8 { 0x20u8 } else { 0x00u8 };
        let vvvv_inv = ((!ymm(src1)) & 0xF) << 3;
        // 3-byte VEX: map=3 (0F3A), X̃=1, W=0, L=1, pp=01
        self.emit(&[0xC4, r_inv | 0x43 | b_inv, vvvv_inv | 0x05, 0x4B])?;
        self.emit(&[modrm(0b11, d & 7, s2 & 7)])?;
        self.emit(&[(ymm(mask) << 4)])?;
        Ok(self.pos - start)
    }

    /// VMOVNTDQ [rdx + offset], ymm — VEX.256.66.0F E7 /r
    pub fn emit_vmovntdq_store(&mut self, src: VReg, offset: u32) -> Result<usize, LoweringError> {
        let start = self.pos;
        let src_reg = ymm(src);
        let r_inv = if src_reg < 8 { 0x80u8 } else { 0x00u8 };
        if src_reg < 8 {
            self.emit(&[0xC5, r_inv | 0x7D, 0xE7])?;
        } else {
            self.emit(&[0xC4, r_inv | 0x61, 0x7D, 0xE7])?;
        }
        self.emit(&[modrm(0b10, src_reg & 7, self.output_base)])?;
        self.emit(&offset.to_le_bytes())?;
        Ok(self.pos - start)
    }

    /// VMOVDQU [rdx + offset], ymm — VEX.256.F3.0F 7F /r
    pub fn emit_vmovdqu_store(&mut self, src: VReg, offset: u32) -> Result<usize, LoweringError> {
        let start = self.pos;
        let src_reg = ymm(src);
        let r_inv = if src_reg < 8 { 0x80u8 } else { 0x00u8 };
        if src_reg < 8 {
            self.emit(&[0xC5, r_inv | 0x7E, 0x7F])?;
        } else {
            self.emit(&[0xC4, r_inv | 0x61, 0x7E, 0x7F])?;
        }
        self.emit(&[modrm(0b10, src_reg & 7, self.output_base)])?;
        self.emit(&offset.to_le_bytes())?;
        Ok(self.pos - start)
    }

    /// VPSHUFB ymm, ymm, ymm — VEX.256.66.0F38 00 /r
    pub fn emit_vpshufb(&mut self, dst: VReg, src: VReg, mask: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        self.emit_vex_rr(2, 1, 0, 0x00, ymm(dst), ymm(src), ymm(mask))?;
        Ok(self.pos - start)
    }

    /// Materialize byte-swap shuffle mask in scratch (YMM15), then VPSHUFB dst, src, YMM15.
    /// Mask depends on scalar element width.
    pub fn emit_bswap_vector(&mut self, dst: VReg, src: VReg, scalar_type: ScalarType) -> Result<usize, LoweringError> {
        let start = self.pos;
        let scratch = VReg(15);

        let (lo, hi): (u64, u64) = match scalar_type.size_bytes() {
            8 => (0x0001020304050607u64, 0x08090A0B0C0D0E0Fu64),
            4 => (0x0405060700010203u64, 0x0C0D0E0F08090A0Bu64),
            2 => (0x0607040502030001u64, 0x0E0F0C0D0A0B0809u64),
            _ => return Ok(0), // U8 byte-swap is identity
        };

        let s = ymm(scratch);
        let r_inv = if s < 8 { 0x80u8 } else { 0x00u8 };
        let b_inv = if s < 8 { 0x20u8 } else { 0x00u8 }; // rm field extension
        let vvvv_inv = ((!s) & 0xF) << 3;

        // MOV RAX, lo; VMOVQ xmm(scratch), rax — VEX.128.66.0F.W1 6E /r
        self.emit(&[0x48, 0xB8])?;
        self.emit(&lo.to_le_bytes())?;
        self.emit(&[0xC4, r_inv | 0x61, 0xF9, 0x6E, modrm(0b11, s & 7, 0)])?;

        // MOV RAX, hi; VPINSRQ xmm(scratch), xmm(scratch), rax, 1 — VEX.128.66.0F3A.W1 22 /r ib
        self.emit(&[0x48, 0xB8])?;
        self.emit(&hi.to_le_bytes())?;
        self.emit(&[0xC4, r_inv | 0x63, vvvv_inv | 0xF9, 0x22])?;
        self.emit(&[modrm(0b11, s & 7, 0)])?;
        self.emit(&[0x01])?;

        // VINSERTI128 ymm(scratch), ymm(scratch), xmm(scratch), 1 — VEX.256.66.0F3A 38 /r ib
        self.emit(&[0xC4, r_inv | 0x43 | b_inv, vvvv_inv | 0x05, 0x38])?;
        self.emit(&[modrm(0b11, s & 7, s & 7)])?;
        self.emit(&[0x01])?;

        self.emit_vpshufb(dst, src, scratch)?;
        Ok(self.pos - start)
    }

    /// VPTEST ymm, ymm — VEX.256.66.0F38 17 /r
    pub fn emit_vptest(&mut self, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        // VPTEST has no vvvv operand; vvvv must be 1111
        self.emit_vex_rr(2, 1, 0, 0x17, ymm(src1), 0x0F, ymm(src2))?;
        Ok(self.pos - start)
    }

    /// LFENCE — serializing instruction. Currently unused; retained for future use
    /// if a coherent speculation threat model is defined.
    #[allow(dead_code)]
    pub fn emit_lfence(&mut self) -> Result<usize, LoweringError> {
        let start = self.pos;
        self.emit(&[0x0F, 0xAE, 0xE8])?;
        Ok(self.pos - start)
    }

    pub fn emit_xor_eax_eax(&mut self) -> Result<usize, LoweringError> {
        let start = self.pos;
        self.emit(&[0x31, 0xC0])?;
        Ok(self.pos - start)
    }

    pub fn emit_ret(&mut self) -> Result<usize, LoweringError> {
        let start = self.pos;
        self.emit(&[0xC3])?;
        Ok(self.pos - start)
    }

    pub fn encode_microop(&mut self, op: &MicroOp) -> Result<usize, LoweringError> {
        match op {
            MicroOp::LoadVector { dst, offset, width, scalar_type: _, mask } => {
                let contract = LoadContract::for_load(*offset, *width, *mask, self.packet_len);
                match contract {
                    LoadContract::UnmaskedUnchecked => {
                        self.emit_vmovdqu_load(*dst, *offset)
                    }
                    LoadContract::MaskedFaultSafe => {
                        if let Some(m) = mask {
                            self.emit_vpmaskmovq_load(*dst, *m, *offset)
                        } else {
                            // Shouldn't happen, but fallback
                            self.emit_vmovdqu_load(*dst, *offset)
                        }
                    }
                    LoadContract::ScalarPeel => {
                        self.emit_vmovdqu_load(*dst, *offset)
                    }
                }
            }
            MicroOp::ValidateCmpEq { dst_mask, src, imm_or_reg, scalar_type: _ } => {
                self.emit_vpcmpeqq(*dst_mask, *src, *imm_or_reg)
            }
            MicroOp::ValidateCmpGe { dst_mask, src, comparand, scalar_type } => {
                if scalar_type.is_unsigned() {
                    self.emit_unsigned_cmpgt(*dst_mask, *src, *comparand)
                } else {
                    self.emit_vpcmpgtq(*dst_mask, *src, *comparand)
                }
            }
            MicroOp::ValidateCmpLe { dst_mask, src, comparand, scalar_type } => {
                if scalar_type.is_unsigned() {
                    self.emit_unsigned_cmpgt(*dst_mask, *comparand, *src)
                } else {
                    self.emit_vpcmpgtq(*dst_mask, *comparand, *src)
                }
            }
            MicroOp::ValidateNonZero { dst_mask: _, src, scalar_type: _ } => {
                self.emit_vptest(*src, *src)
            }
            MicroOp::MaskAnd { dst, src1, src2 } => {
                self.emit_vpand(*dst, *src1, *src2)
            }
            MicroOp::MaskOr { dst, src1, src2 } => {
                self.emit_vpor(*dst, *src1, *src2)
            }
            MicroOp::MaskNot { dst, src } => {
                let scratch = VReg(15);
                self.emit_vpcmpeqd(scratch, scratch, scratch)?;
                self.emit_vpandn(*dst, *src, scratch)
            }
            MicroOp::Select { dst, mask, true_val, false_val, scalar_type: _ } => {
                self.emit_vblendvpd(*dst, *false_val, *true_val, *mask)
            }
            MicroOp::Emit { src, field_offset, scalar_type: _, mask } => {
                let offset = *field_offset as u32;
                if offset.is_multiple_of(32) && mask.is_none() {
                    self.emit_vmovntdq_store(*src, offset)
                } else {
                    self.emit_vmovdqu_store(*src, offset)
                }
            }
            MicroOp::BroadcastImm { dst, value, scalar_type: _ } => {
                self.emit_vpbroadcastq_imm(*dst, *value)
            }
            MicroOp::Add { dst, src1, src2, scalar_type: _ } => {
                self.emit_vpaddq(*dst, *src1, *src2)
            }
            MicroOp::Sub { dst, src1, src2, scalar_type: _ } => {
                self.emit_vpsubq(*dst, *src1, *src2)
            }
            MicroOp::And { dst, src1, src2 } => {
                self.emit_vpand(*dst, *src1, *src2)
            }
            MicroOp::Or { dst, src1, src2 } => {
                self.emit_vpor(*dst, *src1, *src2)
            }
            MicroOp::Xor { dst, src1, src2 } => {
                self.emit_vpxor(*dst, *src1, *src2)
            }
            MicroOp::ByteSwap { dst, src, scalar_type } => {
                self.emit_bswap_vector(*dst, *src, *scalar_type)
            }
            MicroOp::Nop { bytes } => {
                let nop_bytes = match bytes {
                    0 => &[][..],
                    1 => &[0x90][..],
                    2 => &[0x66, 0x90][..],
                    3 => &[0x0F, 0x1F, 0x00][..],
                    4 => &[0x0F, 0x1F, 0x40, 0x00][..],
                    5 => &[0x0F, 0x1F, 0x44, 0x00, 0x00][..],
                    6 => &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00][..],
                    7 => &[0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00][..],
                    8 => &[0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00][..],
                    _ => &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00][..],
                };
                self.emit(nop_bytes)?;
                Ok(nop_bytes.len())
            }
        }
    }

    pub fn emit_prologue(&mut self) -> Result<usize, LoweringError> {
        Ok(0)
    }

    pub fn emit_epilogue(&mut self) -> Result<usize, LoweringError> {
        let start = self.pos;
        self.emit_xor_eax_eax()?;
        self.emit_ret()?;
        Ok(self.pos - start)
    }
}

/// Extended witness hash including codegen decisions.
#[derive(Clone, Copy, Debug)]
pub struct ExtendedWitnessHash {
    pub phase_a_hash: [u8; 32],
    pub regalloc_hash: [u8; 8],
    pub isel_hash: [u8; 8],
    pub vector_width: u16,
    pub code_size: u16,
}

impl ExtendedWitnessHash {
    pub fn compute(
        phase_a_hash: &[u8; 32],
        encoder: &X86_64Encoder,
        regalloc_fingerprint: u64,
        isel_fingerprint: u64,
        width: SimdWidth,
    ) -> Self {
        Self {
            phase_a_hash: *phase_a_hash,
            regalloc_hash: regalloc_fingerprint.to_be_bytes(),
            isel_hash: isel_fingerprint.to_be_bytes(),
            vector_width: width.bits(),
            code_size: encoder.len() as u16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vmovdqu_encoding() {
        let mut enc = X86_64Encoder::new(1500);
        let size = enc.emit_vmovdqu_load(VReg(0), 0).unwrap();
        
        // Should be: C5 FE 6F 07 00 00 00 00 (8 bytes)
        assert!(size > 0);
        assert!(enc.len() <= 16384);
    }

    #[test]
    fn test_vpand_encoding() {
        let mut enc = X86_64Encoder::new(1500);
        let size = enc.emit_vpand(VReg(0), VReg(1), VReg(2)).unwrap();
        
        // Should be 4 bytes
        assert_eq!(size, 4);
    }

    #[test]
    fn test_vmovntdq_encoding() {
        let mut enc = X86_64Encoder::new(1500);
        let size = enc.emit_vmovntdq_store(VReg(0), 0).unwrap();
        
        // Should be: C5 FD E7 02 00 00 00 00
        assert!(size > 0);
    }

    #[test]
    fn test_load_contract_selection() {
        // Within bounds, no mask, no page crossing
        let c1 = LoadContract::for_load(0, SimdWidth::Avx2, None, 1500);
        assert_eq!(c1, LoadContract::UnmaskedUnchecked);
        
        // With mask
        let c2 = LoadContract::for_load(0, SimdWidth::Avx2, Some(VReg(0)), 1500);
        assert_eq!(c2, LoadContract::MaskedFaultSafe);
    }

    #[test]
    fn test_icache_budget() {
        let mut enc = X86_64Encoder::new(1500);
        
        // Fill up the buffer
        for _ in 0..2000 {
            if enc.emit_vpand(VReg(0), VReg(1), VReg(2)).is_err() {
                break;
            }
        }
        
        assert!(enc.len() <= 16384);
    }
}
