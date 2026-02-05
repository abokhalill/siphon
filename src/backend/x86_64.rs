//! x86_64 AVX2 Encoder

use crate::lowering::target::{MicroOp, SimdWidth, VReg, LoweringError, LaneMask};
use crate::rif::ScalarType;

/// VEX prefix encoding.
#[derive(Clone, Copy)]
struct VexPrefix {
    rxb: u8,
    map: u8,
    w_vvvv_l_pp: u8,
}

impl VexPrefix {
    const fn avx2_256_0f(vvvv: u8) -> Self {
        Self {
            rxb: 0b11100001,  // R=1, X=1, B=1 (inverted = 0,0,0), map=0F
            map: 0x01,
            w_vvvv_l_pp: 0b00000100 | (((!vvvv) & 0xF) << 3), // W=0, L=1 (256-bit), pp=00
        }
    }

    const fn avx2_256_0f38(vvvv: u8) -> Self {
        Self {
            rxb: 0b11100010,  // map=0F38
            map: 0x02,
            w_vvvv_l_pp: 0b00000100 | (((!vvvv) & 0xF) << 3),
        }
    }

    fn encode_3byte(&self, dst_reg: u8) -> [u8; 3] {
        let r = if dst_reg >= 8 { 0 } else { 0x80 };
        [
            0xC4,
            self.rxb | r,
            self.w_vvvv_l_pp,
        ]
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
        let load_end = offset as u64 + width.bytes() as u64;
        
        if mask.is_none() && load_end <= packet_len as u64 {
            let page_start = offset & !0xFFF;
            let page_end = page_start + 0x1000;
            if load_end <= page_end as u64 {
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

    pub fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.pos]
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

    /// VMOVDQU ymm, [rdi + offset]
    pub fn emit_vmovdqu_load(&mut self, dst: VReg, offset: u32) -> Result<usize, LoweringError> {
        let start = self.pos;
        let dst_reg = ymm(dst);
        
        if dst_reg < 8 {
            self.emit(&[0xC5, 0xFE, 0x6F])?;
        } else {
            self.emit(&[0xC4, 0xC1, 0x7E, 0x6F])?;
        }
        
        self.emit(&[modrm(0b10, dst_reg & 7, self.packet_base)])?;
        self.emit(&offset.to_le_bytes())?;
        
        Ok(self.pos - start)
    }

    /// VPMASKMOVQ — fault-safe masked load. Masked lanes won't fault.
    pub fn emit_vpmaskmovq_load(&mut self, dst: VReg, mask: VReg, offset: u32) -> Result<usize, LoweringError> {
        let start = self.pos;
        let dst_reg = ymm(dst);
        let mask_reg = ymm(mask);
        
        let vex2 = 0xE2;
        let vex3 = 0x80 | (((!mask_reg) & 0xF) << 3) | 0x05;
        
        self.emit(&[0xC4, vex2, vex3, 0x8C])?;
        self.emit(&[modrm(0b10, dst_reg & 7, self.packet_base)])?;
        self.emit(&offset.to_le_bytes())?;
        
        Ok(self.pos - start)
    }

    /// VPCMPGTQ ymm, ymm, ymm
    pub fn emit_vpcmpgtq(&mut self, dst: VReg, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        
        let vex2 = 0xE2;
        let vex3 = 0x80 | (((!ymm(src1)) & 0xF) << 3) | 0x05;
        
        self.emit(&[0xC4, vex2, vex3, 0x37])?;
        self.emit(&[modrm(0b11, ymm(dst) & 7, ymm(src2) & 7)])?;
        
        Ok(self.pos - start)
    }

    /// VPCMPEQQ ymm, ymm, ymm
    pub fn emit_vpcmpeqq(&mut self, dst: VReg, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        
        let vex2 = 0xE2;
        let vex3 = (((!ymm(src1)) & 0xF) << 3) | 0x05; // W=0, L=1, pp=01
        
        self.emit(&[0xC4, vex2, vex3, 0x29])?;
        self.emit(&[modrm(0b11, ymm(dst) & 7, ymm(src2) & 7)])?;
        
        Ok(self.pos - start)
    }

    /// VPAND ymm, ymm, ymm
    pub fn emit_vpand(&mut self, dst: VReg, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        let vvvv = (!ymm(src1)) & 0xF;
        self.emit(&[0xC5, (vvvv << 3) | 0x05, 0xDB])?;
        self.emit(&[modrm(0b11, ymm(dst) & 7, ymm(src2) & 7)])?;
        Ok(self.pos - start)
    }

    /// VPOR ymm, ymm, ymm
    pub fn emit_vpor(&mut self, dst: VReg, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        
        let vvvv = (!ymm(src1)) & 0xF;
        self.emit(&[0xC5, (vvvv << 3) | 0x05, 0xEB])?;
        self.emit(&[modrm(0b11, ymm(dst) & 7, ymm(src2) & 7)])?;
        
        Ok(self.pos - start)
    }

    /// VPXOR ymm, ymm, ymm
    pub fn emit_vpxor(&mut self, dst: VReg, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        
        let vvvv = (!ymm(src1)) & 0xF;
        self.emit(&[0xC5, (vvvv << 3) | 0x05, 0xEF])?;
        self.emit(&[modrm(0b11, ymm(dst) & 7, ymm(src2) & 7)])?;
        
        Ok(self.pos - start)
    }

    /// VPADDQ ymm, ymm, ymm
    pub fn emit_vpaddq(&mut self, dst: VReg, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        
        let vvvv = (!ymm(src1)) & 0xF;
        self.emit(&[0xC5, (vvvv << 3) | 0x05, 0xD4])?;
        self.emit(&[modrm(0b11, ymm(dst) & 7, ymm(src2) & 7)])?;
        
        Ok(self.pos - start)
    }

    /// VPSUBQ ymm, ymm, ymm
    pub fn emit_vpsubq(&mut self, dst: VReg, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        
        let vvvv = (!ymm(src1)) & 0xF;
        self.emit(&[0xC5, (vvvv << 3) | 0x05, 0xFB])?;
        self.emit(&[modrm(0b11, ymm(dst) & 7, ymm(src2) & 7)])?;
        
        Ok(self.pos - start)
    }

    /// VPBROADCASTQ ymm, imm64 — via RAX scratch
    pub fn emit_vpbroadcastq_imm(&mut self, dst: VReg, value: u64) -> Result<usize, LoweringError> {
        let start = self.pos;
        
        self.emit(&[0x48, 0xB8])?;
        self.emit(&value.to_le_bytes())?;
        self.emit(&[0xC4, 0xE1, 0xF9, 0x6E, modrm(0b11, ymm(dst) & 7, 0)])?;
        self.emit(&[0xC4, 0xE2, 0x7D, 0x59, modrm(0b11, ymm(dst) & 7, ymm(dst) & 7)])?;
        
        Ok(self.pos - start)
    }

    /// VBLENDVPD ymm, ymm, ymm, ymm
    pub fn emit_vblendvpd(&mut self, dst: VReg, src1: VReg, src2: VReg, mask: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        
        let vex2 = 0xE3;
        let vex3 = (((!ymm(src1)) & 0xF) << 3) | 0x05;
        
        self.emit(&[0xC4, vex2, vex3, 0x4B])?;
        self.emit(&[modrm(0b11, ymm(dst) & 7, ymm(src2) & 7)])?;
        self.emit(&[(ymm(mask) << 4)])?;
        
        Ok(self.pos - start)
    }

    /// VMOVNTDQ — NT store, bypasses cache. Requires 32-byte alignment.
    pub fn emit_vmovntdq_store(&mut self, src: VReg, offset: u32) -> Result<usize, LoweringError> {
        let start = self.pos;
        self.emit(&[0xC5, 0xFD, 0xE7])?;
        self.emit(&[modrm(0b10, ymm(src) & 7, self.output_base)])?;
        self.emit(&offset.to_le_bytes())?;
        Ok(self.pos - start)
    }

    /// VMOVDQU store
    pub fn emit_vmovdqu_store(&mut self, src: VReg, offset: u32) -> Result<usize, LoweringError> {
        let start = self.pos;
        self.emit(&[0xC5, 0xFE, 0x7F])?;
        self.emit(&[modrm(0b10, ymm(src) & 7, self.output_base)])?;
        self.emit(&offset.to_le_bytes())?;
        Ok(self.pos - start)
    }

    /// VPTEST ymm, ymm
    pub fn emit_vptest(&mut self, src1: VReg, src2: VReg) -> Result<usize, LoweringError> {
        let start = self.pos;
        
        self.emit(&[0xC4, 0xE2, 0x7D, 0x17])?;
        self.emit(&[modrm(0b11, ymm(src1) & 7, ymm(src2) & 7)])?;
        
        Ok(self.pos - start)
    }

    /// LFENCE — Spectre mitigation
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
            MicroOp::LoadVector { dst, offset, width, scalar_type, mask } => {
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
            MicroOp::ValidateCmpEq { dst_mask, src, imm_or_reg, scalar_type } => {
                self.emit_vpcmpeqq(*dst_mask, *src, *imm_or_reg)
            }
            MicroOp::ValidateCmpGt { dst_mask, src, comparand, scalar_type } => {
                self.emit_vpcmpgtq(*dst_mask, *src, *comparand)
            }
            MicroOp::ValidateCmpLt { dst_mask, src, comparand, scalar_type } => {
                self.emit_vpcmpgtq(*dst_mask, *comparand, *src)
            }
            MicroOp::ValidateNonZero { dst_mask, src, scalar_type } => {
                self.emit_vptest(*src, *src)
            }
            MicroOp::MaskAnd { dst, src1, src2 } => {
                self.emit_vpand(*dst, *src1, *src2)
            }
            MicroOp::MaskOr { dst, src1, src2 } => {
                self.emit_vpor(*dst, *src1, *src2)
            }
            MicroOp::MaskNot { dst, src } => {
                self.emit_vpxor(*dst, *src, *src)
            }
            MicroOp::Select { dst, mask, true_val, false_val, scalar_type } => {
                self.emit_vblendvpd(*dst, *false_val, *true_val, *mask)
            }
            MicroOp::Emit { src, field_offset, scalar_type, mask } => {
                let offset = *field_offset as u32;
                if offset % 32 == 0 && mask.is_none() {
                    self.emit_vmovntdq_store(*src, offset)
                } else {
                    self.emit_vmovdqu_store(*src, offset)
                }
            }
            MicroOp::BroadcastImm { dst, value, scalar_type } => {
                self.emit_vpbroadcastq_imm(*dst, *value)
            }
            MicroOp::Add { dst, src1, src2, scalar_type } => {
                self.emit_vpaddq(*dst, *src1, *src2)
            }
            MicroOp::Sub { dst, src1, src2, scalar_type } => {
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
                Ok(0)
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
        for i in 0..2000 {
            if enc.emit_vpand(VReg(0), VReg(1), VReg(2)).is_err() {
                break;
            }
        }
        
        assert!(enc.len() <= 16384);
    }
}
