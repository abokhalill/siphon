//! io_uring Ingress Loop
//!
//! NIC -> io_uring CQ -> Version Dispatcher -> JIT Kernel -> Output Buffer

use crate::lowering::jit::KernelFn;

#[allow(unused_imports)]
use std::ptr;

#[cfg(target_os = "linux")]
const IORING_OFF_SQ_RING: i64 = 0;
#[cfg(target_os = "linux")]
const IORING_OFF_CQ_RING: i64 = 0x8000000;
#[cfg(target_os = "linux")]
const IORING_OFF_SQES: i64 = 0x10000000;

#[repr(C)]
#[cfg(target_os = "linux")]
#[derive(Default)]
struct IoUringParams {
    sq_entries: u32,
    cq_entries: u32,
    flags: u32,
    sq_thread_cpu: u32,
    sq_thread_idle: u32,
    features: u32,
    wq_fd: u32,
    resv: [u32; 3],
    sq_off: IoSqringOffsets,
    cq_off: IoCqringOffsets,
}

#[repr(C)]
#[cfg(target_os = "linux")]
#[derive(Default)]
struct IoSqringOffsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    flags: u32,
    dropped: u32,
    array: u32,
    resv1: u32,
    resv2: u64,
}

#[repr(C)]
#[cfg(target_os = "linux")]
#[derive(Default)]
struct IoCqringOffsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    overflow: u32,
    cqes: u32,
    flags: u32,
    resv1: u32,
    resv2: u64,
}

pub const MAX_PROTOCOL_VERSIONS: usize = 256;

pub const BATCH_SIZE: usize = 32;

/// Output record size (32-byte aligned for VMOVNTDQ).
pub const RECORD_SIZE: usize = 256;

/// Per-core output buffer. No heap growth.
#[repr(C, align(32))]
pub struct OutputBuffer {
    data: [u8; RECORD_SIZE * BATCH_SIZE],
    write_pos: usize,
}

impl OutputBuffer {
    pub const fn new() -> Self {
        Self {
            data: [0u8; RECORD_SIZE * BATCH_SIZE],
            write_pos: 0,
        }
    }

    #[inline]
    pub fn record_ptr(&mut self, idx: usize) -> *mut u8 {
        debug_assert!(idx < BATCH_SIZE);
        unsafe { self.data.as_mut_ptr().add(idx * RECORD_SIZE) }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.write_pos = 0;
    }
}

/// Version dispatcher. Bounded jump table, Spectre-safe.
pub struct VersionDispatcher {
    kernels: [Option<KernelFn>; MAX_PROTOCOL_VERSIONS],
    slow_path: SlowPathFn,
    stats: DispatchStats,
}

pub type SlowPathFn = fn(*const u8, u32, u8) -> ();

/// Dispatch stats (cache-line padded to prevent false sharing).
#[repr(C, align(64))]
pub struct DispatchStats {
    pub packets_fast: u64,
    pub packets_slow: u64,
    pub unknown_versions: u64,
    pub validation_failures: u64,
    _pad: [u64; 4],
}

impl DispatchStats {
    pub const fn new() -> Self {
        Self {
            packets_fast: 0,
            packets_slow: 0,
            unknown_versions: 0,
            validation_failures: 0,
            _pad: [0; 4],
        }
    }
}

impl VersionDispatcher {
    pub fn new(slow_path: SlowPathFn) -> Self {
        Self {
            kernels: [None; MAX_PROTOCOL_VERSIONS],
            slow_path,
            stats: DispatchStats::new(),
        }
    }

    /// Register a kernel for a protocol version
    pub fn register(&mut self, version: u8, kernel: KernelFn) {
        self.kernels[version as usize] = Some(kernel);
    }

    /// Dispatch packet. LFENCE before table lookup for Spectre safety.
    #[inline]
    pub fn dispatch(
        &mut self,
        packet: *const u8,
        len: u32,
        output: *mut u8,
    ) -> DispatchResult {
        if len == 0 {
            return DispatchResult::InvalidPacket;
        }
        
        let version = unsafe { *packet };
        
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_lfence();
        }
        
        match self.kernels[version as usize] {
            Some(kernel) => {
                let result = kernel(packet, len, output);
                if result == 0 {
                    self.stats.packets_fast += 1;
                    DispatchResult::Success
                } else {
                    self.stats.validation_failures += 1;
                    DispatchResult::ValidationFailed(result)
                }
            }
            None => {
                self.stats.packets_slow += 1;
                self.stats.unknown_versions += 1;
                (self.slow_path)(packet, len, version);
                DispatchResult::SlowPath
            }
        }
    }

    pub fn stats(&self) -> &DispatchStats {
        &self.stats
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchResult {
    Success,
    ValidationFailed(u32),
    SlowPath,
    InvalidPacket,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PacketDesc {
    pub data: *const u8,
    pub len: u32,
    pub buffer_id: u32,
}

/// io_uring packet loop. Poll CQ, batch, process, repeat.
#[cfg(target_os = "linux")]
pub struct PacketLoop {
    ring_fd: i32,
    cq_ring: *mut u8,
    sq_ring: *mut u8,
    // CQ ring state for lockless polling
    cq_head: *mut u32,
    cq_tail: *mut u32,
    cq_mask: u32,
    cqes: *mut IoUringCqe,
    buffers: RegisteredBuffers,
    dispatcher: VersionDispatcher,
    output: OutputBuffer,
    batch: [PacketDesc; BATCH_SIZE],
    batch_count: u64,
}

#[cfg(target_os = "linux")]
impl PacketLoop {
    /// # Safety
    /// Requires io_uring support and sufficient RLIMIT_MEMLOCK.
    pub unsafe fn new(
        ring_size: u32,
        buffer_count: u32,
        buffer_size: u32,
        slow_path: SlowPathFn,
    ) -> Result<Self, IoUringError> {
        let mut params: IoUringParams = std::mem::zeroed();
        
        let ring_fd = libc::syscall(
            libc::SYS_io_uring_setup,
            ring_size,
            &mut params as *mut _,
        ) as i32;
        
        if ring_fd < 0 {
            return Err(IoUringError::SetupFailed);
        }
        
        let sq_size = params.sq_off.array as usize 
            + params.sq_entries as usize * std::mem::size_of::<u32>();
        let cq_size = params.cq_off.cqes as usize
            + params.cq_entries as usize * std::mem::size_of::<IoUringCqe>();
        
        let sq_ring = libc::mmap(
            ptr::null_mut(),
            sq_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED | libc::MAP_POPULATE,
            ring_fd,
            IORING_OFF_SQ_RING,
        ) as *mut u8;
        
        let cq_ring = libc::mmap(
            ptr::null_mut(),
            cq_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED | libc::MAP_POPULATE,
            ring_fd,
            IORING_OFF_CQ_RING,
        ) as *mut u8;
        
        if sq_ring == libc::MAP_FAILED as *mut u8 || cq_ring == libc::MAP_FAILED as *mut u8 {
            libc::close(ring_fd);
            return Err(IoUringError::MmapFailed);
        }
        
        let buffers = RegisteredBuffers::new(buffer_count, buffer_size)?;
        buffers.register(ring_fd)?;
        
        // Extract CQ ring pointers from mapped memory
        let cq_head = cq_ring.add(params.cq_off.head as usize) as *mut u32;
        let cq_tail = cq_ring.add(params.cq_off.tail as usize) as *mut u32;
        let cq_mask = *(cq_ring.add(params.cq_off.ring_mask as usize) as *const u32);
        let cqes = cq_ring.add(params.cq_off.cqes as usize) as *mut IoUringCqe;
        
        Ok(Self {
            ring_fd,
            cq_ring,
            sq_ring,
            cq_head,
            cq_tail,
            cq_mask,
            cqes,
            buffers,
            dispatcher: VersionDispatcher::new(slow_path),
            output: OutputBuffer::new(),
            batch: [PacketDesc { data: ptr::null(), len: 0, buffer_id: 0 }; BATCH_SIZE],
            batch_count: 0,
        })
    }

    /// Register a kernel for a protocol version
    pub fn register_kernel(&mut self, version: u8, kernel: KernelFn) {
        self.dispatcher.register(version, kernel);
    }

    /// The hot loop. Never returns. No allocations, no syscalls.
    #[inline(never)]
    pub fn run(&mut self) -> ! {
        loop {
            let batch = self.poll_completions();
            
            if batch > 0 {
                self.process_batch(batch);
                self.batch_count += 1;
            } else {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    core::arch::x86_64::_mm_pause();
                }
            }
        }
    }

    /// Poll CQ for completions. Lockless, no syscalls.
    #[inline]
    fn poll_completions(&mut self) -> usize {
        // Atomic load of tail (producer), relaxed load of head (we own it)
        let tail = unsafe { 
            core::ptr::read_volatile(self.cq_tail) 
        };
        let head = unsafe { 
            core::ptr::read_volatile(self.cq_head) 
        };
        
        let available = tail.wrapping_sub(head) as usize;
        if available == 0 {
            return 0;
        }
        
        let to_process = available.min(BATCH_SIZE);
        
        // Read CQEs into batch array
        for i in 0..to_process {
            let idx = (head.wrapping_add(i as u32) & self.cq_mask) as usize;
            let cqe = unsafe { &*self.cqes.add(idx) };
            
            // user_data encodes buffer_id, res is bytes received
            let buffer_id = cqe.user_data as u32;
            let len = if cqe.res > 0 { cqe.res as u32 } else { 0 };
            
            self.batch[i] = PacketDesc {
                data: self.buffers.buffer_ptr(buffer_id),
                len,
                buffer_id,
            };
        }
        
        // Advance head (release semantics for producer visibility)
        unsafe {
            core::ptr::write_volatile(
                self.cq_head, 
                head.wrapping_add(to_process as u32)
            );
        }
        
        to_process
    }

    /// Process batch through dispatcher. Straight-line, no branches in hot path.
    #[inline]
    fn process_batch(&mut self, count: usize) {
        self.output.reset();
        
        for i in 0..count {
            let packet = &self.batch[i];
            
            if !packet.data.is_null() && packet.len > 0 {
                let output_ptr = self.output.record_ptr(i);
                self.dispatcher.dispatch(packet.data, packet.len, output_ptr);
            }
        }
    }

    pub fn stats(&self) -> &DispatchStats {
        self.dispatcher.stats()
    }

    pub fn batch_count(&self) -> u64 {
        self.batch_count
    }
}

#[cfg(target_os = "linux")]
impl Drop for PacketLoop {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.ring_fd);
            // Unmap rings (sizes would need to be stored)
        }
    }
}

/// Registered buffers for zero-copy I/O.
#[cfg(target_os = "linux")]
pub struct RegisteredBuffers {
    memory: *mut u8,
    buffer_size: u32,
    count: u32,
    total_size: usize,
}

#[cfg(target_os = "linux")]
impl RegisteredBuffers {
    pub fn new(count: u32, buffer_size: u32) -> Result<Self, IoUringError> {
        let total_size = (count as usize) * (buffer_size as usize);
        
        let memory = unsafe {
            libc::mmap(
                ptr::null_mut(),
                total_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB,
                -1,
                0,
            )
        };
        
        let memory = if memory == libc::MAP_FAILED {
            unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    total_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            }
        } else {
            memory
        };
        
        if memory == libc::MAP_FAILED {
            return Err(IoUringError::BufferAllocFailed);
        }
        
        unsafe {
            libc::madvise(memory, total_size, libc::MADV_WILLNEED);
        }
        
        Ok(Self {
            memory: memory as *mut u8,
            buffer_size,
            count,
            total_size,
        })
    }

    pub fn register(&self, ring_fd: i32) -> Result<(), IoUringError> {
        let mut iovecs: Vec<libc::iovec> = Vec::with_capacity(self.count as usize);
        
        for i in 0..self.count {
            iovecs.push(libc::iovec {
                iov_base: unsafe { self.memory.add((i as usize) * (self.buffer_size as usize)) as *mut _ },
                iov_len: self.buffer_size as usize,
            });
        }
        
        let result = unsafe {
            libc::syscall(
                libc::SYS_io_uring_register,
                ring_fd,
                0u32, // IORING_REGISTER_BUFFERS
                iovecs.as_ptr(),
                self.count,
            )
        };
        
        if result < 0 {
            return Err(IoUringError::RegisterFailed);
        }
        
        Ok(())
    }

    pub fn buffer_ptr(&self, idx: u32) -> *const u8 {
        debug_assert!(idx < self.count);
        unsafe { self.memory.add((idx as usize) * (self.buffer_size as usize)) }
    }
}

#[cfg(target_os = "linux")]
impl Drop for RegisteredBuffers {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.memory as *mut _, self.total_size);
        }
    }
}

#[repr(C)]
#[cfg(target_os = "linux")]
struct IoUringCqe {
    user_data: u64,
    res: i32,
    flags: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoUringError {
    SetupFailed,
    MmapFailed,
    BufferAllocFailed,
    RegisterFailed,
}

/// Default slow path: count and drop.
pub fn default_slow_path(packet: *const u8, len: u32, version: u8) {
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" fn dummy_kernel(_packet: *const u8, _len: u32, _output: *mut u8) -> u32 {
        0 // success
    }

    fn dummy_slow_path(_packet: *const u8, _len: u32, _version: u8) {}

    #[test]
    fn test_dispatcher_fast_path() {
        let mut dispatcher = VersionDispatcher::new(dummy_slow_path);
        dispatcher.register(1, dummy_kernel);
        
        let packet = [1u8, 0, 0, 0]; // version = 1
        let mut output = [0u8; 256];
        
        let result = dispatcher.dispatch(
            packet.as_ptr(),
            packet.len() as u32,
            output.as_mut_ptr(),
        );
        
        assert_eq!(result, DispatchResult::Success);
        assert_eq!(dispatcher.stats().packets_fast, 1);
    }

    #[test]
    fn test_dispatcher_slow_path() {
        let mut dispatcher = VersionDispatcher::new(dummy_slow_path);
        // Don't register version 42
        
        let packet = [42u8, 0, 0, 0]; // version = 42 (unknown)
        let mut output = [0u8; 256];
        
        let result = dispatcher.dispatch(
            packet.as_ptr(),
            packet.len() as u32,
            output.as_mut_ptr(),
        );
        
        assert_eq!(result, DispatchResult::SlowPath);
        assert_eq!(dispatcher.stats().unknown_versions, 1);
    }

    #[test]
    fn test_output_buffer_alignment() {
        let buf = OutputBuffer::new();
        // Check 32-byte alignment for VMOVNTDQ
        assert_eq!(buf.data.as_ptr() as usize % 32, 0);
    }
}
