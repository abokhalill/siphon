//! Zero-Overhead Telemetry

pub const CACHE_LINE_SIZE: usize = 64;

pub const MAX_CORES: usize = 128;

/// Per-core telemetry. Cache-line padded.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct CoreTelemetry {
    pub packets: u64,
    pub batches: u64,
    pub total_cycles: u64,
    pub min_cycles: u64,
    pub max_cycles: u64,
    pub validation_failures: u64,
    pub unknown_versions: u64,
    _reserved: u64,
}

impl CoreTelemetry {
    pub const fn new() -> Self {
        Self {
            packets: 0,
            batches: 0,
            total_cycles: 0,
            min_cycles: u64::MAX,
            max_cycles: 0,
            validation_failures: 0,
            unknown_versions: 0,
            _reserved: 0,
        }
    }

    #[inline]
    pub fn record_batch(&mut self, packets: u64, cycles: u64) {
        self.packets += packets;
        self.batches += 1;
        self.total_cycles += cycles;
        
        if cycles < self.min_cycles {
            self.min_cycles = cycles;
        }
        if cycles > self.max_cycles {
            self.max_cycles = cycles;
        }
    }

    #[inline]
    pub fn avg_cycles_per_packet(&self) -> u64 {
        if self.packets == 0 {
            0
        } else {
            self.total_cycles / self.packets
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Global telemetry array.
pub struct TelemetryArray {
    cores: [CoreTelemetry; MAX_CORES],
}

impl TelemetryArray {
    pub const fn new() -> Self {
        Self {
            cores: [CoreTelemetry::new(); MAX_CORES],
        }
    }

    #[inline]
    pub fn core_mut(&mut self, core_id: usize) -> &mut CoreTelemetry {
        debug_assert!(core_id < MAX_CORES);
        &mut self.cores[core_id]
    }

    #[inline]
    pub fn core(&self, core_id: usize) -> &CoreTelemetry {
        debug_assert!(core_id < MAX_CORES);
        &self.cores[core_id]
    }

    pub fn aggregate(&self) -> AggregateStats {
        let mut agg = AggregateStats::new();
        
        for core in &self.cores {
            if core.packets > 0 {
                agg.total_packets += core.packets;
                agg.total_batches += core.batches;
                agg.total_cycles += core.total_cycles;
                agg.validation_failures += core.validation_failures;
                agg.unknown_versions += core.unknown_versions;
                agg.active_cores += 1;
                
                if core.min_cycles < agg.min_cycles {
                    agg.min_cycles = core.min_cycles;
                }
                if core.max_cycles > agg.max_cycles {
                    agg.max_cycles = core.max_cycles;
                }
            }
        }
        
        agg
    }
}

/// Aggregated stats.
#[derive(Clone, Copy, Debug)]
pub struct AggregateStats {
    pub total_packets: u64,
    pub total_batches: u64,
    pub total_cycles: u64,
    pub min_cycles: u64,
    pub max_cycles: u64,
    pub validation_failures: u64,
    pub unknown_versions: u64,
    pub active_cores: u64,
}

impl AggregateStats {
    pub const fn new() -> Self {
        Self {
            total_packets: 0,
            total_batches: 0,
            total_cycles: 0,
            min_cycles: u64::MAX,
            max_cycles: 0,
            validation_failures: 0,
            unknown_versions: 0,
            active_cores: 0,
        }
    }

    pub fn avg_cycles_per_packet(&self) -> u64 {
        if self.total_packets == 0 {
            0
        } else {
            self.total_cycles / self.total_packets
        }
    }

    pub fn packets_per_second(&self, elapsed_ns: u64, cpu_freq_ghz: f64) -> f64 {
        if elapsed_ns == 0 {
            return 0.0;
        }
        (self.total_packets as f64) / (elapsed_ns as f64 / 1_000_000_000.0)
    }
}

/// RDTSC timer.
pub struct CycleTimer {
    start: u64,
}

impl CycleTimer {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    pub fn start() -> Self {
        let start = unsafe { core::arch::x86_64::_rdtsc() };
        Self { start }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn start() -> Self {
        Self { start: 0 }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    pub fn stop(&self) -> u64 {
        let end = unsafe { core::arch::x86_64::_rdtsc() };
        end.saturating_sub(self.start)
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn stop(&self) -> u64 {
        0
    }

    /// Stop with RDTSCP serialization for accurate measurement.
    #[inline]
    #[cfg(target_arch = "x86_64")]
    pub fn stop_serialized(&self) -> u64 {
        let mut aux: u32 = 0;
        let end = unsafe { core::arch::x86_64::__rdtscp(&mut aux) };
        end.saturating_sub(self.start)
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn stop_serialized(&self) -> u64 {
        0
    }
}

/// Batch timing helper.
pub struct BatchTimer<'a> {
    telemetry: &'a mut CoreTelemetry,
    timer: CycleTimer,
    packet_count: u64,
}

impl<'a> BatchTimer<'a> {
    #[inline]
    pub fn start(telemetry: &'a mut CoreTelemetry) -> Self {
        Self {
            telemetry,
            timer: CycleTimer::start(),
            packet_count: 0,
        }
    }

    #[inline]
    pub fn add_packets(&mut self, count: u64) {
        self.packet_count += count;
    }

    #[inline]
    pub fn record_validation_failure(&mut self) {
        self.telemetry.validation_failures += 1;
    }

    #[inline]
    pub fn record_unknown_version(&mut self) {
        self.telemetry.unknown_versions += 1;
    }
}

impl<'a> Drop for BatchTimer<'a> {
    #[inline]
    fn drop(&mut self) {
        let cycles = self.timer.stop();
        self.telemetry.record_batch(self.packet_count, cycles);
    }
}

/// Latency histogram. Power-of-2 buckets, O(1) insertion. Off by default.
#[derive(Clone)]
pub struct LatencyHistogram {
    buckets: [u64; 64],
    enabled: bool,
}

impl LatencyHistogram {
    pub const fn new() -> Self {
        Self {
            buckets: [0; 64],
            enabled: false,
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    #[inline]
    pub fn record(&mut self, cycles: u64) {
        if !self.enabled {
            return;
        }
        
        let bucket = if cycles == 0 {
            0
        } else {
            (64 - cycles.leading_zeros()).min(63) as usize
        };
        
        self.buckets[bucket] += 1;
    }

    pub fn percentile(&self, p: f64) -> u64 {
        let total: u64 = self.buckets.iter().sum();
        if total == 0 {
            return 0;
        }
        
        let target = (total as f64 * p) as u64;
        let mut cumulative = 0u64;
        
        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                // Return midpoint of bucket
                return 1u64 << i;
            }
        }
        
        u64::MAX
    }

    pub fn reset(&mut self) {
        self.buckets = [0; 64];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_telemetry_alignment() {
        assert_eq!(std::mem::size_of::<CoreTelemetry>(), CACHE_LINE_SIZE);
        assert_eq!(std::mem::align_of::<CoreTelemetry>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn test_batch_recording() {
        let mut telem = CoreTelemetry::new();
        
        telem.record_batch(10, 1000);
        telem.record_batch(20, 2000);
        
        assert_eq!(telem.packets, 30);
        assert_eq!(telem.batches, 2);
        assert_eq!(telem.total_cycles, 3000);
        assert_eq!(telem.avg_cycles_per_packet(), 100);
    }

    #[test]
    fn test_histogram_percentile() {
        let mut hist = LatencyHistogram::new();
        hist.enable();
        
        // Record some values
        for _ in 0..100 {
            hist.record(100); // bucket ~7
        }
        for _ in 0..100 {
            hist.record(1000); // bucket ~10
        }
        
        let p50 = hist.percentile(0.5);
        let p99 = hist.percentile(0.99);
        
        // p50 should be around 100-ish
        assert!(p50 >= 64 && p50 <= 256);
        // p99 should be around 1000-ish
        assert!(p99 >= 512 && p99 <= 2048);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_cycle_timer() {
        let timer = CycleTimer::start();
        
        // Do some work
        let mut x = 0u64;
        for i in 0..1000 {
            x = x.wrapping_add(i);
        }
        
        let cycles = timer.stop();
        
        // Should have taken some cycles
        assert!(cycles > 0);
        // But not too many (sanity check)
        assert!(cycles < 1_000_000);
        
        // Prevent optimization
        assert!(x > 0);
    }
}
