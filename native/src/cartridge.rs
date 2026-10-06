//! Cartridge-formula training RNG: the mirror's own $D44D arithmetic over its
//! own zero page, with seed-derived phase and an optional recorded scratch
//! overlay approximating between-update sound/NMI zero-page writes.
//!
//! This is cartridge-rng-v1. It is a training randomness backend, not NES RNG
//! equivalence: overlay rows are per-logical-frame boundary captures, not
//! instruction-interleaved writes, and the phase expansion below is a seed
//! derivation convention, not a cartridge mechanism.
use std::sync::Arc;

/// Zero-page addresses the gameplay mirror itself reads or writes (including
/// observation fields) must never be overlaid: the overlay may only touch
/// bytes whose sole consumer is the shared RNG's walking cursor.
pub(crate) fn overlay_allowed(address: u8) -> bool {
    !matches!(address,
        0x00 | 0x06 | 0x08 | 0x0a | 0x0b | 0x0f | 0x10 | 0x11 | 0x12
        | 0x13..=0x22
        | 0x45 | 0x46 | 0x49
        | 0x51..=0x5b
        | 0x62 | 0x64..=0x68 | 0x6a | 0x6c
        | 0x6f..=0x76
        | 0x7f..=0xef)
}

/// Recorded per-frame values for a fixed set of non-gameplay zero-page bytes.
/// Row r holds one byte per address, captured at a logical frame boundary.
pub struct ScratchTrace {
    addresses: Vec<u8>,
    values: Vec<u8>,
}

impl ScratchTrace {
    pub fn new(addresses: Vec<u8>, values: Vec<u8>) -> Result<Self, &'static str> {
        if addresses.is_empty() {
            return if values.is_empty() {
                Ok(Self { addresses, values })
            } else {
                Err("scratch values without addresses")
            };
        }
        let mut seen = [false; 256];
        for &address in &addresses {
            if !overlay_allowed(address) {
                return Err("scratch address collides with mirrored gameplay state");
            }
            if std::mem::replace(&mut seen[usize::from(address)], true) {
                return Err("duplicate scratch address");
            }
        }
        if values.is_empty() || values.len() % addresses.len() != 0 {
            return Err("scratch values are not whole rows");
        }
        Ok(Self { addresses, values })
    }

    #[cfg(test)]
    pub fn empty() -> Arc<Self> {
        Arc::new(Self { addresses: Vec::new(), values: Vec::new() })
    }

    pub fn rows(&self) -> usize {
        if self.addresses.is_empty() { 0 } else { self.values.len() / self.addresses.len() }
    }

    fn row(&self, index: usize) -> (&[u8], &[u8]) {
        let width = self.addresses.len();
        (&self.addresses, &self.values[index * width..(index + 1) * width])
    }
}

fn split_mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

/// Seed-derived start phase for the cartridge formula and scratch replay.
/// This mimics the console's only true entropy: when START was pressed, the
/// RNG state/cursor and the audio timeline are at an arbitrary phase.
pub(crate) struct CartridgeRng {
    pub(crate) seed: u64,
    pub(crate) trace: Arc<ScratchTrace>,
    rng_state: u8,
    rng_cursor: u8,
    phase: usize,
}

impl CartridgeRng {
    pub(crate) fn new(seed: u64, trace: Arc<ScratchTrace>) -> Self {
        // cartridge-rng-v1 expansion: three SplitMix64-v1 outputs, high bytes
        // for the $0F/$10 phase, the third taken modulo the trace length.
        let mut state = seed;
        let rng_state = (split_mix(&mut state) >> 56) as u8;
        let rng_cursor = (split_mix(&mut state) >> 56) as u8;
        let rows = trace.rows();
        let phase = if rows == 0 {
            0
        } else {
            (split_mix(&mut state) % rows as u64) as usize
        };
        Self { seed, trace, rng_state, rng_cursor, phase }
    }

    /// Overwrite only the RNG phase bytes; everything else stays snapshot RAM.
    pub(crate) fn apply_phase(&self, ram: &mut [u8]) {
        ram[0x0f] = self.rng_state;
        ram[0x10] = self.rng_cursor;
    }

    /// Replay one recorded scratch row for this logical frame.
    pub(crate) fn apply_scratch(&self, ram: &mut [u8], frame: u32) {
        let rows = self.trace.rows();
        if rows == 0 {
            return;
        }
        let (addresses, values) = self.trace.row((self.phase + frame as usize) % rows);
        for (&address, &value) in addresses.iter().zip(values) {
            ram[usize::from(address)] = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_denies_every_mirrored_gameplay_byte_and_allows_known_gaps() {
        for denied in [0x00u8, 0x06, 0x0a, 0x0f, 0x10, 0x17, 0x1b, 0x45, 0x51,
                       0x5b, 0x62, 0x68, 0x73, 0x7f, 0x90, 0xa0, 0xcc, 0xef] {
            assert!(!overlay_allowed(denied), "{denied:#x} must stay protected");
        }
        for allowed in [0x01u8, 0x05, 0x07, 0x09, 0x0c, 0x23, 0x44, 0x47, 0x4a,
                        0x5c, 0x63, 0x69, 0x6d, 0x77, 0x7e, 0xf0, 0xff] {
            assert!(overlay_allowed(allowed), "{allowed:#x} should be overlayable");
        }
    }

    #[test]
    fn trace_rejects_collisions_duplicates_and_ragged_rows() {
        assert!(ScratchTrace::new(vec![0x0f], vec![1]).is_err());
        assert!(ScratchTrace::new(vec![0x01, 0x01], vec![1, 2]).is_err());
        assert!(ScratchTrace::new(vec![0x01, 0x02], vec![1, 2, 3]).is_err());
        assert!(ScratchTrace::new(vec![0x01], Vec::new()).is_err());
        assert!(ScratchTrace::new(Vec::new(), vec![1]).is_err());
        assert_eq!(ScratchTrace::new(Vec::new(), Vec::new()).unwrap().rows(), 0);
        assert_eq!(ScratchTrace::new(vec![0x01, 0x02], vec![1, 2, 3, 4]).unwrap().rows(), 2);
    }

    #[test]
    fn expansion_is_deterministic_per_seed_and_phase_respects_trace_length() {
        let trace = Arc::new(ScratchTrace::new(vec![0x01], vec![9, 8, 7]).unwrap());
        let a = CartridgeRng::new(5, trace.clone());
        let b = CartridgeRng::new(5, trace.clone());
        assert_eq!((a.rng_state, a.rng_cursor, a.phase), (b.rng_state, b.rng_cursor, b.phase));
        assert!(a.phase < 3);
        let c = CartridgeRng::new(6, trace);
        assert_ne!((a.rng_state, a.rng_cursor), (c.rng_state, c.rng_cursor));
        let empty = CartridgeRng::new(5, ScratchTrace::empty());
        assert_eq!(empty.phase, 0);
    }

    #[test]
    fn scratch_rows_cycle_from_the_seeded_phase() {
        let trace = Arc::new(ScratchTrace::new(vec![0x01, 0x02], vec![1, 2, 3, 4, 5, 6]).unwrap());
        let rng = CartridgeRng::new(0, trace);
        let mut ram = [0u8; 2048];
        let mut rows = Vec::new();
        for frame in 0..4 {
            rng.apply_scratch(&mut ram, frame);
            rows.push((ram[1], ram[2]));
        }
        assert_eq!(rows[0], rows[3]);
        assert_ne!(rows[0], rows[1]);
        assert!(rows.iter().all(|r| matches!(r, (1, 2) | (3, 4) | (5, 6))));
    }
}
