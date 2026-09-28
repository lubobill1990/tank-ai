//! Diagnostic-only fixed-reference randomness. Not a training backend.

#[derive(Default)]
pub(crate) struct RandomReplay {
    samples: Vec<(u8, u8)>,
    consumed: usize,
    error: u32,
}

impl RandomReplay {
    pub(crate) fn new(samples: &[(u8, u8)]) -> Self {
        Self { samples: samples.to_vec(), consumed: 0, error: 0 }
    }

    pub(crate) fn take(&mut self, cursor: u8) -> u8 {
        let index = self.consumed;
        self.consumed += 1;
        let Some(&(value, expected_cursor)) = self.samples.get(index) else {
            self.error |= 1; // Unexpected extra RNG call; result must be rejected.
            return 0;
        };
        if cursor != expected_cursor {
            self.error |= 2; // Call-order/state divergence.
        }
        value
    }

    pub(crate) fn status(&self) -> u32 {
        self.error | if self.consumed != self.samples.len() { 4 } else { 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_reference_sequence_consumption_is_required() {
        let mut rng = RandomReplay::new(&[(91, 8), (17, 9)]);
        assert_eq!(rng.status(), 4);
        assert_eq!(rng.take(8), 91);
        assert_eq!(rng.take(9), 17);
        assert_eq!(rng.status(), 0);
    }

    #[test]
    fn wrong_cursor_or_extra_call_cannot_silently_pass() {
        let mut rng = RandomReplay::new(&[(91, 8)]);
        assert_eq!(rng.take(9), 91);
        assert_eq!(rng.status(), 2);
        rng.take(10);
        assert_ne!(rng.status(), 0);
    }

    #[test]
    fn exhausted_tape_cannot_hang_item_position_rejection() {
        let mut ram = [0u8; 2048];
        ram[0x68] = 0x80;
        ram[0x51] = 3;
        ram[0x80] = 1;
        ram[0x100] = 10;
        ram[0xa0] = 0x80;
        ram[0x90] = 48;
        ram[0x98] = 48;
        ram[0xa2] = 0xa2;
        ram[0xaa] = 0x84;
        ram[0x92] = 88;
        ram[0x9a] = 100;
        ram[0xcc] = 0x40;
        ram[0xb8] = 88;
        ram[0xc2] = 110;
        let mut engine = crate::engine::Engine::new(ram);
        assert_ne!(engine.step_with_reference_rng(0, &[]), 0);
    }

    #[test]
    fn logical_update_uses_given_clock_without_a_second_increment() {
        let mut ram = [0u8; 2048];
        ram[0x68] = 0x80;
        ram[0x51] = 3;
        ram[0x80] = 1;
        ram[0xa0] = 0x80;
        ram[0x90] = 88;
        ram[0x98] = 160;
        ram[0x0b] = 32;
        let mut engine = crate::engine::Engine::new(ram);
        assert_eq!(engine.logic_with_reference_rng(128, 128, 33, 7, &[]), 0);
        assert_eq!(engine.ram[0x0b], 33);
        assert_eq!(engine.ram[0x0a], 7);
        assert_eq!(engine.ram[0x90], 89);
        assert_eq!(engine.frame, 1);
    }
}
