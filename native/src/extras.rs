//! Special-mechanics hooks, compared against local cartridge subroutines.
use crate::engine::Engine;

impl Engine {
    /// E181: sample before marking this tank's occupancy, not after movement.
    pub(crate) fn extras_ice_sample(&mut self, player: usize) {
        if !(0x80..0xe0).contains(&self.ram[0xa0 + player]) {
            return;
        }
        let cell = 0x400
            + usize::from(self.ram[0x98 + player] / 8) * 32
            + usize::from(self.ram[0x90 + player] / 8);
        let ice = if self.ram[cell] == 33 { 0x80 } else { 0 };
        self.ram[0x103 + player] = (self.ram[0x103 + player] & 0x7f) | ice;
    }

    /// DB75: input/status phase only; does not attempt movement or collision.
    pub(crate) fn extras_player_input(&mut self, player: usize, input: u8) {
        if self.ram[0x0b] & 3 == 2 || !(0x80..0xe0).contains(&self.ram[0xa0 + player]) {
            return;
        }
        let locked = self.ram[0x6f + player] != 0;
        if locked {
            self.ram[0x6f + player] -= 1;
        }
        let forced_coast = self.ram[0x103 + player] & 0x90 == 0x90;
        let face = if input & 0x80 != 0 {
            Some(3)
        } else if input & 0x40 != 0 {
            Some(1)
        } else if input & 0x20 != 0 {
            Some(2)
        } else if input & 0x10 != 0 {
            Some(0)
        } else {
            None
        };
        if locked || forced_coast || face.is_none() {
            self.ram[0] = 0x80;
            self.ram[0xa0 + player] = (self.ram[0xa0 + player] & 15) | 0x88;
            return;
        }
        let face = face.unwrap();
        self.ram[0] = face;
        if self.ram[0x103 + player] & 0x9f == 0x80 {
            self.ram[0x103 + player] = 0x9c;
        }
        if (self.ram[0xa0 + player] ^ face) & 1 != 0 {
            for address in [0x90 + player, 0x98 + player] {
                self.ram[address] = self.ram[address].wrapping_add(4) & 0xf8;
            }
        }
        self.ram[0xa0 + player] = 0xa0 | face;
    }

    /// DC52: only for the eligible player's 80..8F movement state.
    /// True requests ordinary collision/movement, including its final B0 XOR 4.
    pub(crate) fn extras_player_idle_step(&mut self, player: usize) -> bool {
        let inertia = self.ram[0x103 + player];
        if inertia & 0x80 != 0 && inertia & 0x7f != 0 {
            self.ram[0x103 + player] -= 1;
            self.ram[0xb0 + player] ^= 4;
            true
        } else {
            let status = self.ram[0xa0 + player].wrapping_sub(4);
            self.ram[0xa0 + player] = if status & 12 == 0 {
                self.ram[0] = 0xa0;
                (status & 15) | 0xa0
            } else {
                status
            };
            false
        }
    }

    /// E8BE: called on the red tank hit, before either armor decrement.
    pub(crate) fn extras_drop_item(&mut self) {
        loop {
            for address in [0x86, 0x87] {
                let grid = self.random_byte() & 3;
                // The shared RNG can later read this scratch byte.
                self.ram[0] = grid;
                self.ram[address] = 48 + grid * 48;
            }
            #[cfg(feature = "reference-rng")]
            if self.reference_rng_failed() {
                // Do not spin in the rejection loop after a failed diagnostic tape.
                return;
            }
            self.ram[0x88] = 0xff;
            self.ram[0x62] = 0;
            self.ram[0x49] = 1;
            for player in (0..2).rev() {
                let status = self.ram[0xa0 + player];
                let dx = self.ram[0x90 + player].wrapping_sub(self.ram[0x86]) as i8;
                let dy = self.ram[0x98 + player].wrapping_sub(self.ram[0x87]) as i8;
                if (0x80..0xe0).contains(&status)
                    && dx.unsigned_abs() < 12
                    && dy.unsigned_abs() < 12
                {
                    self.ram[0x62] = 50;
                    break;
                }
                self.ram[0x49] = self.ram[0x49].wrapping_sub(1);
            }
            if self.ram[0x62] == 0 {
                break;
            }
        }
        let selection = self.random_byte() & 7;
        self.ram[0x88] = [0, 1, 2, 3, 4, 5, 4, 3][usize::from(selection)];
        self.ram[0x62] = 0;
    }

    /// E23B: after gameplay, including pickups made in this frame.
    pub(crate) fn extras_item_tick(&mut self) {
        if self.ram[0x86] != 0 && self.ram[0x62] != 0 {
            self.ram[0x62] -= 1;
            if self.ram[0x62] == 0 {
                self.ram[0x86] = 0;
            }
        }
    }

    /// DBF1: before movement, so an expiring freeze releases enemies now.
    pub(crate) fn extras_freeze_tick(&mut self) {
        if self.ram[0x0b] & 63 == 0 && self.ram[0x100] != 0 {
            self.ram[0x100] -= 1;
        }
    }

    /// E27C: after movement, before projectile/tank impact and pickup.
    pub(crate) fn extras_shield_tick(&mut self) {
        if self.ram[0x0b] & 63 == 0 {
            for address in [0x89, 0x8a] {
                if self.ram[address] != 0 {
                    self.ram[address] -= 1;
                }
            }
        }
    }

    /// E2A9: after bullet motion, before terrain impact and pickup.
    pub(crate) fn extras_shovel_tick(&mut self) {
        if self.ram[0x45] == 0 || self.ram[0x0b] & 15 != 0 {
            return;
        }
        if self.ram[0x0b] & 63 == 0 {
            self.ram[0x45] -= 1;
            if self.ram[0x45] == 0 {
                self.extras_fortress(false);
                return;
            }
        }
        if self.ram[0x45] < 4 {
            self.extras_fortress(self.ram[0x0b] & 16 != 0);
        }
    }

    /// Both ring routines replace the surrounding ground and eagle as well.
    pub(crate) fn extras_fortress(&mut self, steel: bool) {
        let wall = if steel { 16 } else { 15 };
        for (start, cells) in [
            (0x70c, [0, 0, 0, 0, 0, 0]),
            (0x72c, [0, wall, wall, wall, wall, 0]),
            (0x74c, [0, wall, 200, 202, wall, 0]),
            (0x76c, [0, wall, 201, 203, wall, 0]),
        ] {
            self.ram[start..start + 6].copy_from_slice(&cells);
        }
        self.ram[0x7f3] = if steel { 63 } else { 0 };
        self.ram[0x7f4] = (self.ram[0x7f4] & 0xcc) | if steel { 0x33 } else { 0 };
    }

    /// D138: after score addition; the cartridge checks the ten-thousands digit.
    pub(crate) fn extras_score_life(&mut self) {
        if self.ram[0x68] != 0x80 {
            return;
        }
        for player in 0..2 {
            if player == 1 && self.ram[0x83] == 0 {
                return;
            }
            if self.ram[0x66 + player] == 0 && self.ram[0x17 + player * 8] >= 2 {
                self.ram[0x51 + player] = self.ram[0x51 + player].wrapping_add(1);
                self.ram[0x66 + player] = 1;
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> Engine {
        let mut ram = [0; 2048];
        ram[0x68] = 0x80;
        ram[0x51] = 3;
        ram[0x80] = 1;
        Engine::new(ram)
    }

    #[test]
    fn ice_samples_center_tile_and_preserves_remaining_inertia() {
        let mut e = engine();
        e.ram[0xa0] = 0xa0;
        e.ram[0x90] = 88;
        e.ram[0x98] = 160;
        e.ram[0x103] = 7;
        e.ram[0x68b] = 33;
        e.extras_ice_sample(0);
        assert_eq!(e.ram[0x103], 0x87);
        e.ram[0x68b] = 0xa1;
        e.extras_ice_sample(0);
        assert_eq!(e.ram[0x103], 7);
        e.ram[0xa0] = 0x73;
        e.ram[0x68b] = 33;
        e.extras_ice_sample(0);
        assert_eq!(e.ram[0x103], 7);
    }

    #[test]
    fn starting_on_ice_latches_twenty_eight_and_perpendicular_turn_snaps_both_axes() {
        let mut e = engine();
        e.ram[0x0b] = 1;
        e.ram[0xa0] = 0xa3;
        e.ram[0x90] = 93;
        e.ram[0x98] = 101;
        e.ram[0x103] = 0x80;
        e.extras_player_input(0, 16);
        assert_eq!(
            (e.ram[0xa0], e.ram[0x90], e.ram[0x98], e.ram[0x103]),
            (0xa0, 96, 104, 0x9c)
        );
        e.extras_player_input(0, 64);
        assert_eq!((e.ram[0xa0], e.ram[0x103], e.ram[0]), (0x88, 0x9c, 0x80));
    }

    #[test]
    fn opposite_turn_does_not_snap_and_direction_priority_is_right_first() {
        let mut e = engine();
        e.ram[0x0b] = 1;
        e.ram[0xa0] = 0xa0;
        e.ram[0x90] = 93;
        e.ram[0x98] = 101;
        e.extras_player_input(0, 32);
        assert_eq!((e.ram[0xa0], e.ram[0x90], e.ram[0x98]), (0xa2, 93, 101));
        e.extras_player_input(0, 0xf0);
        assert_eq!((e.ram[0xa0], e.ram[0x90], e.ram[0x98]), (0xa3, 96, 104));
    }

    #[test]
    fn player_input_lockout_and_phase_skip_preserve_cartridge_status_bits() {
        let mut e = engine();
        e.ram[0xa0] = 0xa1;
        e.ram[0x6f] = 2;
        e.ram[0x0b] = 2;
        e.extras_player_input(0, 128);
        assert_eq!((e.ram[0xa0], e.ram[0x6f]), (0xa1, 2));
        e.ram[0x0b] = 3;
        e.extras_player_input(0, 128);
        assert_eq!((e.ram[0xa0], e.ram[0x6f], e.ram[0]), (0x89, 1, 0x80));
    }

    #[test]
    fn idle_ice_coasting_consumes_counter_then_stops_without_moving_itself() {
        let mut e = engine();
        e.ram[0xa0] = 0x88;
        e.ram[0x103] = 0x81;
        e.ram[0xb0] = 0;
        assert!(e.extras_player_idle_step(0));
        assert_eq!((e.ram[0x103], e.ram[0xb0], e.ram[0xa0]), (0x80, 4, 0x88));
        assert!(!e.extras_player_idle_step(0));
        assert_eq!(e.ram[0xa0], 0x84);
        assert!(!e.extras_player_idle_step(0));
        assert_eq!(e.ram[0xa0], 0xa0);
        e.ram[0xa0] = 0x89;
        e.ram[0x103] = 0x1c;
        assert!(!e.extras_player_idle_step(0));
        assert_eq!((e.ram[0x103], e.ram[0xa0]), (0x1c, 0x85));
    }

    #[test]
    fn red_drop_consumes_shared_rng_for_two_grid_coordinates_and_weighted_type() {
        for (seed, expected) in [
            (0, [48, 48, 0, 0, 3, 0]),
            (1, [192, 96, 3, 87, 3, 1]),
            (7, [96, 192, 1, 97, 3, 3]),
            (33, [192, 96, 3, 55, 3, 1]),
            (255, [96, 192, 1, 169, 3, 3]),
        ] {
            let mut e = Engine::new([0; 2048]);
            e.ram[0x0f] = seed;
            e.extras_drop_item();
            assert_eq!(
                [
                    e.ram[0x86],
                    e.ram[0x87],
                    e.ram[0x88],
                    e.ram[0x0f],
                    e.ram[0x10],
                    e.ram[0x00]
                ],
                expected
            );
            assert_eq!((e.ram[0x49], e.ram[0x62]), (255, 0));
        }
    }

    #[test]
    fn red_drop_rejects_player_overlap_without_awarding_a_pickup() {
        let mut e = Engine::new([0; 2048]);
        e.ram[0x90] = 48;
        e.ram[0x98] = 48;
        e.ram[0xa0] = 0xa0;
        e.extras_drop_item();
        assert_eq!(&e.ram[0x86..=0x88], &[48, 144, 4]);
        assert_eq!((e.ram[0x0f], e.ram[0x10], e.ram[0x00]), (206, 99, 2));
        assert_eq!((e.ram[0x49], e.ram[0x62], e.pickups), (255, 0, 0));
    }

    #[test]
    fn picked_item_disappears_only_after_visible_countdown() {
        let mut e = engine();
        e.ram[0x86] = 48;
        e.ram[0x87] = 96;
        e.ram[0x88] = 4;
        e.ram[0x62] = 2;
        e.extras_item_tick();
        assert_eq!((e.ram[0x62], e.ram[0x86]), (1, 48));
        e.extras_item_tick();
        assert_eq!(
            (e.ram[0x62], e.ram[0x86], e.ram[0x87], e.ram[0x88]),
            (0, 0, 96, 4)
        );
        e.ram[0x62] = 9;
        e.extras_item_tick();
        assert_eq!(e.ram[0x62], 9);
    }

    #[test]
    fn freeze_and_shield_expire_on_phase_zero_without_underflow() {
        let mut e = engine();
        e.ram[0x100] = 1;
        e.ram[0x89] = 2;
        e.ram[0x8a] = 0;
        e.ram[0x0b] = 63;
        e.extras_freeze_tick();
        e.extras_shield_tick();
        assert_eq!((e.ram[0x100], e.ram[0x89]), (1, 2));
        e.ram[0x0b] = 64;
        e.extras_freeze_tick();
        e.extras_shield_tick();
        assert_eq!((e.ram[0x100], e.ram[0x89], e.ram[0x8a]), (0, 1, 0));
    }

    #[test]
    fn shovel_last_three_ticks_alternate_and_rebuild_whole_fortress() {
        let mut e = engine();
        e.ram[0x45] = 4;
        e.ram[0x0b] = 64;
        e.ram[0x70c..0x712].fill(34);
        e.ram[0x7f4] = 255;
        e.extras_shovel_tick();
        assert_eq!(e.ram[0x45], 3);
        assert_eq!(&e.ram[0x70c..0x712], &[0, 0, 0, 0, 0, 0]);
        assert_eq!(&e.ram[0x72c..0x732], &[0, 15, 15, 15, 15, 0]);
        assert_eq!(&e.ram[0x74c..0x752], &[0, 15, 200, 202, 15, 0]);
        assert_eq!(&e.ram[0x76c..0x772], &[0, 15, 201, 203, 15, 0]);
        assert_eq!((e.ram[0x7f3], e.ram[0x7f4]), (0, 0xcc));
        e.ram[0x0b] = 80;
        e.extras_shovel_tick();
        assert_eq!((e.ram[0x45], e.ram[0x72d]), (3, 16));
        assert_eq!((e.ram[0x7f3], e.ram[0x7f4]), (63, 255));
        e.ram[0x0b] = 96;
        e.extras_shovel_tick();
        assert_eq!(e.ram[0x72d], 15);
        e.ram[0x45] = 1;
        e.ram[0x0b] = 128;
        e.extras_shovel_tick();
        assert_eq!((e.ram[0x45], e.ram[0x72d]), (0, 15));
    }

    #[test]
    fn shovel_does_not_repair_damage_before_blink_phase() {
        let mut e = engine();
        e.ram[0x45] = 5;
        e.ram[0x0b] = 64;
        e.ram[0x72d] = 3;
        e.extras_shovel_tick();
        assert_eq!((e.ram[0x45], e.ram[0x72d]), (4, 3));
        e.ram[0x45] = 1;
        e.ram[0x0b] = 65;
        e.extras_shovel_tick();
        assert_eq!((e.ram[0x45], e.ram[0x72d]), (1, 3));
    }

    #[test]
    fn score_bonus_is_once_at_twenty_thousand_and_requires_live_base() {
        let mut e = engine();
        e.ram[0x17] = 1;
        e.extras_score_life();
        assert_eq!((e.ram[0x51], e.ram[0x66]), (3, 0));
        e.ram[0x17] = 2;
        e.ram[0x68] = 0x7f;
        e.extras_score_life();
        assert_eq!((e.ram[0x51], e.ram[0x66]), (3, 0));
        e.ram[0x68] = 0x80;
        e.extras_score_life();
        assert_eq!((e.ram[0x51], e.ram[0x66]), (4, 1));
        e.extras_score_life();
        assert_eq!((e.ram[0x51], e.ram[0x66]), (4, 1));
    }

    #[test]
    fn bonus_processes_one_player_per_call_and_uses_byte_lives() {
        let mut e = engine();
        e.ram[0x83] = 1;
        e.ram[0x17] = 2;
        e.ram[0x1f] = 2;
        e.ram[0x51] = 255;
        e.extras_score_life();
        assert_eq!(
            (e.ram[0x51], e.ram[0x52], e.ram[0x66], e.ram[0x67]),
            (0, 0, 1, 0)
        );
        e.extras_score_life();
        assert_eq!((e.ram[0x52], e.ram[0x67]), (1, 1));
    }
}
