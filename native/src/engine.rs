//! Isolated mechanics feasibility experiment, not a replacement NES emulator.

#[path = "enemies.rs"]
mod enemies;

pub const RAM_LEN: usize = 2048;
pub const SUPPORTED: &str = "Verified no-spawn skill scenes after two neutral NES frames: \
    delayed controller input, player movement/turns, projectiles, residual bricks, \
    frozen targets, bullet cancellation, armor, pickups, timed fortress, ice coasting and death events. \
    Experimental cartridge-derived enemy state machine, RNG, fire and spawning.";
pub const LIMITATIONS: &str = "Not a NES emulator. Enemy/spawn flags remain unsupported: \
    the exact RNG reads transient zero-page audio/NMI scratch not reconstructed by this RAM-only mirror. \
    Long host-frame traces also exhibit preterminal 0/2 tick deltas; NMI/host-frame phase is not modeled. \
    Ice remains flagged pending long-trajectory validation despite passing bounded movement probes. \
    No menus, terminal-state continuation, stage transition, PPU/audio, or aligned random item drops. \
    Raw RAM outside modeled fields is retained, not emulated. \
    Call unsupported() to detect unsupported scene mechanics encountered.";
pub const UNSUPPORTED_ENEMIES: u32 = 1;
pub const UNSUPPORTED_SPAWNING: u32 = 2;
pub const UNSUPPORTED_TERRAIN: u32 = 4;
pub const UNSUPPORTED_ITEM_DROP: u32 = 8;
pub const UNSUPPORTED_ICE: u32 = 16;
pub const UNSUPPORTED_STAGE_END: u32 = 32;
pub const UNSUPPORTED_P2: u32 = 128;

const DELTA: [(i16, i16); 4] = [(0, -1), (-1, 0), (0, 1), (1, 0)];

pub struct Engine {
    pub ram: [u8; RAM_LEN],
    pub frame: u32,
    pub kills: u32,
    pub deaths: u32,
    pub shots: u32,
    pub pickups: u32,
    initial: [u8; RAM_LEN],
    pending_input: u8,
    last_input: u8,
    unsupported: u32,
    #[cfg(feature = "reference-rng")]
    reference_rng: Option<crate::random_replay::RandomReplay>,
}

impl Engine {
    pub fn new(ram: [u8; RAM_LEN]) -> Self {
        let mut engine = Self {
            ram,
            initial: ram,
            frame: 0,
            kills: 0,
            deaths: 0,
            shots: 0,
            pickups: 0,
            pending_input: 0,
            last_input: 0,
            unsupported: 0,
            #[cfg(feature = "reference-rng")]
            reference_rng: None,
        };
        engine.check_support();
        engine
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.initial);
    }

    pub fn unsupported(&self) -> u32 {
        self.unsupported
    }

    pub(crate) fn random_byte(&mut self) -> u8 {
        // $D44D mixes the incremented zero-page cursor, including ADC carry.
        let first = u16::from(self.ram[0x0f].wrapping_mul(7)) + u16::from(self.ram[0x0a]);
        self.ram[0x10] = self.ram[0x10].wrapping_add(1);
        let indexed = self.ram[usize::from(self.ram[0x10])];
        let result = (first as u8)
            .wrapping_add(indexed)
            .wrapping_add((first >> 8) as u8);
        #[cfg(feature = "reference-rng")]
        let result = if let Some(replay) = &mut self.reference_rng {
            replay.take(self.ram[0x10])
        } else {
            result
        };
        self.ram[0x0f] = result;
        result
    }

    #[cfg(feature = "reference-rng")]
    pub fn step_with_reference_rng(&mut self, action: u8, samples: &[(u8, u8)]) -> u32 {
        self.reference_rng = Some(crate::random_replay::RandomReplay::new(samples));
        self.step(action);
        self.reference_rng.take().expect("diagnostic replay initialized").status()
    }

    #[cfg(feature = "reference-rng")]
    pub(crate) fn reference_rng_failed(&self) -> bool {
        self.reference_rng.as_ref().is_some_and(|replay| replay.status() & 3 != 0)
    }

    /// Diagnostic C200 -> C206 update, conditioned on observed input/clock/RNG.
    /// No normal runtime caller uses this, and it does not synchronize game state.
    #[cfg(feature = "reference-rng")]
    pub fn logic_with_reference_rng(
        &mut self, input: u8, pressed: u8, tick: u8, coarse: u8, samples: &[(u8, u8)],
    ) -> u32 {
        self.reference_rng = Some(crate::random_replay::RandomReplay::new(samples));
        self.frame = self.frame.wrapping_add(1);
        self.ram[0x06] = input;
        self.ram[0x08] = pressed;
        self.ram[0x0b] = tick;
        self.ram[0x0a] = coarse;
        self.advance_gameplay(input, pressed);
        self.reference_rng.take().expect("diagnostic replay initialized").status()
    }

    #[cfg(feature = "reference-rng")]
    pub fn logic_with_native_rng(&mut self, input: u8, pressed: u8, tick: u8, coarse: u8) {
        self.reference_rng = None;
        self.frame = self.frame.wrapping_add(1);
        self.ram[0x06] = input;
        self.ram[0x08] = pressed;
        self.ram[0x0b] = tick;
        self.ram[0x0a] = coarse;
        self.advance_gameplay(input, pressed);
    }

    pub fn step(&mut self, action: u8) {
        let input = self.pending_input;
        self.pending_input = action;
        self.frame = self.frame.wrapping_add(1);
        self.ram[0x0b] = self.ram[0x0b].wrapping_add(1);
        if self.ram[0x0b] & 63 == 0 {
            self.ram[0x0a] = self.ram[0x0a].wrapping_add(1);
        }
        self.ram[0x06] = input;
        self.ram[0x08] = input & !self.last_input;
        if self.ram[0x30f] == 1 {
            self.ram[0x30f] = 2;
        }
        self.advance_gameplay(input, input & !self.last_input);
        self.last_input = input;
        self.ram[0x06] = action;
        self.ram[0x08] = action & !input;
        self.finish_gameplay_frame();
    }

    fn advance_gameplay(&mut self, input: u8, pressed: u8) {
        self.check_support();
        self.occupancy(true);
        self.extras_player_input(0, input);
        self.extras_freeze_tick();
        self.update_enemies();
        if self.animate_tanks() {
            self.tank_advance(0);
            self.ram[0xb0] ^= 4;
        }
        self.occupancy(false);
        self.advance_bullets();
        self.extras_shovel_tick();
        self.extras_shield_tick();
        if pressed & 1 != 0 {
            self.fire();
        }
        self.fire_enemies();
        self.spawn_enemy();
        self.projectile_terrain();
        self.cancel_bullets();
        self.projectile_tanks();
        self.pickup();
        self.extras_item_tick();
    }

    fn finish_gameplay_frame(&mut self) {
        let lost = self.ram[0x68] == 0
            || (self.ram[0x80] != 0 && self.ram[0x51].wrapping_add(self.ram[0x52]) == 0);
        if !lost && self.ram[0x80] != 0 {
            return;
        }
        // Terminal entry bookkeeping ($C728/$C223), not the following presentation loop.
        if lost {
            self.ram[0x105..0x109].copy_from_slice(&[0x70, 0xf0, 0, 0x11]);
        }
        self.ram[0x0b] = 0;
        self.ram[0x0a] = if self.ram[0x108] != 0 { 0xfe } else { 0 };
        self.ram[0x311] = 0;
        self.ram[0x312] = 0;
    }

    fn check_support(&mut self) {
        if self.ram[0x7f] != 0 {
            self.unsupported |= UNSUPPORTED_SPAWNING;
        }
        if active(self.ram[0xa1]) {
            self.unsupported |= UNSUPPORTED_P2;
        }
        if self.ram[0x80] == 0 || self.ram[0x68] != 0x80 || self.ram[0x51] == 0 {
            self.unsupported |= UNSUPPORTED_STAGE_END;
        }
        if self.ram[0x100] == 0 && self.ram[0xa2..0xa8].iter().any(|&s| active(s)) {
            self.unsupported |= UNSUPPORTED_ENEMIES;
        }
        for row in 2..28 {
            for col in 2..28 {
                let a = 0x400 + row * 32 + col;
                let value = clean(self.ram[a]);
                if value == 33 {
                    self.unsupported |= UNSUPPORTED_ICE;
                }
                if !matches!(value, 0..=16 | 18 | 32..=34 | 200..=207) {
                    self.unsupported |= UNSUPPORTED_TERRAIN;
                }
            }
        }
    }

    fn fire(&mut self) {
        if !active(self.ram[0xa0]) {
            return;
        }
        let power = (self.ram[0x101] >> 5) & 3;
        if self.ram[0xcc] != 0 {
            if power < 2 || self.ram[0xd4] != 0 {
                return;
            }
            for address in [0xb8, 0xc2, 0xcc, 0xd6] {
                self.ram[address + 8] = self.ram[address];
            }
        }
        let face = self.ram[0xa0] & 3;
        let (dx, dy) = DELTA[usize::from(face)];
        self.ram[0xb8] = (i16::from(self.ram[0x90]) + dx * 8) as u8;
        self.ram[0xc2] = (i16::from(self.ram[0x98]) + dy * 8) as u8;
        self.ram[0xcc] = 0x40 | face;
        self.ram[0xd6] = if power == 3 { 3 } else { u8::from(power > 0) };
        self.ram[0x30f] = 1;
        self.shots += 1;
    }

    fn advance_bullets(&mut self) {
        for i in 0..10 {
            let status = self.ram[0xcc + i];
            if flight(status) {
                let speed = if self.ram[0xd6 + i] & 1 != 0 { 4 } else { 2 };
                let (dx, dy) = DELTA[usize::from(status & 3)];
                self.ram[0xb8 + i] = (i16::from(self.ram[0xb8 + i]) + dx * speed) as u8;
                self.ram[0xc2 + i] = (i16::from(self.ram[0xc2 + i]) + dy * speed) as u8;
            } else if (0x10..0x40).contains(&status) {
                self.ram[0xcc + i] = animation_next(status, false);
            }
        }
    }

    fn projectile_terrain(&mut self) {
        for i in (0..10).rev() {
            if !flight(self.ram[0xcc + i]) {
                continue;
            }
            let flags = self.ram[0xd6 + i];
            if flags == 0 && (i as u8 ^ self.ram[0x0b]) & 1 == 0 {
                continue;
            }
            let (x, y) = (i16::from(self.ram[0xb8 + i]), i16::from(self.ram[0xc2 + i]));
            if !(16..224).contains(&x) || !(16..224).contains(&y) {
                self.ram[0xcc + i] = 0x33;
                continue;
            }
            let horizontal = self.ram[0xcc + i] & 1 != 0;
            let mut hit = false;
            for offset in [0, -1] {
                let (px, py) = if horizontal {
                    (x, y + offset)
                } else {
                    (x + offset, y)
                };
                let address = cell(px, py);
                let value = clean(self.ram[address]);
                if (1..=15).contains(&value) {
                    if value & quarter(px, py) == 0 {
                        continue;
                    }
                    hit = true;
                    if flags & 2 != 0 {
                        self.ram[address] = 0;
                    } else {
                        let mask = if horizontal {
                            if px & 4 == 0 { 5 } else { 10 }
                        } else if py & 4 == 0 {
                            3
                        } else {
                            12
                        };
                        self.ram[address] = value & !mask;
                    }
                } else if value == 16 || value == 17 {
                    hit = true;
                    if value == 16 && flags & 2 != 0 {
                        self.ram[address] = 0;
                    }
                } else if (200..=203).contains(&value) {
                    hit = true;
                    self.ram[0x68] = 0x73;
                    for (a, tile) in [(0x74e, 204), (0x74f, 206), (0x76e, 205), (0x76f, 207)] {
                        self.ram[a] = tile;
                    }
                }
            }
            if hit {
                self.ram[0xcc + i] = 0x33;
            }
        }
    }

    fn cancel_bullets(&mut self) {
        for player in [9, 8, 1, 0] {
            if !flight(self.ram[0xcc + player]) {
                continue;
            }
            for enemy in (0..10).rev() {
                if player & 7 != enemy & 7
                    && flight(self.ram[0xcc + enemy])
                    && self.ram[0xb8 + player].abs_diff(self.ram[0xb8 + enemy]) < 6
                    && self.ram[0xc2 + player].abs_diff(self.ram[0xc2 + enemy]) < 6
                {
                    self.ram[0xcc + player] = 0;
                    self.ram[0xcc + enemy] = 0;
                }
            }
        }
    }

    fn projectile_tanks(&mut self) {
        // The cartridge checks players first, then enemies in descending slot order.
        for tank in (0..2).rev() {
            if !active(self.ram[0xa0 + tank]) {
                continue;
            }
            for bullet in (2..8).rev() {
                if !self.projectile_overlaps_tank(bullet, tank) {
                    continue;
                }
                self.ram[0xcc + bullet] = 0x33;
                if self.ram[0x89 + tank] != 0 {
                    self.ram[0xcc + bullet] = 0;
                    continue;
                }
                self.kill(tank, true);
                break;
            }
        }
        for tank in (2..8).rev() {
            if !active(self.ram[0xa0 + tank]) {
                continue;
            }
            for bullet in [9, 8, 1, 0] {
                if !self.projectile_overlaps_tank(bullet, tank) {
                    continue;
                }
                self.ram[0xcc + bullet] = 0x33;
                let attribute = self.ram[0xa8 + tank];
                if attribute & 4 != 0 {
                    self.unsupported |= UNSUPPORTED_ITEM_DROP;
                    self.ram[0x5a] = tank as u8;
                    self.ram[0x5b] = bullet as u8;
                    self.extras_drop_item();
                    if attribute == 0xe4 {
                        self.ram[0xa8 + tank] = 0xe3;
                    }
                }
                if self.ram[0xa8 + tank] & 3 != 0 {
                    self.ram[0xa8 + tank] -= 1;
                } else {
                    self.kill(tank, true);
                    break;
                }
            }
        }
    }

    fn projectile_overlaps_tank(&self, bullet: usize, tank: usize) -> bool {
        flight(self.ram[0xcc + bullet])
            && self.ram[0xb8 + bullet].abs_diff(self.ram[0x90 + tank]) < 10
            && self.ram[0xc2 + bullet].abs_diff(self.ram[0x98 + tank]) < 10
    }

    pub(crate) fn kill(&mut self, tank: usize, shell: bool) {
        self.ram[0xa0 + tank] = 0x73;
        if tank == 0 {
            self.deaths += 1;
            self.ram[0x101] = 0;
            self.ram[0xa8] = 0;
        } else if tank >= 2 {
            self.kills += 1;
            if shell {
                let kind = usize::from((self.ram[0xa8 + tank] >> 5) & 3);
                self.ram[0x73 + kind] = self.ram[0x73 + kind].wrapping_add(1);
                self.add_score((kind as u32 + 1) * 100);
            }
        }
    }

    fn animate_tanks(&mut self) -> bool {
        let mut move_player = false;
        for i in (0..2).rev() {
            let status = self.ram[0xa0 + i];
            let eligible = self.ram[0x0b] % 4 != 2;
            if eligible {
                if status & 0xf0 == 0x80 {
                    let coast = self.extras_player_idle_step(i);
                    if i == 0 {
                        move_player = coast;
                    }
                } else if i == 0 && status & 0xf0 == 0xa0 {
                    move_player = true;
                }
            }
            if (0x10..0x80).contains(&status) && eligible {
                let next = animation_next(status, true);
                self.ram[0xa0 + i] = next;
                if next == 0 {
                    if i == 0 {
                        self.ram[0x51] = self.ram[0x51].saturating_sub(1);
                        if self.ram[0x51] != 0 {
                            self.ram[0xa0] = 0xf0;
                            self.ram[0x90] = 88;
                            self.ram[0x98] = 216;
                        }
                    }
                }
            } else if i == 0 && status >= 0xe0 && eligible {
                match status {
                    0xfd => self.ram[0xa0] = 0xe0,
                    0xed => {
                        self.ram[0xa0] = 0xa0;
                        self.ram[0x89] = 3;
                        self.ram[0xb0] = 0;
                    }
                    _ => self.ram[0xa0] = status.wrapping_add(1),
                }
            }
        }
        move_player
    }

    fn pickup(&mut self) {
        if !active(self.ram[0xa0])
            || self.ram[0x86] == 0
            || self.ram[0x62] != 0
            || self.ram[0x88] >= 6
            || self.ram[0x90].abs_diff(self.ram[0x86]) >= 12
            || self.ram[0x98].abs_diff(self.ram[0x87]) >= 12
        {
            return;
        }
        self.ram[0x49] = 0;
        self.ram[0x62] = 50;
        self.pickups += 1;
        self.add_score(500);
        match self.ram[0x88] {
            0 => self.ram[0x89] = 10,
            1 => self.ram[0x100] = 10,
            2 => {
                if self.ram[0x68] & 0x80 != 0 {
                    self.extras_fortress(true);
                    self.ram[0x45] = 20;
                }
            }
            3 => {
                let power = ((self.ram[0x101] & 0x60) + 0x20).min(0x60);
                self.ram[0x101] = power;
                self.ram[0xa8] = power;
            }
            4 => {
                for i in 2..8 {
                    if active(self.ram[0xa0 + i]) {
                        self.kill(i, false);
                        self.ram[0xa8 + i] = 0;
                    }
                }
            }
            5 => self.ram[0x51] = self.ram[0x51].wrapping_add(1),
            _ => unreachable!(),
        }
    }

    pub(crate) fn add_score(&mut self, points: u32) {
        let mut carry = points;
        for a in (0x15..=0x1b).rev() {
            carry += u32::from(self.ram[a]);
            self.ram[a] = (carry % 10) as u8;
            carry /= 10;
        }
        self.extras_score_life();
    }
}

fn active(status: u8) -> bool {
    (0x80..0xe0).contains(&status)
}
fn flight(status: u8) -> bool {
    status & 0xf0 == 0x40
}
fn clean(value: u8) -> u8 {
    let low = value & 0x7f;
    if matches!(low, 0..=16 | 18 | 32..=34) {
        low
    } else {
        value
    }
}
fn cell(x: i16, y: i16) -> usize {
    0x400 + (y as usize / 8) * 32 + x as usize / 8
}
fn quarter(x: i16, y: i16) -> u8 {
    1 << ((x & 4) / 4 + (y & 4) / 2)
}
fn animation_next(status: u8, tank: bool) -> u8 {
    if status & 15 > 1 {
        status - 1
    } else if status <= 0x11 {
        0
    } else if tank && status == 0x21 {
        0x16
    } else {
        status - 0x10 + 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene() -> [u8; RAM_LEN] {
        let mut r = [0; RAM_LEN];
        r[0x0b] = 32;
        r[0x51] = 3;
        r[0x68] = 0x80;
        r[0x80] = 1;
        r[0x85] = 1;
        r[0x90] = 88;
        r[0x98] = 160;
        r[0xa0] = 0x88;
        r
    }

    #[test]
    fn cartridge_rng_reads_incremented_zero_page_cursor_and_adc_carry() {
        for (value, tick, cursor, index_value, expected) in [
            (1, 2, 31, 4, 13),
            (255, 10, 143, 88, 92),
            (3, 1, 255, 16, 38),
        ] {
            let mut e = Engine::new(scene());
            e.ram[0x0f] = value;
            e.ram[0x0a] = tick;
            e.ram[0x10] = cursor;
            e.ram[usize::from(cursor.wrapping_add(1))] = index_value;
            assert_eq!(e.random_byte(), expected);
            assert_eq!(e.ram[0x0f], expected);
            assert_eq!(e.ram[0x10], cursor.wrapping_add(1));
        }
    }

    #[test]
    fn cartridge_rng_cursor_can_read_its_own_state() {
        let mut e = Engine::new(scene());
        e.ram[0x0f] = 9;
        e.ram[0x10] = 14;
        assert_eq!(e.random_byte(), 72);
        e.ram[0x0f] = 9;
        assert_eq!(e.random_byte(), 79);
    }

    fn enemy_scene() -> [u8; RAM_LEN] {
        let mut r = scene();
        r[0x92] = 88;
        r[0x9a] = 80;
        r[0xa2] = 0xa2;
        r[0xaa] = 0x80;
        r[0x0f] = 3;
        r[0x10] = 0xaf;
        r[0xb1] = 13;
        r[0x84] = 186;
        r[0x6c] = 5;
        r
    }

    #[test]
    fn live_enemy_moves_then_fires_with_two_rng_calls() {
        let mut e = Engine::new(enemy_scene());
        e.step(0);
        assert_eq!((e.ram[0xf], e.ram[0x10]), (160, 177));
        assert_eq!(
            (e.ram[0x92], e.ram[0x9a], e.ram[0xa2], e.ram[0xb2]),
            (88, 81, 162, 4)
        );
        assert_eq!(
            (e.ram[0xba], e.ram[0xc4], e.ram[0xce], e.ram[0xd8]),
            (88, 89, 66, 0)
        );
        assert_eq!(
            e.shots, 0,
            "Only player shots count toward the episode counter"
        );
    }

    #[test]
    fn aligned_enemy_random_turn_replaces_movement_but_does_not_suppress_fire() {
        let mut r = enemy_scene();
        r[0xf] = 0;
        r[0xb1] = 0;
        let mut e = Engine::new(r);
        e.step(0);
        assert_eq!((e.ram[0xf], e.ram[0x10]), (0, 178));
        assert_eq!((e.ram[0x9a], e.ram[0xa2], e.ram[0xb2]), (80, 160, 0));
        assert_eq!((e.ram[0xc4], e.ram[0xce]), (72, 64));
    }

    #[test]
    fn unaligned_enemy_skips_grid_turn_rng_but_still_fires() {
        let mut r = enemy_scene();
        r[0xf] = 0;
        r[0xb1] = 0;
        r[0x9a] = 81;
        let mut e = Engine::new(r);
        e.step(0);
        assert_eq!((e.ram[0xf], e.ram[0x10]), (0, 176));
        assert_eq!((e.ram[0x9a], e.ram[0xc4], e.ram[0xce]), (82, 90, 66));
    }

    #[test]
    fn spawner_uses_highest_free_slot_timer_and_round_robin_lane() {
        let mut r = enemy_scene();
        r[0xa2] = 0;
        r[0x7f] = 18;
        r[0x6a] = 0;
        let mut e = Engine::new(r);
        e.step(0);
        assert_eq!(
            (e.ram[0x95], e.ram[0x9d], e.ram[0xa5], e.ram[0xad]),
            (120, 24, 240, 0)
        );
        assert_eq!((e.ram[0x7f], e.ram[0x82], e.ram[0x6a]), (17, 186, 1));
        assert_eq!((e.ram[0xf], e.ram[0x10]), (3, 175));
    }

    #[test]
    fn enemy_activation_consumes_stage_inventory_and_can_fire_immediately() {
        let mut r = enemy_scene();
        r[0xa2] = 0xed;
        r[0xaa] = 4;
        r[0x8b] = 1;
        r[0xf] = 0;
        r[0xb1] = 0;
        let mut e = Engine::new(r);
        e.step(0);
        assert_eq!((e.ram[0xa2], e.ram[0xaa], e.ram[0x8b]), (162, 132, 0));
        assert_eq!((e.ram[0xc4], e.ram[0xce]), (88, 66));
        assert_eq!((e.ram[0xf], e.ram[0x10]), (0, 176));
    }

    #[test]
    fn enemy_bump_and_aim_states_follow_cartridge_rng_consumption() {
        for (status, wall, b1, want_rng, want_status, want_bx, want_by, want_bs) in [
            (0xa2, true, 13, 96, 0x90, 88, 72, 0x40),
            (0xa2, true, 0, 9, 0x8a, 0, 0, 0),
            (0x92, false, 13, 96, 0xa1, 80, 80, 0x41),
            (0xb2, false, 13, 160, 0xa3, 96, 80, 0x43),
        ] {
            let mut r = enemy_scene();
            r[0xa2] = status;
            r[0xb1] = b1;
            if wall {
                r[0x56a] = 15;
            }
            let mut e = Engine::new(r);
            e.step(0);
            assert_eq!((e.ram[0xf], e.ram[0xa2]), (want_rng, want_status));
            assert_eq!(
                (e.ram[0xba], e.ram[0xc4], e.ram[0xce]),
                (want_bx, want_by, want_bs)
            );
        }
    }

    #[test]
    fn fast_enemies_move_every_tick_and_busy_shots_still_consume_rng() {
        let mut r = enemy_scene();
        r[0x0b] = 33;
        let mut slow = Engine::new(r);
        slow.step(0);
        assert_eq!((slow.ram[0x9a], slow.ram[0xf]), (80, 21));
        r[0xaa] = 0xa0;
        let mut fast = Engine::new(r);
        fast.step(0);
        assert_eq!((fast.ram[0x9a], fast.ram[0xf]), (81, 160));
        let mut r = enemy_scene();
        r[0xba] = 160;
        r[0xc4] = 160;
        r[0xce] = 0x42;
        let mut busy = Engine::new(r);
        busy.step(0);
        assert_eq!((busy.ram[0xf], busy.ram[0xc4]), (160, 162));
    }

    #[test]
    fn odd_slot_terrain_checks_even_tick_and_steel_flag_bypasses_cadence() {
        for (slot, flags) in [(5, 0), (2, 2)] {
            let mut r = scene();
            r[0x0b] = 33;
            r[0x5cc] = 16;
            r[0x5ec] = 16;
            r[0xb8 + slot] = 98;
            r[0xc2 + slot] = 120;
            r[0xcc + slot] = 0x43;
            r[0xd6 + slot] = flags;
            let mut e = Engine::new(r);
            e.step(0);
            assert_eq!(e.ram[0xcc + slot], 0x33);
        }
    }

    #[test]
    fn shield_hit_removes_enemy_projectile_without_explosion_occupancy() {
        let mut r = scene();
        r[0x89] = 3;
        r[0xba] = 88;
        r[0xc4] = 150;
        r[0xce] = 0x42;
        let mut e = Engine::new(r);
        e.step(0);
        assert_eq!((e.ram[0xce], e.deaths), (0, 0));
    }

    #[test]
    fn one_frame_input_lag_and_three_of_four_movement() {
        let mut e = Engine::new(scene());
        for want in [88, 88, 89, 90, 91, 91, 92, 93] {
            e.step(128);
            assert_eq!(e.ram[0x90], want);
        }
        assert_eq!((e.frame, e.ram[0x0b], e.ram[0xa0] & 3), (8, 40, 3));
        e.step(0);
        assert_eq!(e.ram[0x90], 94);
        e.step(0);
        e.step(0);
        assert_eq!(e.ram[0x90], 94);
    }

    #[test]
    fn projectile_hits_highest_enemy_slot_when_impact_boxes_overlap() {
        let mut r = scene();
        r[0xa2] = 0xa2;
        r[0xa4] = 0xa0;
        r[0xaa] = 0x80;
        r[0xac] = 0x80;
        r[0x92] = 88;
        r[0x94] = 88;
        r[0x9a] = 100;
        r[0x9c] = 116;
        r[0xcc] = 0x40;
        r[0xb8] = 88;
        r[0xc2] = 108;
        let mut e = Engine::new(r);
        e.projectile_tanks();
        assert_eq!(e.ram[0xa4], 0x73);
        assert!(active(e.ram[0xa2]));
        assert_eq!(e.kills, 1);
    }

    #[test]
    fn terrain_probe_miss_does_not_clear_neighbor_cell_after_other_probe_hits() {
        let mut r = scene();
        r[0x0b] = 182;
        r[0xcf] = 0x43; // Enemy slot 3; horizontal projectile probes y=160 and y=159.
        r[0xbb] = 110;
        r[0xc5] = 160;
        r[cell(110, 160)] = 12; // Bottom quarters: y=160 probe misses.
        r[cell(110, 159)] = 10; // Right quarters: y=159 probe hits.
        let mut e = Engine::new(r);
        e.projectile_terrain();
        assert_eq!(e.ram[cell(110, 160)], 12);
        assert_eq!(e.ram[cell(110, 159)], 0);
        assert_eq!(e.ram[0xcf], 0x33);
    }

    #[test]
    fn input_registers_latch_current_buttons_while_gameplay_uses_previous_frame() {
        let mut e = Engine::new(scene());
        e.step(1);
        assert_eq!((e.ram[6], e.ram[8], e.shots), (1, 1, 0));
        e.step(1);
        assert_eq!((e.ram[6], e.ram[8], e.shots), (1, 0, 1));
    }

    fn ice_scene(blocked: bool) -> [u8; RAM_LEN] {
        let mut r = scene();
        r[0xa0] = 0xa3;
        r[0x103] = 2;
        for row in 2..28 {
            for col in 2..28 {
                r[0x400 + row * 32 + col] = 33;
            }
        }
        if blocked {
            r[0x90] = 96;
            r[0x66d] = 16;
            r[0x68d] = 16;
        }
        r
    }

    #[test]
    fn ice_coast_moves_through_collision_path_and_cancels_track_animation_xors() {
        for (blocked, x_positions) in [(false, [89, 89, 90, 90]), (true, [96, 96, 96, 96])] {
            let mut e = Engine::new(ice_scene(blocked));
            assert_ne!(e.unsupported() & UNSUPPORTED_ICE, 0);
            for (x, inertia, status) in [
                (x_positions[0], 129, 139),
                (x_positions[1], 129, 139),
                (x_positions[2], 128, 139),
                (x_positions[3], 128, 135),
            ] {
                e.step(0);
                assert_eq!(
                    (e.ram[0x90], e.ram[0x103], e.ram[0xa0], e.ram[0xb0]),
                    (x, inertia, status, 0)
                );
            }
        }
    }

    #[test]
    fn ice_sample_precedes_occupancy_and_release_coasts_twenty_eight_movement_ticks() {
        let mut r = ice_scene(false);
        r[0x103] = 0;
        let mut e = Engine::new(r);
        for f in 0..40 {
            e.step(if f < 2 { 128 } else { 0 });
            if f == 2 {
                assert_eq!((e.ram[0x90], e.ram[0x103], e.ram[0xb0]), (89, 156, 4));
            }
        }
        assert_eq!((e.ram[0x90], e.ram[0x103], e.ram[0xb0]), (117, 128, 4));
        e.step(0);
        assert_eq!((e.ram[0x90], e.ram[0xa0]), (117, 135));
    }

    #[test]
    fn turn_snaps_half_up_and_moves_after_snap() {
        for (x, want) in [(89, 88), (91, 88), (92, 96), (95, 96)] {
            let mut r = scene();
            r[0x90] = x;
            r[0xa0] = 0x83;
            let mut e = Engine::new(r);
            for _ in 0..3 {
                e.step(16);
            }
            assert_eq!((e.ram[0x90], e.ram[0x98]), (want, 159));
        }
    }

    #[test]
    fn skipped_turn_still_fires_old_facing() {
        let mut e = Engine::new(scene());
        e.step(129);
        e.step(129);
        assert_eq!((e.ram[0xb8], e.ram[0xc2], e.ram[0xcc]), (88, 152, 0x40));
        assert_eq!(e.shots, 1);
        e.step(128);
        assert_eq!((e.ram[0x90], e.ram[0xc2]), (89, 150));
    }

    #[test]
    fn two_star_transfer_preserves_trajectory() {
        let mut r = scene();
        r[0x101] = 0x40;
        r[0xa8] = 0x40;
        let mut e = Engine::new(r);
        for a in [129, 129, 128, 128, 129, 129] {
            e.step(a);
        }
        assert_eq!(e.shots, 2);
        assert_eq!((e.ram[0xb8], e.ram[0xc2], e.ram[0xcc]), (99, 160, 0x43));
        assert_eq!(
            (e.ram[0xc0], e.ram[0xca], e.ram[0xd4], e.ram[0xde]),
            (88, 136, 0x40, 1)
        );
    }

    #[test]
    fn stationary_shoot_skill_kill_occurs_on_frame_34() {
        let mut r = scene();
        r[0x92] = 88;
        r[0x9a] = 80;
        r[0xa2] = 0xa2;
        r[0xaa] = 0x80;
        r[0x100] = 10;
        let mut e = Engine::new(r);
        for f in 0..34 {
            e.step(if f % 4 < 2 { 1 } else { 0 });
            assert_eq!(e.kills, u32::from(f == 33));
        }
        assert_eq!((e.ram[0xa2], e.ram[0xc2], e.ram[0xcc]), (0x73, 88, 0x33));
        assert_eq!((e.ram[0x73], e.ram[0x19], e.ram[0x80]), (1, 1, 1));
    }

    #[test]
    fn counter_skill_cancels_both_projectiles_on_frame_17() {
        let mut r = scene();
        r[0x98] = 144;
        r[0xba] = 88;
        r[0xc4] = 68;
        r[0xce] = 0x42;
        let mut e = Engine::new(r);
        for f in 0..17 {
            e.step(if f % 4 < 2 { 1 } else { 0 });
        }
        assert_eq!((e.ram[0xcc], e.ram[0xce]), (0, 0));
        assert_eq!((e.ram[0xc2], e.ram[0xc4]), (106, 102));
        assert_eq!((e.kills, e.deaths, e.shots), (0, 0, 1));
    }

    #[test]
    fn bullet_cancellation_includes_five_pixel_offset_but_not_six() {
        for (offset, statuses) in [(4, (0, 0)), (5, (0, 0)), (6, (0x42, 0x40))] {
            let mut r = scene();
            r[0xb8] = 88;
            r[0xc2] = 102;
            r[0xcc] = 0x42;
            r[0xba] = 88 + offset;
            r[0xc4] = 106;
            r[0xce] = 0x40;
            let mut e = Engine::new(r);
            e.step(0);
            assert_eq!((e.ram[0xcc], e.ram[0xce]), statuses);
        }
    }

    #[test]
    fn secondary_slot_cancels_all_overlapping_opponents_before_primary_slot() {
        let mut r = scene();
        for (slot, y, facing) in [(0, 102, 2), (8, 102, 2), (2, 106, 0), (3, 106, 0)] {
            r[0xb8 + slot] = 88;
            r[0xc2 + slot] = y;
            r[0xcc + slot] = 0x40 | facing;
        }
        let mut e = Engine::new(r);
        e.step(0);
        assert_eq!((e.ram[0xcc], e.ram[0xd4], e.ram[0xce], e.ram[0xcf]), (0x42, 0, 0, 0));
    }

    #[test]
    fn demolition_quarters_clear_on_odd_tick_not_previous_even_tick() {
        let mut r = scene();
        r[0x98] = 120;
        r[0xa0] = 0x83;
        r[0x5cc] = 8;
        r[0x5ec] = 2;
        let mut e = Engine::new(r);
        for a in [1, 1, 0, 0] {
            e.step(a);
        }
        assert_eq!((e.ram[0x5cc], e.ram[0x5ec], e.ram[0xb8]), (8, 2, 100));
        e.step(1);
        assert_eq!(
            (e.ram[0x5cc], e.ram[0x5ec], e.ram[0xb8], e.ram[0xcc]),
            (0, 0, 102, 0x33)
        );
    }

    #[test]
    fn any_brick_quarter_blocks_tank_but_missed_quarter_survives_shots() {
        let mut r = scene();
        r[0x98] = 120;
        r[0xa0] = 0x83;
        r[0x5cc] = 0x81;
        let mut e = Engine::new(r);
        for f in 0..48 {
            e.step(if f % 4 < 2 { 129 } else { 128 });
        }
        assert!(e.shots > 0);
        assert_eq!((e.ram[0x90], e.ram[0x98]), (88, 120));
        assert_eq!(e.ram[0x5cc] & 0x7f, 1);
    }

    #[test]
    fn bullet_impact_animation_keeps_slot_busy_for_nine_frames() {
        let mut r = scene();
        r[0xb8] = 226;
        r[0xc2] = 120;
        r[0xcc] = 0x33;
        let mut e = Engine::new(r);
        for want in [0x32, 0x31, 0x23, 0x22, 0x21, 0x13, 0x12, 0x11, 0] {
            e.step(0);
            assert_eq!(e.ram[0xcc], want);
            assert_eq!(e.ram[0xb8], 226);
        }
    }

    #[test]
    fn enemy_bullet_death_is_immediate_but_life_loss_is_deferred() {
        let mut r = scene();
        r[0x98] = 144;
        r[0xba] = 96;
        r[0xc4] = 68;
        r[0xce] = 0x42;
        let mut e = Engine::new(r);
        for _ in 0..33 {
            e.step(0);
        }
        assert_eq!(e.deaths, 0);
        e.step(0);
        assert_eq!((e.deaths, e.ram[0xa0], e.ram[0x51]), (1, 0x73, 3));
        for _ in 0..30 {
            e.step(0);
        }
        assert_eq!(e.ram[0x51], 3);
        e.step(0);
        assert_eq!((e.ram[0x51], e.ram[0xa0]), (2, 0xf0));
    }

    #[test]
    fn final_life_animation_resets_ram_clock_but_not_host_frame_counter() {
        let mut r = scene();
        r[0x0a] = 17;
        r[0x0b] = 182;
        r[0x51] = 1;
        r[0xa0] = 0x11;
        let mut e = Engine::new(r);
        e.step(0);
        assert_eq!((e.ram[0x51], e.ram[0xa0], e.ram[0x0a], e.ram[0x0b]), (0, 0, 254, 0));
        assert_eq!(&e.ram[0x105..0x109], &[112, 240, 0, 17]);
        assert_eq!(e.frame, 1);
    }

    #[test]
    fn nonfinal_life_does_not_reset_ram_clock_and_clear_uses_zero_coarse_clock() {
        let mut r = scene();
        r[0x0a] = 17;
        r[0x0b] = 182;
        r[0x51] = 2;
        r[0xa0] = 0x11;
        let mut e = Engine::new(r);
        e.step(0);
        assert_eq!((e.ram[0x51], e.ram[0xa0], e.ram[0x0a], e.ram[0x0b]), (1, 240, 17, 183));
        let mut r = scene();
        r[0x0a] = 4;
        r[0xa2] = 0x11;
        r[0xaa] = 0x80;
        let mut e = Engine::new(r);
        e.step(0);
        assert_eq!((e.ram[0x80], e.ram[0x0a], e.ram[0x0b], e.ram[0x108]), (0, 0, 0, 0));
    }

    #[test]
    fn six_pickups_have_distinct_effects_and_score_without_kills() {
        for kind in 0..6 {
            let mut r = scene();
            r[0x86] = 99;
            r[0x87] = 160;
            r[0x88] = kind;
            let mut e = Engine::new(r);
            e.step(0);
            assert_eq!(
                (e.pickups, e.kills, e.ram[0x19], e.ram[0x62]),
                (1, 0, 5, 49)
            );
            match kind {
                0 => assert_eq!(e.ram[0x89], 10),
                1 => assert_eq!(e.ram[0x100], 10),
                2 => assert_eq!(e.ram[0x400 + 25 * 32 + 13], 16),
                3 => assert_eq!((e.ram[0x101], e.ram[0xa8]), (32, 32)),
                4 => assert_eq!(e.kills, 0),
                5 => assert_eq!(e.ram[0x51], 4),
                _ => unreachable!(),
            }
            e.step(0);
            assert_eq!(e.pickups, 1);
        }
    }

    #[test]
    fn pickup_timers_are_not_decremented_again_on_the_collection_boundary() {
        for (kind, timer, expected) in [(0, 0x89, 10), (1, 0x100, 10), (2, 0x45, 20)] {
            let mut r = scene();
            r[0x0b] = 63;
            r[0x86] = 88;
            r[0x87] = 160;
            r[0x88] = kind;
            let mut e = Engine::new(r);
            e.step(0);
            assert_eq!((e.ram[timer], e.ram[0x62]), (expected, 49));
        }
    }

    #[test]
    fn shield_expires_before_impact_and_empty_item_slot_does_not_tick() {
        let mut r = scene();
        r[0x0b] = 63;
        r[0x89] = 1;
        r[0x62] = 1;
        r[0xba] = 88;
        r[0xc4] = 150;
        r[0xce] = 0x42;
        let mut e = Engine::new(r);
        e.step(0);
        assert_eq!((e.deaths, e.ram[0x89], e.ram[0x62]), (1, 0, 1));
    }

    #[test]
    fn pickup_score_threshold_awards_one_life_and_shovel_replaces_perimeter() {
        let mut r = scene();
        r[0x17] = 1;
        r[0x18] = 9;
        r[0x19] = 9;
        r[0x86] = 88;
        r[0x87] = 160;
        r[0x88] = 2;
        r[0x70c] = 15;
        let mut e = Engine::new(r);
        e.step(0);
        assert_eq!((e.ram[0x51], e.ram[0x66]), (4, 1));
        assert_eq!((e.ram[0x70c], e.ram[0x72d]), (0, 16));
        e.add_score(500);
        assert_eq!(e.ram[0x51], 4);
    }

    #[test]
    fn grenade_clears_enemy_attributes_and_life_pickup_wraps() {
        let mut r = enemy_scene();
        r[0x100] = 10;
        r[0x86] = 88;
        r[0x87] = 160;
        r[0x88] = 4;
        let mut e = Engine::new(r);
        e.step(0);
        assert_eq!((e.kills, e.ram[0xaa]), (1, 0));
        let mut r = scene();
        r[0x51] = 255;
        r[0x86] = 88;
        r[0x87] = 160;
        r[0x88] = 5;
        let mut e = Engine::new(r);
        e.step(0);
        assert_eq!(e.ram[0x51], 0);
    }

    #[test]
    fn red_drop_precedes_armor_and_shell_counter_and_preserves_basic_carrier_bit() {
        for (attribute, expected_attribute, kills) in [(0x84, 0x84, 1), (0xe4, 0xe2, 0)] {
            let mut r = enemy_scene();
            r[0x100] = 10;
            r[0xf] = 0;
            r[0x10] = 0x70;
            r[0xaa] = attribute;
            r[0xb8] = 88;
            r[0xc2] = 90;
            r[0xcc] = 0x40;
            let mut e = Engine::new(r);
            e.step(0);
            assert_eq!((e.ram[0x86], e.ram[0x87], e.ram[0x88]), (48, 48, 0));
            assert_eq!((e.ram[0xf], e.ram[0x10]), (0, 0x73));
            assert_eq!(
                (e.ram[0xaa], e.kills, e.pickups),
                (expected_attribute, kills, 0)
            );
            assert_eq!(e.ram[0x73], kills as u8);
        }
    }

    #[test]
    fn red_drop_rng_observes_caller_tank_and_bullet_slot_registers() {
        for (bullet, expected_rng) in [(0, 98), (8, 154)] {
            let mut r = enemy_scene();
            r[0x100] = 10;
            r[0xf] = 0;
            r[0x10] = 0x59;
            r[0xaa] = 0x84;
            r[0xb8 + bullet] = 88;
            r[0xc2 + bullet] = 90;
            r[0xcc + bullet] = 0x40;
            let mut e = Engine::new(r);
            e.step(0);
            assert_eq!((e.ram[0x86], e.ram[0x87], e.ram[0x88]), (144, 144, 2));
            assert_eq!((e.ram[0xf], e.ram[0x10]), (expected_rng, 0x5c));
        }
    }

    #[test]
    fn respawn_animation_uses_movement_ticks_and_cannot_turn_on_activation() {
        let mut r = scene();
        r[0xa0] = 0xf0;
        r[0x0b] = 72;
        r[0x98] = 216;
        r[0xb0] = 4;
        let mut e = Engine::new(r);
        for _ in 0..36 {
            e.step(65);
        }
        assert_eq!(e.ram[0xa0], 0xed);
        e.step(0);
        assert_eq!(
            (e.ram[0xa0] & 3, e.ram[0x90], e.ram[0x98], e.ram[0x89]),
            (0, 88, 216, 3)
        );
        assert_eq!(e.ram[0xa0], 0xa0);
        assert_eq!(e.ram[0xb0], 0);
    }

    #[test]
    fn base_destruction_rewrites_tiles_in_cartridge_order() {
        let mut r = scene();
        for (a, value) in [(0x74e, 200), (0x74f, 201), (0x76e, 202), (0x76f, 203)] {
            r[a] = value;
        }
        r[0xb8] = 120;
        r[0xc2] = 206;
        r[0xcc] = 0x42;
        let mut e = Engine::new(r);
        e.step(0);
        assert_ne!(e.ram[0x68], 0x80);
        assert_eq!(
            (e.ram[0x74e], e.ram[0x74f], e.ram[0x76e], e.ram[0x76f]),
            (204, 206, 205, 207)
        );
    }

    #[test]
    fn brick_blast_clears_perpendicular_half_cells_without_touching_other_rows() {
        for (y, expected) in [
            (112, [5, 5, 15, 15]),
            (114, [15, 5, 15, 15]),
            (116, [15, 5, 15, 15]),
            (118, [15, 5, 15, 15]),
            (120, [15, 5, 5, 15]),
            (122, [15, 15, 5, 15]),
        ] {
            let mut r = scene();
            for row in 13..17 {
                r[0x400 + row * 32 + 12] = 15;
            }
            r[0xb8] = 98;
            r[0xc2] = y;
            r[0xcc] = 0x43;
            let mut e = Engine::new(r);
            e.step(0);
            for (row, want) in (13..17).zip(expected) {
                assert_eq!(e.ram[0x400 + row * 32 + 12], want, "y={y}, row={row}");
            }
        }
    }

    #[test]
    fn steel_requires_break_flag_independently_of_speed() {
        for flags in 0..4 {
            let mut r = scene();
            r[0x5cc] = 16;
            r[0x5ec] = 16;
            r[0xb8] = 98;
            r[0xc2] = 120;
            r[0xcc] = 0x43;
            r[0xd6] = flags;
            let mut e = Engine::new(r);
            e.step(0);
            assert_eq!(e.ram[0xcc], 0x33);
            let expected = if flags & 2 != 0 { 0 } else { 16 };
            assert_eq!((e.ram[0x5cc], e.ram[0x5ec]), (expected, expected));
        }
    }

    #[test]
    fn armor_hit_is_not_kill_and_grenade_is_not_shell_score() {
        let mut r = scene();
        r[0x100] = 10;
        r[0x92] = 88;
        r[0x9a] = 80;
        r[0xa2] = 0xa2;
        r[0xaa] = 0xe3;
        r[0xb8] = 88;
        r[0xc2] = 90;
        r[0xcc] = 0x40;
        let mut e = Engine::new(r);
        e.step(0);
        assert_eq!((e.ram[0xaa], e.kills, e.ram[0x19]), (0xe2, 0, 0));
        e.ram[0x86] = 88;
        e.ram[0x87] = 160;
        e.ram[0x88] = 4;
        e.step(0);
        assert_eq!((e.kills, e.pickups, e.ram[0x76], e.ram[0x19]), (1, 1, 0, 5));
        assert_eq!((e.ram[0xa2], e.ram[0x80]), (0x73, 1));
    }

    #[test]
    fn full_stage_paths_remain_flagged_until_differentially_verified() {
        let mut r = scene();
        r[0x7f] = 1;
        r[0xa2] = 0xa2;
        r[0x92] = 120;
        r[0x9a] = 80;
        r[0x442] = 255;
        r[0x68] = 0x73;
        let e = Engine::new(r);
        assert_eq!(
            e.unsupported()
                & (UNSUPPORTED_ENEMIES
                    | UNSUPPORTED_SPAWNING
                    | UNSUPPORTED_TERRAIN
                    | UNSUPPORTED_STAGE_END),
            UNSUPPORTED_ENEMIES
                | UNSUPPORTED_SPAWNING
                | UNSUPPORTED_TERRAIN
                | UNSUPPORTED_STAGE_END
        );
    }

    #[test]
    fn reset_restores_original_ram_counters_and_input_pipeline() {
        let mut e = Engine::new(scene());
        e.step(129);
        e.step(129);
        assert_eq!(e.shots, 1);
        e.reset();
        assert_eq!(e.ram, scene());
        assert_eq!(
            (e.frame, e.kills, e.deaths, e.shots, e.pickups),
            (0, 0, 0, 0, 0)
        );
        e.step(128);
        assert_eq!(e.ram[0x90], 88);
    }

    #[test]
    #[ignore = "invoked with binary fixtures by scratchengineproof.py"]
    fn rom_trace_probe() {
        let input = std::fs::read(std::env::var("ENGINE_PROOF_INPUT").unwrap()).unwrap();
        let mut engine = Engine::new(input[..RAM_LEN].try_into().unwrap());
        let mut output = Vec::new();
        for &action in &input[RAM_LEN..] {
            engine.step(action);
            output.extend_from_slice(&engine.ram);
            for count in [
                engine.frame,
                engine.kills,
                engine.deaths,
                engine.shots,
                engine.pickups,
            ] {
                output.extend_from_slice(&count.to_le_bytes());
            }
        }
        std::fs::write(std::env::var("ENGINE_PROOF_OUTPUT").unwrap(), output).unwrap();
    }
}
