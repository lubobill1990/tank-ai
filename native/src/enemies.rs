//! Enemy state transitions independently derived from local cartridge probes.
//! Small behavior tables below describe direction preference/category ordering;
//! they contain no executable cartridge bytes.
use super::{DELTA, Engine, UNSUPPORTED_SPAWNING, active, animation_next, cell};

// Category order observed in the stage inventory selector at $E3CB/$E4EC.
const CATEGORIES: [[u8; 4]; 35] = [
    [0x80, 0xa0, 0xc0, 0xe0],
    [0xe0, 0xa0, 0xc0, 0x80],
    [0x80, 0xa0, 0xc0, 0xe0],
    [0xc0, 0xa0, 0x80, 0xe0],
    [0xc0, 0xe0, 0x80, 0xa0],
    [0xc0, 0xa0, 0x80, 0xe0],
    [0x80, 0xa0, 0xc0, 0x80],
    [0xc0, 0xe0, 0xa0, 0x80],
    [0x80, 0xa0, 0xc0, 0xe0],
    [0x80, 0xa0, 0xc0, 0xe0],
    [0xa0, 0xe0, 0xc0, 0xa0],
    [0xc0, 0xa0, 0x80, 0xe0],
    [0xc0, 0xa0, 0x80, 0xe0],
    [0xc0, 0xa0, 0x80, 0xe0],
    [0x80, 0xc0, 0xa0, 0xe0],
    [0x80, 0xc0, 0xa0, 0xe0],
    [0xe0, 0xa0, 0xc0, 0x80],
    [0xe0, 0x80, 0xc0, 0xa0],
    [0xa0, 0xe0, 0x80, 0xc0],
    [0xa0, 0x80, 0xc0, 0xe0],
    [0xc0, 0xa0, 0x80, 0xe0],
    [0xa0, 0x80, 0xc0, 0xe0],
    [0xe0, 0x80, 0xc0, 0xa0],
    [0xc0, 0xe0, 0xa0, 0x80],
    [0xc0, 0xa0, 0x80, 0xe0],
    [0xa0, 0xe0, 0x80, 0xc0],
    [0xc0, 0xe0, 0xa0, 0x80],
    [0xa0, 0xe0, 0x80, 0xc0],
    [0xc0, 0xa0, 0x80, 0xe0],
    [0x80, 0xa0, 0xc0, 0xe0],
    [0xc0, 0xa0, 0xe0, 0xc0],
    [0xe0, 0x80, 0xc0, 0xa0],
    [0xa0, 0xe0, 0xc0, 0xa0],
    [0xc0, 0xa0, 0x80, 0xe0],
    [0xc0, 0xa0, 0x80, 0xe0],
];

impl Engine {
    pub(super) fn occupancy(&mut self, occupied: bool) {
        for i in (0..8).rev() {
            self.ram[0x5a] = i as u8;
            if !active(self.ram[0xa0 + i]) {
                continue;
            }
            let pointer = if occupied {
                if i < 2 {
                    self.extras_ice_sample(i);
                }
                let x = self.ram[0x90 + i].wrapping_sub(8);
                let y = self.ram[0x98 + i].wrapping_sub(8);
                let address = cell(i16::from(x), i16::from(y));
                self.ram[0xe0 + i] = address as u8;
                self.ram[0xe8 + i] = ((address >> 8) & 3) as u8
                    | if self.ram[0x90 + i] & 7 == 0 { 0x80 } else { 0 }
                    | if self.ram[0x98 + i] & 7 == 0 { 0x40 } else { 0 };
                address
            } else {
                usize::from(self.ram[0xe0 + i]) | ((usize::from(self.ram[0xe8 + i] & 3) | 4) << 8)
            };
            self.map_pointer(pointer);
            for (offset, enabled) in [
                (33, true),
                (32, self.ram[0xe8 + i] & 0x80 != 0),
                (1, self.ram[0xe8 + i] & 0x40 != 0),
            ] {
                if enabled && pointer + offset < self.ram.len() {
                    if occupied {
                        self.ram[pointer + offset] |= 0x80;
                    } else {
                        self.ram[pointer + offset] &= 0x7f;
                    }
                }
            }
        }
        self.ram[0x5a] = 255;
    }

    pub(super) fn update_enemies(&mut self) {
        for i in (2..8).rev() {
            self.ram[0x5a] = i as u8;
            let state = self.ram[0xa0 + i];
            if self.ram[0x100] != 0 && active(state) {
                continue;
            }
            if self.ram[0xa8 + i] & 0xf0 != 0xa0 && (i as u8 ^ self.ram[0x0b]) & 1 == 0 {
                continue;
            }
            // The indirect dispatch pointer is observable to the cartridge RNG.
            let address = [
                0xdbf0, 0xddea, 0xddea, 0xddea, 0xddea, 0xddea, 0xddea, 0xddea, 0xdc52, 0xdd48,
                0xdc7c, 0xdd94, 0xdd89, 0xdd7e, 0xde64, 0xde55,
            ][usize::from(state >> 4)];
            self.map_pointer(address);
            match state >> 4 {
                0 => {}
                1..=7 => {
                    self.ram[0xa0 + i] = animation_next(state, true);
                    if self.ram[0xa0 + i] == 0 {
                        self.ram[0x80] = self.ram[0x80].saturating_sub(1);
                    }
                }
                8 => {
                    self.ram[0xa0 + i] = state.wrapping_sub(4);
                    if self.ram[0xa0 + i] & 12 == 0 {
                        self.enemy_mode(i, 0xa0);
                    }
                }
                9 => {
                    if self.random_byte() & 1 == 0 {
                        self.choose_enemy_mode(i);
                    } else {
                        let delta = if self.random_byte() & 1 == 0 { 255 } else { 1 };
                        self.ram[0xa0 + i] = 0xa0 | (state.wrapping_add(delta) & 3);
                    }
                }
                10 => {
                    if self.ram[0x90 + i] & 7 == 0
                        && self.ram[0x98 + i] & 7 == 0
                        && self.random_byte() & 15 == 0
                    {
                        self.choose_enemy_mode(i);
                    } else {
                        self.move_enemy(i);
                    }
                }
                11 => self.aim_enemy(i, 120, 216),
                12 => self.aim_enemy(i, self.ram[0x91], self.ram[0x99]),
                13 => self.aim_enemy(i, self.ram[0x90], self.ram[0x98]),
                14 => {
                    if state & 15 == 13 {
                        self.activate_enemy(i);
                    } else {
                        self.ram[0xa0 + i] = state.wrapping_add(1);
                    }
                }
                15 => {
                    self.ram[0xa0 + i] = if state & 15 == 13 {
                        0xe0
                    } else {
                        state.wrapping_add(1)
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    fn enemy_mode(&mut self, i: usize, mode: u8) {
        self.ram[0] = mode;
        self.ram[0xa0 + i] = (self.ram[0xa0 + i] & 15) | mode;
    }

    fn choose_enemy_mode(&mut self, i: usize) {
        let elapsed = self.ram[0x0a];
        if elapsed > self.ram[0x84] >> 2 {
            self.enemy_mode(i, 0xb0);
        } else if elapsed <= self.ram[0x84] >> 3 {
            self.ram[0xa0 + i] = 0xa0 | (self.random_byte() & 3);
        } else if self.ram[0xa0] != 0 && (i & 1 == 0 || self.ram[0xa1] == 0) {
            self.enemy_mode(i, 0xd0);
        } else {
            self.enemy_mode(i, 0xc0);
        }
    }

    fn aim_enemy(&mut self, i: usize, x: u8, y: u8) {
        self.ram[0x71] = x;
        self.ram[0x72] = y;
        let sx = i16::from(x).cmp(&i16::from(self.ram[0x90 + i])) as i8;
        let sy = i16::from(y).cmp(&i16::from(self.ram[0x98 + i])) as i8;
        self.ram[0x65] = (sy + 1) as u8;
        self.ram[0x64] = ((sy + 1) * 3 + sx + 1) as u8;
        let prefer_x = self.random_byte() & 1 != 0;
        let face = if prefer_x && sx != 0 {
            if sx < 0 { 1 } else { 3 }
        } else if sy != 0 {
            if sy < 0 { 0 } else { 2 }
        } else if sx != 0 {
            if sx < 0 { 1 } else { 3 }
        } else {
            0
        };
        self.ram[0xa0 + i] = 0xa0 | face;
    }

    fn move_enemy(&mut self, i: usize) {
        if self.tank_advance(i) {
            self.ram[0xb0 + i] ^= 4;
            return;
        }
        if self.random_byte() & 3 == 0 {
            if self.ram[0x90 + i] & 7 == 0 && self.ram[0x98 + i] & 7 == 0 {
                self.enemy_mode(i, 0x90);
            }
            self.ram[0xa0 + i] ^= 2;
        } else {
            self.enemy_mode(i, 0x80);
            self.ram[0xa0 + i] |= 8;
            self.ram[0xb0 + i] ^= 4;
        }
    }

    pub(super) fn tank_advance(&mut self, i: usize) -> bool {
        let (dx, dy) = DELTA[usize::from(self.ram[0xa0 + i] & 3)];
        let x = i16::from(self.ram[0x90 + i]) + dx;
        let y = i16::from(self.ram[0x98 + i]) + dy;
        self.ram[0x56] = x as u8;
        self.ram[0x57] = y as u8;
        self.ram[0x58] = (dx * 8) as u8;
        self.ram[0x59] = (dy * 8) as u8;
        for (mut px, mut py) in [
            (x + dx * 8 + dy * 8, y + dx * 8 + dy * 8),
            (x + dx * 8 - dy * 8, y + dy * 8 - dx * 8),
        ] {
            if px >= x {
                px -= 1;
            }
            if py >= y {
                py -= 1;
            }
            let address = cell(i16::from(px as u8), i16::from(py as u8));
            self.map_pointer(address);
            let material = self.ram[address];
            if material & 0x80 != 0 || (material != 0 && material < 32) {
                return false;
            }
        }
        if !(24..=216).contains(&x) || !(24..=216).contains(&y) {
            return false;
        }
        self.ram[0x90 + i] = x as u8;
        self.ram[0x98 + i] = y as u8;
        true
    }

    pub(super) fn fire_enemies(&mut self) {
        if self.ram[0x100] != 0 {
            return;
        }
        for i in (2..8).rev() {
            if active(self.ram[0xa0 + i]) && self.random_byte() & 31 == 0 && self.ram[0xcc + i] == 0
            {
                let face = self.ram[0xa0 + i] & 3;
                let (dx, dy) = DELTA[usize::from(face)];
                self.ram[0xb8 + i] = (i16::from(self.ram[0x90 + i]) + dx * 8) as u8;
                self.ram[0xc2 + i] = (i16::from(self.ram[0x98 + i]) + dy * 8) as u8;
                self.ram[0xcc + i] = 0x40 | face;
                self.ram[0xd6 + i] = u8::from(self.ram[0xa8 + i] & 0xf0 == 0xc0);
            }
        }
    }

    pub(super) fn spawn_enemy(&mut self) {
        if self.ram[0x82] != 0 {
            self.ram[0x82] -= 1;
            return;
        }
        if self.ram[0x7f] == 0 {
            return;
        }
        let last = usize::from(self.ram[0x6c]);
        if !(2..8).contains(&last) {
            self.unsupported |= UNSUPPORTED_SPAWNING;
            return;
        }
        for i in (2..=last).rev() {
            self.ram[0x5a] = i as u8;
            if self.ram[0xa0 + i] != 0 {
                continue;
            }
            self.ram[0x82] = self.ram[0x84];
            self.ram[0x6a] = (self.ram[0x6a] + 1) % 3;
            self.ram[0x90 + i] = [24, 120, 216][usize::from(self.ram[0x6a])];
            self.ram[0x98 + i] = 24;
            let carrier = matches!(self.ram[0x7f], 3 | 10 | 17);
            self.ram[0xa8 + i] = if carrier { 4 } else { 0 };
            if carrier {
                self.ram[0x86] = 0;
            }
            self.ram[0xa0 + i] = 0xf0;
            self.ram[0x7f] -= 1;
            return;
        }
    }

    fn activate_enemy(&mut self, i: usize) {
        let stage = usize::from(if self.ram[0x46] != 0 {
            35
        } else {
            self.ram[0x85]
        });
        let mut group = usize::from(self.ram[0x8f]);
        while group < 4 && self.ram[0x8b + group] == 0 {
            group += 1;
        }
        if !(1..=35).contains(&stage) || group >= 4 {
            self.unsupported |= UNSUPPORTED_SPAWNING;
            return;
        }
        self.ram[0x8f] = group as u8;
        self.ram[0x8b + group] -= 1;
        let category = CATEGORIES[stage - 1][group];
        let armor = if category == 0xe0 && self.ram[0xa8 + i] & 4 == 0 {
            3
        } else {
            0
        };
        self.ram[0xa8 + i] |= category | armor;
        self.ram[0xa0 + i] = 0xa2;
        self.ram[0xb0 + i] = 0;
    }

    fn map_pointer(&mut self, address: usize) {
        self.ram[0x11] = address as u8;
        self.ram[0x12] = (address >> 8) as u8;
    }
}
