use crate::engine::Engine;

const N: usize = 26;
const PLANE: usize = N * N;
pub const SIZE: usize = 47 * PLANE;
const DX: [i32; 4] = [0, -1, 0, 1];
const DY: [i32; 4] = [-1, 0, 1, 0];

fn active(v: u8) -> bool {
    (0x80..0xe0).contains(&v)
}

fn clean(value: u8) -> u8 {
    let v = value & 0x7f;
    if value >= 128 && (v <= 16 || [18, 32, 33, 34].contains(&v)) {
        v
    } else {
        value
    }
}

fn cells(x: i32, y: i32, mut visit: impl FnMut(usize)) {
    for r in ((y - 24).div_euclid(8)).max(0)..=((y - 9).div_euclid(8)).min(25) {
        for c in ((x - 24).div_euclid(8)).max(0)..=((x - 9).div_euclid(8)).min(25) {
            visit(r as usize * N + c as usize);
        }
    }
}

fn box_time(x: f32, y: f32, vx: f32, vy: f32, bounds: [f32; 4]) -> f32 {
    let (mut enter, mut leave) = (0.0f32, f32::INFINITY);
    for (p, v, lo, hi) in [(x, vx, bounds[0], bounds[2]), (y, vy, bounds[1], bounds[3])] {
        if v == 0.0 {
            if p < lo || p > hi {
                return f32::INFINITY;
            }
        } else {
            let (a, b) = ((lo - p) / v, (hi - p) / v);
            enter = enter.max(a.min(b));
            leave = leave.min(a.max(b));
        }
    }
    if enter <= leave { enter } else { f32::INFINITY }
}

fn obstruction(ram: &[u8; 2048], x: f32, y: f32, vx: f32, vy: f32) -> f32 {
    let (ex, ey) = (x + vx * 24.0, y + vy * 24.0);
    let c0 = (((x.min(ex) - 17.0) / 8.0).floor() as i32).max(0);
    let c1 = (((x.max(ex) - 15.0) / 8.0).ceil() as i32).min(26);
    let r0 = (((y.min(ey) - 17.0) / 8.0).floor() as i32).max(0);
    let r1 = (((y.max(ey) - 15.0) / 8.0).ceil() as i32).min(26);
    let mut hit = f32::INFINITY;
    for r in r0..r1 {
        for c in c0..c1 {
            let value = clean(ram[0x400 + (r as usize + 2) * 32 + c as usize + 2]);
            if [0, 18, 32, 33, 34].contains(&value) {
                continue;
            }
            let (left, top) = (16.0 + c as f32 * 8.0, 16.0 + r as f32 * 8.0);
            if value > 0 && value < 16 {
                for bit in 0..4 {
                    if value & (1 << bit) != 0 {
                        let (l, t) = (left + (bit % 2) as f32 * 4.0, top + (bit / 2) as f32 * 4.0);
                        hit = hit.min(box_time(
                            x,
                            y,
                            vx,
                            vy,
                            [
                                l,
                                t,
                                l + 4.0 - f32::from(vx != 0.0),
                                t + 4.0 - f32::from(vy != 0.0),
                            ],
                        ));
                    }
                }
            } else {
                hit = hit.min(box_time(
                    x,
                    y,
                    vx,
                    vy,
                    [
                        left,
                        top,
                        left + 8.0 - f32::from(vx != 0.0),
                        top + 8.0 - f32::from(vy != 0.0),
                    ],
                ));
            }
        }
    }
    if hit <= 24.0 { hit } else { f32::INFINITY }
}

pub fn masks(engine: &Engine, out: &mut [u8]) {
    out.fill(1);
    let r = &engine.ram;
    let power = (r[0x101] >> 5) & 3;
    out[5] = u8::from(active(r[0xa0]) && (r[0xcc] == 0 || (power >= 2 && r[0xd4] == 0)));
    if active(r[0xa0]) {
        let facing = usize::from(r[0xa0] & 3);
        let (x, y) = (r[0x90], r[0x98]);
        let edge = (facing == 0 && y <= 24)
            || (facing == 1 && x <= 24)
            || (facing == 2 && y >= 216)
            || (facing == 3 && x >= 216);
        if edge {
            let public = [0, 2, 1, 3][facing];
            out[public + 1] = 0;
            out[public + 6] = 0;
            out[5] = 0;
        }
    }
}

pub fn write(engine: &Engine, max_frames: u32, out: &mut [f32]) {
    out.fill(0.0);
    let ram = &engine.ram;
    for row in 0..N {
        for col in 0..N {
            let idx = row * N + col;
            let v = clean(ram[0x400 + (row + 2) * 32 + col + 2]);
            let channel = match v {
                1..=15 => Some(0),
                16 => Some(1),
                18 => Some(2),
                33 => Some(3),
                0 | 32 | 34 | 200..=207 => None,
                _ => Some(1),
            };
            if let Some(ch) = channel {
                out[ch * PLANE + idx] = 1.0;
            }
            if (1..=15).contains(&v) {
                for bit in 0..4 {
                    out[(39 + bit) * PLANE + idx] = f32::from(v & (1 << bit) != 0);
                }
            }
        }
    }
    for r in 24..26 {
        for c in 12..14 {
            out[4 * PLANE + r * N + c] = 1.0;
        }
    }
    for slot in 0..8 {
        if !active(ram[0xa0 + slot]) || slot == 1 {
            continue;
        }
        let x = i32::from(ram[0x90 + slot]);
        let y = i32::from(ram[0x98 + slot]);
        let face = usize::from(ram[0xa0 + slot] & 3);
        cells(x, y, |idx| {
            if slot == 0 {
                out[5 * PLANE + idx] = 1.0;
                out[6 * PLANE + idx] = DX[face] as f32;
                out[7 * PLANE + idx] = DY[face] as f32;
                out[43 * PLANE + idx] = (x - 16).rem_euclid(8) as f32 / 7.0;
                out[44 * PLANE + idx] = (y - 16).rem_euclid(8) as f32 / 7.0;
            } else {
                out[8 * PLANE + idx] += 1.0 / 6.0;
                out[9 * PLANE + idx] += DX[face] as f32;
                out[10 * PLANE + idx] += DY[face] as f32;
                let attr = ram[0xa8 + slot];
                let armor = if attr == 0xe4 { 4 } else { (attr & 3) + 1 };
                out[11 * PLANE + idx] = out[11 * PLANE + idx].max(f32::from(armor) / 4.0);
                if attr & 4 != 0 {
                    out[46 * PLANE + idx] = 1.0;
                }
            }
        });
    }
    for slot in 0..10 {
        if ram[0xcc + slot] & 0xf0 != 0x40 {
            continue;
        }
        let owner = slot & 7;
        if owner == 1 {
            continue;
        }
        let face = usize::from(ram[0xcc + slot] & 3);
        let speed = if ram[0xd6 + slot] & 1 != 0 { 4 } else { 2 };
        let (vx, vy) = (DX[face] * speed, DY[face] * speed);
        let (x, y) = (i32::from(ram[0xb8 + slot]), i32::from(ram[0xc2 + slot]));
        let (row, col) = ((y - 16).div_euclid(8), (x - 16).div_euclid(8));
        if (0..26).contains(&row) && (0..26).contains(&col) {
            let idx = row as usize * N + col as usize;
            let ch = if owner == 0 { 12 } else { 15 };
            out[ch * PLANE + idx] += if owner == 0 { 0.5 } else { 1.0 / 6.0 };
            out[(ch + 1) * PLANE + idx] += vx as f32 / 4.0;
            out[(ch + 2) * PLANE + idx] += vy as f32 / 4.0;
        }
        if owner >= 2 && ram[0x68] == 0x80 && ram[0x51] != 0 {
            let blocked = obstruction(ram, x as f32, y as f32, vx as f32, vy as f32);
            for r in 1..26 {
                for c in 1..26 {
                    let cx = (16 + c * 8) as f32;
                    let cy = (16 + r * 8) as f32;
                    let t = box_time(
                        x as f32,
                        y as f32,
                        vx as f32,
                        vy as f32,
                        [cx - 9.0, cy - 9.0, cx + 9.0, cy + 9.0],
                    );
                    if t <= 24.0 && t < blocked {
                        let idx = 45 * PLANE + r * N + c;
                        out[idx] = out[idx].max((25.0 - t) / 25.0);
                    }
                }
            }
        }
    }
    let alive = active(ram[0xa0]);
    let power = (ram[0x101] >> 5) & 3;
    let scalars = [
        f32::from(ram[0x80]) / 20.0,
        f32::from(ram[0x7f]) / 20.0,
        f32::from(ram[0x51]) / 10.0,
        if alive { f32::from(power) / 3.0 } else { 0.0 },
        f32::from(alive && ram[0x89] != 0),
        f32::from(ram[0x100] != 0),
        f32::from(alive && (ram[0xcc] == 0 || (power >= 2 && ram[0xd4] == 0))),
    ];
    for (i, v) in scalars.iter().enumerate() {
        out[(18 + i) * PLANE..(19 + i) * PLANE].fill(*v);
    }
    if ram[0x86] != 0 && ram[0x62] == 0 && ram[0x88] < 6 {
        let (r, c) = (
            (i32::from(ram[0x87]) - 16).div_euclid(8),
            (i32::from(ram[0x86]) - 16).div_euclid(8),
        );
        if (0..26).contains(&r) && (0..26).contains(&c) {
            out[(25 + usize::from(ram[0x88])) * PLANE + r as usize * N + c as usize] = 1.0;
        }
    }
    out[36 * PLANE..37 * PLANE]
        .fill((1.0_f64 - engine.frame as f64 / max_frames as f64).max(0.0) as f32);
    for v in out {
        *v = v.clamp(-1.0, 1.0);
    }
}
