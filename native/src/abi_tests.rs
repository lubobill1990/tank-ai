use super::*;
use std::ffi::c_void;
use std::ptr::{null, null_mut};

struct Handle(*mut c_void);

impl Drop for Handle {
    fn drop(&mut self) { unsafe { mirror_close(self.0) }; }
}

fn scene() -> [u8; 2048] {
    let mut ram = [0; 2048];
    ram[0x0b] = 32;
    ram[0x51] = 3;
    ram[0x68] = 0x80;
    ram[0x80] = 20;
    ram[0x85] = 1;
    ram[0x90] = 88;
    ram[0x98] = 160;
    ram[0xa0] = 0x88;
    ram
}

fn enemies() -> [u8; 2048] {
    let mut ram = scene();
    ram[0x92] = 88;
    ram[0x9a] = 80;
    ram[0xa2] = 0xa2;
    ram[0xaa] = 0x80;
    ram[0x84] = 186;
    ram[0x6c] = 5;
    ram
}

fn seeded(ram: &[u8], seeds: &[u64], max_frames: u32) -> Handle {
    assert_eq!(ram.len(), seeds.len() * 2048);
    let handle = Handle(unsafe {
        mirror_create_seeded(ram.as_ptr(), seeds.len(), max_frames, seeds.as_ptr())
    });
    assert!(!handle.0.is_null());
    handle
}

fn ram(handle: &Handle, index: usize) -> [u8; 2048] {
    let mut out = [0; 2048];
    assert_eq!(unsafe { mirror_ram(handle.0, index, out.as_mut_ptr()) }, 0);
    out
}

#[derive(Debug, PartialEq)]
struct Output {
    obs: Vec<f32>,
    rewards: Vec<f32>,
    dones: Vec<u8>,
    stats: Vec<u32>,
    support: Vec<u32>,
    masks: Vec<u8>,
}

impl Output {
    fn new(count: usize) -> Self {
        Self {
            obs: vec![7.0; count * observe::SIZE],
            rewards: vec![7.0; count], dones: vec![7; count],
            stats: vec![7; count * 5], support: vec![7; count], masks: vec![7; count * 10],
        }
    }

    fn step(&mut self, handle: &Handle, actions: &[u8], fused: bool) -> i32 {
        unsafe {
            if fused {
                mirror_step_batch(handle.0, actions.as_ptr(), actions.len(), self.obs.as_mut_ptr(),
                    self.rewards.as_mut_ptr(), self.dones.as_mut_ptr(), self.stats.as_mut_ptr(),
                    self.support.as_mut_ptr(), self.masks.as_mut_ptr())
            } else {
                let status = mirror_step(handle.0, actions.as_ptr(), actions.len(), self.obs.as_mut_ptr(),
                    self.rewards.as_mut_ptr(), self.dones.as_mut_ptr(), self.stats.as_mut_ptr());
                if status == 0 {
                    for i in 0..actions.len() { self.support[i] = mirror_support(handle.0, i); }
                    assert_eq!(mirror_masks(handle.0, self.masks.as_mut_ptr(), self.masks.len()), 0);
                }
                status
            }
        }
    }
}

#[test]
fn seeded_abi_rejects_invalid_creation_arguments_and_identifies_build() {
    let r = scene();
    let seed = 0;
    unsafe {
        for (data, count, frames, seeds) in [
            (null(), 1, 100, &seed as *const u64),
            (r.as_ptr(), 1, 100, null()),
            (r.as_ptr(), 0, 100, &seed),
            (r.as_ptr(), 4097, 100, &seed),
            (r.as_ptr(), usize::MAX, 100, &seed),
            (r.as_ptr(), 1, 0, &seed),
        ] {
            assert!(mirror_create_seeded(data, count, frames, seeds).is_null());
        }
    }
    assert_eq!(mirror_rng_version(), 1);
    assert_eq!(mirror_diagnostic_build(), u32::from(cfg!(feature = "reference-rng")));
}

#[test]
fn seeded_vector_order_and_interleaved_batches_reproduce_scalar_trajectories() {
    let r = enemies();
    let data = [r, r].concat();
    let ordered = seeded(&data, &[0, 1], 200);
    let reversed = seeded(&data, &[1, 0], 200);
    let solo = seeded(&r, &[0], 200);
    let mut pair_output = Output::new(2);
    let mut solo_output = Output::new(1);
    for i in 0..20 {
        let actions = [i % 10, (i + 3) % 10];
        assert_eq!(pair_output.step(&ordered, &actions, true), 0);
        assert_eq!(solo_output.step(&solo, &actions[..1], false), 0);
        assert_eq!(pair_output.step(&reversed, &[actions[1], actions[0]], false), 0);
        assert_eq!(ram(&ordered, 0), ram(&reversed, 1));
        assert_eq!(ram(&ordered, 1), ram(&reversed, 0));
        assert_eq!(ram(&ordered, 0), ram(&solo, 0));
    }
    assert_ne!(ram(&ordered, 0), ram(&ordered, 1));
}

#[test]
fn seeded_abi_reset_reseeds_and_legacy_reset_preserves_mode() {
    let r = enemies();
    let handle = seeded(&r, &[0], 100);
    let mut out = Output::new(1);
    out.step(&handle, &[9], true);
    let after_step = ram(&handle, 0);
    unsafe {
        assert_eq!(mirror_reset(handle.0, 0, out.obs.as_mut_ptr()), 0);
    }
    assert_eq!(ram(&handle, 0), r);
    out.step(&handle, &[9], true);
    assert_eq!(ram(&handle, 0), after_step);
    unsafe { assert_eq!(mirror_reset_seeded(handle.0, 0, 1, out.obs.as_mut_ptr()), 0); }
    assert_eq!(ram(&handle, 0), r);
    out.step(&handle, &[9], true);
    assert_ne!(ram(&handle, 0), after_step);
    let first_seed_one = ram(&handle, 0);
    unsafe { assert_eq!(mirror_reset(handle.0, 0, out.obs.as_mut_ptr()), 0); }
    out.step(&handle, &[9], true);
    assert_eq!(ram(&handle, 0), first_seed_one);
    let legacy = Handle(unsafe { mirror_create(r.as_ptr(), 1, 100) });
    unsafe { assert_eq!(mirror_reset_seeded(legacy.0, 0, 0, out.obs.as_mut_ptr()), -3); }
    assert_eq!(ram(&legacy, 0), r);
}

#[test]
fn fused_step_matches_separate_step_support_and_masks_for_both_modes() {
    let data = [enemies(), scene()].concat();
    for is_seeded in [false, true] {
        let create = || if is_seeded { seeded(&data, &[13, 42], 50) } else {
            Handle(unsafe { mirror_create_full(data.as_ptr(), 2, 50) })
        };
        let a = create();
        let b = create();
        let mut fused = Output::new(2);
        let mut separate = Output::new(2);
        for i in 0..15 {
            let actions = [i % 10, (i + 5) % 10];
            assert_eq!(fused.step(&a, &actions, true), 0);
            assert_eq!(separate.step(&b, &actions, false), 0);
            assert_eq!(fused, separate);
            assert_eq!(ram(&a, 0), ram(&b, 0));
            assert_eq!(ram(&a, 1), ram(&b, 1));
        }
    }
}

#[test]
fn fused_invalid_calls_do_not_mutate_any_engine_or_output() {
    let r = enemies();
    let handle = seeded(&[r, r].concat(), &[0, 1], 100);
    let mut out = Output::new(2);
    for missing in 0..8 {
        let actions = [0, 0];
        let status = unsafe {
            mirror_step_batch(
                if missing == 0 { null_mut() } else { handle.0 },
                if missing == 1 { null() } else { actions.as_ptr() }, 2,
                if missing == 2 { null_mut() } else { out.obs.as_mut_ptr() },
                if missing == 3 { null_mut() } else { out.rewards.as_mut_ptr() },
                if missing == 4 { null_mut() } else { out.dones.as_mut_ptr() },
                if missing == 5 { null_mut() } else { out.stats.as_mut_ptr() },
                if missing == 6 { null_mut() } else { out.support.as_mut_ptr() },
                if missing == 7 { null_mut() } else { out.masks.as_mut_ptr() },
            )
        };
        assert_eq!(status, -1);
        assert_eq!(ram(&handle, 0), r);
        assert_eq!(ram(&handle, 1), r);
        assert_eq!(out, Output::new(2));
    }
    assert_eq!(out.step(&handle, &[], true), -2);
    assert_eq!(out.step(&handle, &[0], true), -2);
    assert_eq!(out.step(&handle, &[0, 10], true), -3);
    assert_eq!(ram(&handle, 0), r);
    assert_eq!(ram(&handle, 1), r);
    assert_eq!(out, Output::new(2));
    let control = seeded(&[r, r].concat(), &[0, 1], 100);
    let mut expected = Output::new(2);
    out.step(&handle, &[0, 0], true);
    expected.step(&control, &[0, 0], true);
    assert_eq!(out, expected);
}

#[test]
fn seeded_reset_bad_arguments_leave_state_and_observation_unchanged() {
    let r = enemies();
    let handle = seeded(&r, &[0], 100);
    let mut out = Output::new(1);
    unsafe {
        assert_eq!(mirror_reset_seeded(null_mut(), 0, 1, out.obs.as_mut_ptr()), -1);
        assert_eq!(mirror_reset_seeded(handle.0, 0, 1, null_mut()), -1);
        assert_eq!(mirror_reset_seeded(handle.0, 1, 1, out.obs.as_mut_ptr()), -2);
        assert_eq!(mirror_reset_seeded(handle.0, usize::MAX, 1, out.obs.as_mut_ptr()), -2);
    }
    assert_eq!(ram(&handle, 0), r);
    assert_eq!(out, Output::new(1));
    let control = seeded(&r, &[0], 100);
    let mut expected = Output::new(1);
    out.step(&handle, &[0], true);
    expected.step(&control, &[0], true);
    assert_eq!(out, expected);
}

fn fatal_bullet(lives: u8) -> [u8; 2048] {
    let mut r = scene();
    r[0x51] = lives;
    r[0xba] = 88;
    r[0xc4] = 150;
    r[0xce] = 0x42;
    r
}

#[test]
fn seeded_full_game_survives_nonfinal_death_and_respawns() {
    let r = fatal_bullet(3);
    let handle = seeded(&r, &[0], 500);
    let legacy = Handle(unsafe { mirror_create(r.as_ptr(), 1, 500) });
    let full = Handle(unsafe { mirror_create_full(r.as_ptr(), 1, 500) });
    let mut out = Output::new(1);
    let mut full_out = Output::new(1);
    assert_eq!(out.step(&handle, &[0], true), 0);
    assert_eq!((out.dones[0], out.stats[2], out.stats[0]), (0, 1, 4));
    assert_eq!(out.rewards[0], -3.002);
    full_out.step(&full, &[0], false);
    assert_eq!(out, full_out);
    let mut legacy_out = Output::new(1);
    legacy_out.step(&legacy, &[0], false);
    assert_eq!(legacy_out.dones[0], 1);
    for _ in 0..35 { out.step(&handle, &[0], true); }
    let result = ram(&handle, 0);
    assert_eq!(result[0x51], 2);
    assert!((0x80..0xe0).contains(&result[0xa0]));
    assert_eq!(out.dones[0], 0);
}

#[test]
fn seeded_final_death_truncation_and_clear_are_cached_until_reset() {
    let mut clear = scene();
    clear[0x80] = 1;
    clear[0xa2] = 0x10;
    clear[0xaa] = 0xa0;
    let mut base_loss = scene();
    base_loss[0x68] = 0;
    for (initial, limit, frame, deaths) in [
        (fatal_bullet(1), 100, 1, 1),
        (scene(), 3, 3, 0),
        (clear, 100, 1, 0),
        (base_loss, 100, 1, 0),
    ] {
        let handle = seeded(&initial, &[0], limit);
        let mut out = Output::new(1);
        assert_eq!(out.step(&handle, &[0], true), 0);
        assert_eq!((out.dones[0], out.stats[0], out.stats[2]), (1, frame, deaths));
        let frozen_ram = ram(&handle, 0);
        let first_obs = out.obs.clone();
        let first_stats = out.stats.clone();
        for _ in 0..3 {
            assert_eq!(out.step(&handle, &[9], true), 0);
            assert_eq!(out.dones[0], 1);
            assert_eq!(out.rewards[0], 0.0);
            assert_eq!(out.obs, first_obs);
            assert_eq!(out.stats, first_stats);
            assert_eq!(ram(&handle, 0), frozen_ram);
        }
        unsafe { assert_eq!(mirror_reset_seeded(handle.0, 0, 0, out.obs.as_mut_ptr()), 0); }
        out.step(&handle, &[0], true);
        assert_eq!(out.stats[0], frame);
        assert_eq!(out.obs, first_obs);
        assert_ne!(out.rewards[0], 0.0);
    }
}

#[test]
fn seeded_unknown_terrain_and_p2_remain_unsupported_through_fused_api() {
    let mut r = enemies();
    r[0x442] = 255;
    r[0xa1] = 0x88;
    let handle = seeded(&r, &[0], 100);
    let mut out = Output::new(1);
    out.step(&handle, &[0], true);
    assert_eq!(out.support[0], engine::UNSUPPORTED_TERRAIN | engine::UNSUPPORTED_P2);
}

#[cfg(feature = "reference-rng")]
#[test]
fn diagnostic_abi_rejects_seeded_engines_without_mutation() {
    let r = enemies();
    let handle = seeded(&r, &[0], 100);
    unsafe {
        assert_eq!(mirror_reference_step(handle.0, 0, 128, null(), null(), 0), u32::MAX);
        assert_eq!(mirror_reference_logic(handle.0, 0, 128, 128, 0, 0, null(), null(), 0), u32::MAX);
        assert_eq!(mirror_native_logic(handle.0, 0, 128, 128, 0, 0), u32::MAX);
    }
    assert_eq!(ram(&handle, 0), r);
    let control = seeded(&r, &[0], 100);
    let mut out = Output::new(1);
    let mut expected = Output::new(1);
    out.step(&handle, &[0], true);
    expected.step(&control, &[0], true);
    assert_eq!(out, expected);
}
