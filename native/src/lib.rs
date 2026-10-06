//! RAM mirror ABI: legacy parity probes and explicitly seeded experimental play.
//! Diagnostic builds expose their status and must not be used for training.
mod engine;
mod cartridge;
mod codec;
mod extras;
mod observe;
#[cfg(test)]
mod abi_tests;
#[cfg(feature = "reference-rng")]
mod random_replay;

use engine::Engine;
use std::slice;
pub use engine::{LIMITATIONS, SUPPORTED};

const MASKS: [u8; 10] = [0, 16, 32, 64, 128, 1, 17, 33, 65, 129];

struct Batch {
    engines: Vec<Engine>,
    max_frames: u32,
    stop_on_death: bool,
    terminal: Vec<bool>,
}

/// SplitMix64-v1, using the high byte of each complete 64-bit output.
#[unsafe(no_mangle)]
pub extern "C" fn mirror_rng_version() -> u32 { 1 }

/// cartridge-rng-v1: the cartridge $D44D arithmetic over mirrored zero page,
/// seed-derived $0F/$10/scratch phase, optional recorded scratch replay.
#[unsafe(no_mangle)]
pub extern "C" fn mirror_cartridge_rng_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub extern "C" fn mirror_diagnostic_build() -> u32 {
    u32::from(cfg!(feature = "reference-rng"))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_counters(
    handle: *mut std::ffi::c_void, index: usize, output: *mut u32,
) -> i32 {
    if handle.is_null() || output.is_null() { return -1; }
    let batch = unsafe { &*handle.cast::<Batch>() };
    if index >= batch.engines.len() { return -2; }
    let engine = &batch.engines[index];
    let values = [engine.frame, engine.kills, engine.deaths, engine.shots, engine.pickups];
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), output, 5) };
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_support(handle: *mut std::ffi::c_void, index: usize) -> u32 {
    if handle.is_null() { return u32::MAX; }
    let batch = unsafe { &mut *handle.cast::<Batch>() };
    if index >= batch.engines.len() { return u32::MAX; }
    batch.engines[index].unsupported()
}

/// Diagnostic build only: this is conditional parity, not autonomous simulation.
#[cfg(feature = "reference-rng")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_reference_step(
    handle: *mut std::ffi::c_void, index: usize, control: u8,
    values: *const u8, cursors: *const u8, count: usize,
) -> u32 {
    if handle.is_null() || (count != 0 && (values.is_null() || cursors.is_null())) {
        return u32::MAX;
    }
    let batch = unsafe { &mut *handle.cast::<Batch>() };
    if index >= batch.engines.len() || count > 4096 || batch.engines[index].is_seeded() {
        return u32::MAX;
    }
    let samples = if count == 0 {
        Vec::new()
    } else {
        let values = unsafe { slice::from_raw_parts(values, count) };
        let cursors = unsafe { slice::from_raw_parts(cursors, count) };
        values.iter().copied().zip(cursors.iter().copied()).collect()
    };
    batch.engines[index].step_with_reference_rng(control, &samples)
}

#[cfg(feature = "reference-rng")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_reference_logic(
    handle: *mut std::ffi::c_void, index: usize,
    input: u8, pressed: u8, tick: u8, coarse: u8,
    values: *const u8, cursors: *const u8, count: usize,
) -> u32 {
    if handle.is_null() || (count != 0 && (values.is_null() || cursors.is_null())) {
        return u32::MAX;
    }
    let batch = unsafe { &mut *handle.cast::<Batch>() };
    if index >= batch.engines.len() || count > 4096 || batch.engines[index].is_seeded() {
        return u32::MAX;
    }
    let samples = if count == 0 {
        Vec::new()
    } else {
        let values = unsafe { slice::from_raw_parts(values, count) };
        let cursors = unsafe { slice::from_raw_parts(cursors, count) };
        values.iter().copied().zip(cursors.iter().copied()).collect()
    };
    batch.engines[index].logic_with_reference_rng(input, pressed, tick, coarse, &samples)
}

#[cfg(feature = "reference-rng")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_native_logic(
    handle: *mut std::ffi::c_void, index: usize,
    input: u8, pressed: u8, tick: u8, coarse: u8,
) -> u32 {
    if handle.is_null() { return u32::MAX; }
    let batch = unsafe { &mut *handle.cast::<Batch>() };
    if index >= batch.engines.len() { return u32::MAX; }
    batch.engines[index].logic_with_native_rng(input, pressed, tick, coarse)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_create(ram: *const u8, count: usize, max_frames: u32) -> *mut std::ffi::c_void {
    if ram.is_null() || count == 0 || count > 4096 || max_frames == 0 {
        return std::ptr::null_mut();
    }
    let data = unsafe { slice::from_raw_parts(ram, count * 2048) };
    let engines = data.chunks_exact(2048).map(|r| {
        let mut initial = [0; 2048];
        initial.copy_from_slice(r);
        Engine::new(initial)
    }).collect();
    Box::into_raw(Box::new(Batch {
        engines, max_frames, stop_on_death: true, terminal: vec![false; count],
    })).cast()
}

/// Opt-in experimental full-game/multi-life simulation, not NES-equivalent RNG.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_create_seeded(
    ram: *const u8, count: usize, max_frames: u32, seeds: *const u64,
) -> *mut std::ffi::c_void {
    if ram.is_null() || seeds.is_null() || count == 0 || count > 4096 || max_frames == 0 {
        return std::ptr::null_mut();
    }
    let data = unsafe { slice::from_raw_parts(ram, count * engine::RAM_LEN) };
    let seeds = unsafe { slice::from_raw_parts(seeds, count) };
    let engines = data.chunks_exact(engine::RAM_LEN).zip(seeds).map(|(r, &seed)| {
        let mut initial = [0; engine::RAM_LEN];
        initial.copy_from_slice(r);
        Engine::new_seeded(initial, seed)
    }).collect();
    Box::into_raw(Box::new(Batch {
        engines, max_frames, stop_on_death: false, terminal: vec![false; count],
    })).cast()
}

/// Opt-in experimental cartridge-formula randomness: $D44D arithmetic over the
/// mirrored zero page, seed-derived phase, optional recorded scratch replay.
/// The scratch rows are row-major `[row_count][address_count]` boundary
/// captures; addresses colliding with mirrored gameplay state are rejected.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_create_cartridge(
    ram: *const u8, count: usize, max_frames: u32, seeds: *const u64,
    scratch_addresses: *const u8, address_count: usize,
    scratch_values: *const u8, value_count: usize,
) -> *mut std::ffi::c_void {
    if ram.is_null() || seeds.is_null() || count == 0 || count > 4096 || max_frames == 0 {
        return std::ptr::null_mut();
    }
    if (address_count != 0 && scratch_addresses.is_null())
        || (value_count != 0 && scratch_values.is_null())
        || address_count > 256
        || value_count > 1 << 26
    {
        return std::ptr::null_mut();
    }
    let addresses = if address_count == 0 {
        Vec::new()
    } else {
        unsafe { slice::from_raw_parts(scratch_addresses, address_count) }.to_vec()
    };
    let values = if value_count == 0 {
        Vec::new()
    } else {
        unsafe { slice::from_raw_parts(scratch_values, value_count) }.to_vec()
    };
    let Ok(trace) = cartridge::ScratchTrace::new(addresses, values) else {
        return std::ptr::null_mut();
    };
    let trace = std::sync::Arc::new(trace);
    let data = unsafe { slice::from_raw_parts(ram, count * engine::RAM_LEN) };
    let seeds = unsafe { slice::from_raw_parts(seeds, count) };
    let engines = data.chunks_exact(engine::RAM_LEN).zip(seeds).map(|(r, &seed)| {
        let mut initial = [0; engine::RAM_LEN];
        initial.copy_from_slice(r);
        Engine::new_cartridge(initial, seed, trace.clone())
    }).collect();
    Box::into_raw(Box::new(Batch {
        engines, max_frames, stop_on_death: false, terminal: vec![false; count],
    })).cast()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_create_full(
    ram: *const u8, count: usize, max_frames: u32,
) -> *mut std::ffi::c_void {
    let handle = unsafe { mirror_create(ram, count, max_frames) };
    if !handle.is_null() {
        unsafe { &mut *handle.cast::<Batch>() }.stop_on_death = false;
    }
    handle
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_close(handle: *mut std::ffi::c_void) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle.cast::<Batch>()) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_ram(handle: *mut std::ffi::c_void, index: usize, output: *mut u8) -> i32 {
    if handle.is_null() || output.is_null() { return -1; }
    let batch = unsafe { &mut *handle.cast::<Batch>() };
    if index >= batch.engines.len() { return -2; }
    unsafe { std::ptr::copy_nonoverlapping(batch.engines[index].ram.as_ptr(), output, 2048) };
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_frame(handle: *mut std::ffi::c_void, index: usize, control: u8) -> i32 {
    if handle.is_null() { return -1; }
    let batch = unsafe { &mut *handle.cast::<Batch>() };
    if index >= batch.engines.len() { return -2; }
    batch.engines[index].step(control);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_reset(handle: *mut std::ffi::c_void, index: usize, obs: *mut f32) -> i32 {
    if handle.is_null() || obs.is_null() { return -1; }
    let batch = unsafe { &mut *handle.cast::<Batch>() };
    if index >= batch.engines.len() { return -2; }
    let engine = &mut batch.engines[index];
    engine.reset();
    batch.terminal[index] = false;
    observe::write(engine, batch.max_frames, unsafe { slice::from_raw_parts_mut(obs, observe::SIZE) });
    0
}

/// Restore the original RAM and restart the seeded stream. Legacy handles return -3.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_reset_seeded(
    handle: *mut std::ffi::c_void, index: usize, seed: u64, obs: *mut f32,
) -> i32 {
    if handle.is_null() || obs.is_null() { return -1; }
    let batch = unsafe { &mut *handle.cast::<Batch>() };
    if index >= batch.engines.len() { return -2; }
    let engine = &mut batch.engines[index];
    if !engine.reseed(seed) { return -3; }
    batch.terminal[index] = false;
    observe::write(engine, batch.max_frames, unsafe { slice::from_raw_parts_mut(obs, observe::SIZE) });
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_observe(handle: *mut std::ffi::c_void, obs: *mut f32, len: usize) -> i32 {
    if handle.is_null() || obs.is_null() { return -1; }
    let batch = unsafe { &mut *handle.cast::<Batch>() };
    if len != batch.engines.len() * observe::SIZE { return -2; }
    let out = unsafe { slice::from_raw_parts_mut(obs, len) };
    for (engine, target) in batch.engines.iter().zip(out.chunks_exact_mut(observe::SIZE)) {
        observe::write(engine, batch.max_frames, target);
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_masks(handle: *mut std::ffi::c_void, masks: *mut u8, len: usize) -> i32 {
    if handle.is_null() || masks.is_null() { return -1; }
    let batch = unsafe { &mut *handle.cast::<Batch>() };
    if len != batch.engines.len() * 10 { return -2; }
    let out = unsafe { slice::from_raw_parts_mut(masks, len) };
    for (engine, target) in batch.engines.iter().zip(out.chunks_exact_mut(10)) {
        observe::masks(engine, target);
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_step(
    handle: *mut std::ffi::c_void, actions: *const u8, count: usize,
    obs: *mut f32, rewards: *mut f32, dones: *mut u8, statistics: *mut u32,
) -> i32 {
    if handle.is_null() || actions.is_null() || obs.is_null() || rewards.is_null()
        || dones.is_null() || statistics.is_null() { return -1; }
    let batch = unsafe { &mut *handle.cast::<Batch>() };
    if count != batch.engines.len() { return -2; }
    let actions = unsafe { slice::from_raw_parts(actions, count) };
    if actions.iter().any(|a| usize::from(*a) >= MASKS.len()) { return -3; }
    let obs = unsafe { slice::from_raw_parts_mut(obs, count * observe::SIZE) };
    let rewards = unsafe { slice::from_raw_parts_mut(rewards, count) };
    let dones = unsafe { slice::from_raw_parts_mut(dones, count) };
    let statistics = unsafe { slice::from_raw_parts_mut(statistics, count * 5) };
    for i in 0..count {
        let engine = &mut batch.engines[i];
        if engine.is_seeded() && batch.terminal[i] {
            dones[i] = 1;
            rewards[i] = 0.0;
            statistics[i * 5..i * 5 + 5].copy_from_slice(
                &[engine.frame, engine.kills, engine.deaths, engine.shots, engine.pickups]);
            observe::write(engine, batch.max_frames, &mut obs[i * observe::SIZE..(i + 1) * observe::SIZE]);
            continue;
        }
        let (kills, deaths, frame) = (engine.kills, engine.deaths, engine.frame);
        let mut failed = false;
        for f in 0..4 {
            let mask = MASKS[usize::from(actions[i])];
            engine.step(if f < 2 { mask } else { mask & !1 });
            failed = engine.ram[0x68] != 0x80 || engine.ram[0x51] == 0
                || (engine.deaths > deaths && (batch.stop_on_death || engine.ram[0x51] <= 1));
            if failed
                || engine.ram[0x80] == 0 || engine.frame >= batch.max_frames {
                break;
            }
        }
        let elapsed = engine.frame - frame;
        let clear = !failed && engine.ram[0x80] == 0;
        dones[i] = u8::from(failed || clear || engine.frame >= batch.max_frames);
        if engine.is_seeded() { batch.terminal[i] = dones[i] != 0; }
        rewards[i] = (engine.kills - kills) as f32 - 3.0 * (engine.deaths - deaths) as f32
            - 0.0005 * elapsed as f32 - if engine.ram[0x68] != 0x80 { 20.0 } else { 0.0 }
            + if clear { 5.0 } else { 0.0 };
        statistics[i * 5..i * 5 + 5].copy_from_slice(
            &[engine.frame, engine.kills, engine.deaths, engine.shots, engine.pickups]);
        observe::write(engine, batch.max_frames, &mut obs[i * observe::SIZE..(i + 1) * observe::SIZE]);
    }
    0
}

/// Fused outputs: mirror_step remains the only action-repeat/reward implementation.
/// Each environment writes SIZE observations, 5 statistics and 10 action masks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_step_batch(
    handle: *mut std::ffi::c_void, actions: *const u8, count: usize,
    obs: *mut f32, rewards: *mut f32, dones: *mut u8, stats: *mut u32,
    support: *mut u32, masks: *mut u8,
) -> i32 {
    if support.is_null() || masks.is_null() { return -1; }
    let status = unsafe { mirror_step(handle, actions, count, obs, rewards, dones, stats) };
    if status != 0 { return status; }
    let batch = unsafe { &*handle.cast::<Batch>() };
    let support = unsafe { slice::from_raw_parts_mut(support, count) };
    let masks = unsafe { slice::from_raw_parts_mut(masks, count * 10) };
    for (i, engine) in batch.engines.iter().enumerate() {
        support[i] = engine.unsupported();
        observe::masks(engine, &mut masks[i * 10..(i + 1) * 10]);
    }
    0
}
