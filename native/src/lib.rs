//! Isolated feasibility ABI; controlled scenes only, not a replacement NES.
mod engine;
mod extras;
mod observe;
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
    if index >= batch.engines.len() || count > 4096 { return u32::MAX; }
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
    if index >= batch.engines.len() || count > 4096 { return u32::MAX; }
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
    batch.engines[index].logic_with_native_rng(input, pressed, tick, coarse);
    0
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
    Box::into_raw(Box::new(Batch { engines, max_frames, stop_on_death: true })).cast()
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
        rewards[i] = (engine.kills - kills) as f32 - 3.0 * (engine.deaths - deaths) as f32
            - 0.0005 * elapsed as f32 - if engine.ram[0x68] != 0x80 { 20.0 } else { 0.0 }
            + if clear { 5.0 } else { 0.0 };
        statistics[i * 5..i * 5 + 5].copy_from_slice(
            &[engine.frame, engine.kills, engine.deaths, engine.shots, engine.pickups]);
        observe::write(engine, batch.max_frames, &mut obs[i * observe::SIZE..(i + 1) * observe::SIZE]);
    }
    0
}
