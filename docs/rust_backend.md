# Experimental Rust gameplay mirror

The ROM environment remains the production reference. The Rust source in
`native/` is the latest mechanics implementation from the alignment experiment,
not the older performance worktree's engine. Memory work must build on this
source, preserving its enemy logic, item/timer behavior, projectile priority and
per-probe residual-brick collision fixes.

## Verified boundary

The original experiment compared 35 stages, two seeds and four control
schedules: 280/280 trajectories matched through their natural terminal, across
535,811 logical gameplay updates and 2,333,878 reference RNG calls. Entry was
before ROM dispatch at `$C200`, exit before `$C206`. Clock and delayed input
progression were derived locally; only random outputs came from the reference.
These are conditional mechanics comparisons, not independent ROM equivalence,
win rates, or a claim of "99% parity".

Independent cartridge-derived RNG still diverged after 17--29 updates in the
six rechecked cases. Its zero-page inputs include between-update sound/render
scratch that this gameplay mirror does not reconstruct. A game consumes many
random draws; a reproducible independent seed does not reproduce the ROM stream
or establish equal distributions.

`reference-rng` is a diagnostic-only Cargo feature, disabled by default. It must
not supply training randomness. The original conditional corpus and emulator
instrumentation remain local diagnostic artifacts and are not bundled here.

## Build and test

From the project root with the Rust MSVC toolchain:

```powershell
cargo test --manifest-path native\Cargo.toml
cargo test --manifest-path native\Cargo.toml --features reference-rng
cargo build --release --manifest-path native\Cargo.toml
```

The ignored external fixture-driver test requires local trace files; ordinary
unit tests are self-contained. ROMs, DLLs, model archives and runs are excluded
from Git. Supply your own ROM locally when running the Python ROM tests.

## Standalone training mode

`rust_train.py` deliberately uses ten primitive actions, 47 channels and a
two-frame stack. It does not load production fifteen-action checkpoints or
modify `train.py`. Each stage is initialized once with the user's local ROM;
rollout steps and automatic episode resets use only Rust.

Seeded mode uses SplitMix64-v1, taking the high byte of each successive 64-bit
output. Per-environment/episode seeds are the first eight SHA256 bytes
(little-endian) of `tank-ai-episode-v1:base_seed:stream:episode`. PPO supplies
`base_seed = training_seed + environment_index`, and `stream` is that index.
There is no shared RNG, oracle tape or random-state resynchronization. Native
reset restores the initial RAM and explicitly seeds the independent stream.

The seeded mode explicitly accepts enemy/spawn/item-drop/ice mechanics as
experimental. Unknown terrain, P2 and unsupported stage continuation retain
hard errors. Legacy mode retains its original restrictive parity flags.
The Python training binding rejects a `reference-rng` diagnostic DLL.

## Monitored learning study

The `study` command is separate from the short `train`/`compare` storage
acceptance commands below. It compares three fresh primitive-action policies:

| Model | Input history | Feature extractor parameters |
| --- | --- | ---: |
| `cnn` | 2 decision observations | 2,851,680 |
| `spatial` | 2 observations, 169 positioned spatial tokens | 199,264 |
| `history` | 16 raw observations, causal temporal attention | 222,656 |

The historical model has a finite window of about one second, not unbounded
recurrent memory. It encodes every raw frame during PPO replay, never reuses
stale embeddings after a parameter update, and cannot attend to a future frame.
Prefix padding is masked safely; episode resets clear history, while terminal
observations retain the preceding history for timeout value bootstrapping.
Its fixed 13-to-3 spatial average-pooling bins support deterministic CUDA
backpropagation; this is not a request to relax Torch determinism.

```powershell
uv run --no-sync python rust_train.py study `
  --output runs\rust-monitored-study `
  --models cnn,spatial,history --seeds 0,1 `
  --steps 262144 --checkpoint-every 16384 --eval-every 65536 `
  --envs 32 --n-steps 256 --batch-size 256 --epochs 4 `
  --codec cuda --device cuda --parallel 2 `
  --stages 1,2,3 --eval-stages 4,5 --eval-seeds 101,202 `
  --confirm-stages 6,7 --confirm-seeds 1001,2002 `
  --max-frames 12000 --max-wall-seconds 3600 --patience 3
```

This requests six trials with a total cap of **1,572,864 transitions**. Each
trial uses LR 0.00005, target KL 0.01, gamma 0.99, the same sparse reward, and fresh
weights/optimizer. No fifteen-action checkpoint is silently loaded.
At most two trials run together; elapsed time is shared-resource timing, not an
isolated model speed benchmark.

Checkpoints are saved only after optimizer updates, every 16,384 transitions.
Initial and every 65,536-step validation separately report:

- Rust full-stage outcomes under its independent seeded PRNG;
- original-ROM full-stage outcomes, with the full **12,000-frame** allowance;
- original-ROM shooting, dodging, countering, demolition, pickup and defense
  skill success. Controlled skills retain their 600-frame diagnostic cap and
  never count as full-stage wins.

All three evaluation surfaces accept exactly ten primitive actions and derive
their 2/16-frame history from the checkpoint's model contract. Evaluation restores
training RNG state and policy mode. Unavailable-fire and stationary-action
diagnostics are decision-boundary measurements, not claims about authoritative
per-frame shot events. The existing ROM observer caveats still apply.

`best` is selected by deterministic ROM wins, then mean ROM skill success.
Confirmation on stages 6/7 never selects checkpoints. Rust return or survival
alone cannot qualify a model as improving in the cartridge.

Stop conditions are the exact transition cap, one-hour whole-trial wall cap,
three completed validation checks without improvement, or an explicit runtime/
nonfinite failure. Incomplete evaluations remain incomplete and cannot be
presented as successes; no trial automatically extends its budget.

Each checkpoint has its own versioned manifest with model/history/action
contract, Python/native/library/ROM hashes, initial stage RAM hashes, seeds,
optimizer-update count and optimized steps. They are evaluation checkpoints,
not exact simulator/RNG resume snapshots. Existing performance manifests and
the production fifteen-action manifests are separate formats.

Monitoring uses immutable durable status records; a sharing-locked Windows
display pointer must not terminate otherwise healthy training. Read authoritative
status through `rust_learning.read_status`. All completed trial/checkpoint and
validation records are retained, including unsuccessful or stopped trials.

Sparse reward is +1 per kill, -3 per death, -20 for base loss, +5 for clear and
-0.0005 per logical frame. Episodes span multiple lives, ending at base loss,
final-life death, clear or timeout; a timeout is truncated, not a failure.
Original-ROM shot/kill event-observer limitations remain documented below.

Outputs go to a new directory: configuration, model, versioned/hash-bound
manifest and measured result. They record seeds, initial RAM hashes, native
source/library hashes, ROM identity and the action/observation/reward contract.
These are policy checkpoints, **not simulator save states or exact training
resume points**. Rebuilding or changing the stochastic backend needs a new run.

```powershell
# Small CPU smoke; steps must be a multiple of envs * n-steps.
uv run --no-sync python rust_train.py train --output runs\rust-cpu-smoke `
  --device cpu --codec native --envs 2 --n-steps 16 --batch-size 16 --steps 64

# Identical short configurations in fresh processes, different storage only.
uv run --no-sync python rust_train.py compare --output runs\rust-storage-check `
  --device cuda --codec cuda --envs 32 --n-steps 64 --batch-size 256 --steps 4096

# Separate original-ROM evaluation, both deterministic and sampled actions.
uv run --no-sync python rust_train.py eval `
  --model runs\rust-storage-check\cuda\model.zip `
  --output runs\rust-storage-check\rom-transfer.json --device cuda

uv run --no-sync python -m unittest test_rust_env test_rust_training `
  test_compact_rollout test_native_rollout -v
```

## Memory contract

The model-independent chronological stack buffer is reused from the performance
experiment; its current native codec and batch adapter are wired to this
repository's latest engine. No old `app/`, engine, fixed skill-only reset path or
copied mechanics loop is used. Fused native stepping delegates to the same
current `mirror_step` implementation as the unoptimized path.

`dense` is the SB3 baseline; `zlib` uses lossless compression; `native` uses
float32 bit runs with raw fallback; `cuda` expands run lengths and integer bit
patterns on GPU. All preserve signed zero and NaN payloads without arithmetic
conversion. Frame overlap deduplication is limited to adjacent steps in the same
environment. Only minibatches are decoded; there is no dense rollout-sized
observation allocation or persistent decoded epoch cache.

The default 2 GiB cap covers accounted persistent buffer arrays, payloads and
estimated Python overhead. It is **not** a process RSS, GPU-memory or transient
allocation cap. Comparison reports distinguish rollout storage, peak process
RSS and peak CUDA allocation. Public VecEnv outputs and terminal observations
are owned snapshots; native work buffers can be reused without overwriting
SB3's retained episode-start flags. Native low-level batch arrays are borrowed.

Short comparison rejects unequal final policy hashes; matching hashes establish
storage equivalence for the measured run, not higher win rates. Historical
performance-worktree speedups do not apply automatically to this version.

## Integration acceptance measurements

On the local RTX 5090, a repeated fresh-process comparison used 32 environments,
64 rollout steps, minibatches of 256, two epochs, 4,096 transitions per run,
stages 1/5/10/20/35, seed 37 and a 600-frame episode cap:

| Metric | Current engine + dense buffer | Current engine + CUDA bit runs |
|---|---:|---:|
| Accounted rollout storage | 496.58 MiB | 15.57 MiB |
| Peak process working set | 2.93 GiB | 2.01 GiB |
| Peak allocated CUDA memory | 347.40 MiB | 370.04 MiB |
| Training transitions/second | 1,507 | 2,146 |

Final policy parameter hashes were identical:
`15c36e3abb468246a682470b97e96f21976b21ba38df80748c59dc5ed198a2ac`.
This repeat measured about 31.9x smaller rollout storage and 1.42x throughput,
not a 31.9x reduction in total memory. The first short comparison, concurrent
with a CPU stage soak, measured 1.70x throughput; timing is workload-sensitive.
GPU allocation increased slightly. Both comparisons changed storage only;
neither is a new speed comparison against the ROM emulator.

Further acceptance:

- Final regression: 294 Python tests passed, including a generated latest-engine
  snapshot and the external-working-directory comparison regression. Native
  tests passed in default (78) and diagnostic (84) builds; the external trace
  fixture driver remains explicitly ignored in these self-contained suites.
- Latest seeded backend: all 35 stages completed episodes during 105,000 random
  masked decisions; no unsupported-feature errors or nonfinite observations.
  Completed episodes covered 395,071 frames. This is runtime coverage, not parity.
- CPU dense/zlib/native short training produced identical policy hashes, with
  episode resets and time-limit bootstrapping exercised.
- Float-bit, action-mask, return/advantage and CUDA/raw-fallback tests passed.
- Original-ROM transfer ran 20 episodes (five stages, two seeds, deterministic
  and sampled actions) at the same 600-frame cap: zero clears, 18 timeouts and
  two base losses. This validates the evaluation pipeline only.

Detailed reports and short-run checkpoints are local under
`runs/rust-latest-acceptance-cuda/` and
`runs/rust-latest-acceptance-cuda-repeat/`, excluded from Git. No long training
was started.

## Deferred work

- [ ] Reconstruct the between-update memory side effects consumed by ROM RNG,
  then rerun independent logical-boundary comparisons. A full CPU emulator is
  not yet proven necessary.
- [ ] Validate the stochastic distribution and original-ROM transfer of any
  independently seeded training backend. Conditional mechanics evidence alone
  does not validate either.
- [ ] Extend independent long-trajectory validation for ice, stage transitions
  and terminal-state continuation. Menus, audio and rendering are not goals of
  the gameplay training backend.
- [ ] Reconcile ROM event observers with instruction-level shot creation and
  enemies that activate and die within one update. Preserve existing production
  experiment contracts until deliberately versioned.
