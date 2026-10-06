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

## Cartridge-formula training randomness (cartridge-rng-v1)

`--rng-backend cartridge` replaces the independent SplitMix64 stream with the
cartridge's own `$D44D` arithmetic running over the mirror's own zero page:
`7 * previous + coarse clock + RAM[walking cursor]` with the ADC carry. This
restores three structural properties of the console stream that SplitMix64
removes: the weak-LCG low-bit structure the game actually consumes (`&1`,
`&3`, `&15`, `&31`), correlation between random outputs and live gameplay
bytes the cursor walks over, and the feedback loop where policy behavior
changes random consumption and therefore the stream.

The episode seed only sets phase, mimicking the console's sole entropy source
(when START was pressed): a cartridge-rng-v1 expansion takes three
SplitMix64-v1 outputs for `$0F`, `$10` and the scratch-replay start row. Reset
and reseed restore the initial RAM and re-derive the phase; trajectories stay
exactly reproducible per episode seed.

`scratch_capture.py` records, from real local-ROM play, the zero-page bytes
that change during gameplay but are never read or written by the mirror
(observed: `$04`, `$0E`, `$47`, `$48` and the `$F0-$FE` audio-engine block).
`--scratch-trace` replays those rows once per logical frame from the
seed-derived phase, so the walking cursor reads moving sound/NMI scratch
instead of frozen snapshot bytes. Addresses that collide with mirrored
gameplay state are rejected by both the Python loader and the native build.
The trace is ROM-derived and stays local like the ROM itself; its SHA256 is
recorded in the training manifest.

Boundaries: rows are end-of-frame boundary captures, not
instruction-interleaved audio/NMI writes; sound effects triggered by gameplay
events are not synchronized to this run's events; the resulting distribution
is closer in structure to the cartridge stream but remains unvalidated against
it, and none of this is NES RNG equivalence or independent ROM parity.
After the ablation below, bare `cartridge` (no scratch replay) is the
`train`/`compare` default; `--rng-backend splitmix` restores the old stream.
The frozen `study` contract still constructs its environments explicitly on
SplitMix64 and is unchanged.

### RNG backend ablation (2026-10-04)

`rng_compare.py` trained one Grid26CNN MaskablePPO policy per (backend, seed)
under identical budgets — 1,048,576 transitions, stages 1-3, 12,000-frame
episodes, LR 2.5e-4, rollout 8,192, minibatch 256, four epochs — and evaluated
each checkpoint on original-ROM transfer (12 episodes: three stages, two seeds,
deterministic and sampled) plus 12 Rust episodes under each backend's
randomness. A fourth arm trained directly on the original ROM via
`rom_train.py` (same budget and update shape; 16 subprocess environments,
host-limited to all-but-two cores at below-normal priority).

Original-ROM transfer, averaged per arm (reward / base losses out of 12):

| Trained on | Mean reward | Base lost | Seeds |
|---|---:|---:|---|
| cartridge (bare formula) | **-14.7** | **8.0/12** | 37, 1037 |
| direct original-ROM | -14.9 | 8.0/12 | 37 only |
| cartridge + scratch replay | -15.9 | 9.5/12 | 37, 1037 |
| splitmix | -17.5 | 10.5/12 | 37, 1037 |

The direct-ROM policy, cross-evaluated on the three Rust surfaces, scored
-11.6 on the cartridge surface against -17.4/-17.7 on splitmix/scratch —
behavior learned on the real distribution survives best under the cartridge
formula, independently supporting it as the closest simulated stream.

Caveats recorded with the artifacts (`runs/20261003-rng-compare/`,
`runs/20261004-rng-compare-rom/`, local): two seeds per arm (one for the ROM
arm; its second seed was stopped repeatedly by host memory protection), seed
variance comparable to arm gaps, zero clears everywhere (the known CNN
representation plateau), shared-host timing. Directional evidence, not a
significance claim. Scratch replay v1 showed no ROM-transfer gain — its
unsynchronized sound-effect bytes are suspected — so it stays optional and
off by default.

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
nonfinite failure. Host resource pressure is also a stop condition, not evidence
that a policy failed to learn. Incomplete evaluations remain incomplete and cannot be
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

### Host resource protection

The monitored `study`/`learn` path records `resources.jsonl` in the study and each
trial directory. The current collector uses Windows commit counters, process
private/working-set memory, and CUDA allocated/reserved/peak allocated memory
after CUDA is initialized. It does not initialize CUDA merely to collect a
sample. Unsupported hosts or counter errors fail explicitly; the legacy
`train`/`compare`/`eval` commands are unchanged.

The runtime keeps an **8 GiB system commit headroom** threshold. Admission adds
an **8 GiB startup allowance per concurrent worker slot**: 16 GiB for one slot,
24 GiB when admitting a second worker. This conservatively includes workers that
have started but have not yet allocated their training buffers. Headroom means
Windows commit limit minus committed bytes, not free VRAM or free physical RAM.
These are heuristic safety reserves, not OS-enforced reservations or a promise
against concurrent allocations by other applications.

Sampling is throttled to once per second during rollout/evaluation decisions and
the supervisor loop, and forced before admission, optimizer updates and
checkpoints. Pressure stops new work without automatic retry, marks the trial
`incomplete` with `stop_reason=resource_pressure`, and keeps its last optimized
checkpoint. It does not save partially updated weights as a final model. A
supervisor pressure/error abort terminates only its owned child process trees;
their last observed exposure is not claimed to be exact. No system/pagefile
settings or unrelated processes are changed.

Unexpected exceptions retain full tracebacks. Trial `failure.json` and
`supervisor-failure.json` are small failure records independent of the normal
large reports. Status/result JSON and report bodies are streamed to avoid
whole-document string/byte copies. Immutable status publication and sharing-lock
handling remain unchanged. Telemetry is flushed for live inspection; completed
failure/result records are fsynced. An exhausted OS or failed disk can still
prevent a write, in which case errors propagate rather than claiming success.

The first six-trial study stopped abnormally and is preserved, not overwritten.
A separate six-update history diagnostic (49,152 transitions, with two explicitly
short smoke evaluations) showed private memory stabilizing around 7.46 GiB after
warm-up; it did not reproduce continuous growth. That does not establish the
cause of the earlier allocation failures or validate long-run stability. Changes
to monitoring still change the frozen source signature; old checkpoints are not
silently accepted as exact continuation under new code.

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

### Isolated victory research

`learning_lab/` contains explicit experimental reward, demonstration and
imitation pipelines. It does not replace the production environment or the
frozen `rust_train.py study` contract. In particular:

- `RewardProfileVecEnv(..., profile="loss_balanced_v1")` subtracts an additional
  20 only on `lives_exhausted`, and fixes the wrapped episode-return accounting.
  Baseline physics, all other reward terms and timeout truncation are unchanged.
- The separate `FiniteVictoryVecEnv` experiment requires **gamma=1**. Its
  potential-shaped rewards telescope to +1 for a clear or -1 for any failure,
  including timeout. The finite frame limit is part of this task, so timeouts
  are terminal for learning rather than value-bootstrapped truncations. Kill
  progress provides intermediate credit but cannot improve a losing episode's
  total return. This intentionally different objective is not a silent change
  to the production sparse reward.
- `ResearchBudget` uses a durable SQLite ledger. Work reserves its maximum
  interactions before starting; crashed/unsettled work remains charged at that
  upper bound. Evaluation and teacher decisions count too. Repeated offline
  supervised samples are reported separately, not as new game interactions.
- `DemonstrationDataset` stores bit-exact compact 47-channel frames and rebuilds
  two-frame history with an all-zero episode prefix. Labels, masks, executed
  actions and intervention provenance are distinct. DAgger collection clears
  are **not** learned-model victories.
- Imitation, class-balanced, recurrent and egocentric candidates have separate
  loaders and source/data-bound manifests. Never use the old fifteen-action or
  Rust-study loader as an interchange shortcut. The recurrent wrapper receives
  two stacked frames for compatibility, consumes current47, and resets memory
  only on the zero previous-frame episode prefix (not on a life loss).
- `rom_player.play_development_episode` uses an ordinary full-game ROM reset,
  ten primitives and optional streaming MP4 capture. It records the initial RAM
  hash, action sequence, checkpoint/policy hashes and terminal remaining-enemy/
  base state. It refuses sealed maps and opening phases. Probability-only
  inference calibration is labeled separately from ordinary policy prediction.

The authorized research tranche is local under `runs/20260930-victory-lab/`.
Its immutable protocol/opening-pool files and budget ledger define the
8-hour/4,000,000-interaction ceiling. Neural jobs run serially. Development
opening phases are 11/20/22/23; twenty other phases are held out. Stages 8/18/28
remain reserved for final generalization checks, not training or selection.
All actual offline-training invocations also receive a deadline derived from
the shared remaining wall budget.

A first CNN clear was recorded on stage1/seed1 after deliberately fitting a
single winning trajectory. The same checkpoint failed the other three
development openings. This is a **memorization/capacity control**, not stable
single-stage performance or generalization. Its duplicate-trajectory
validation accuracy must not be interpreted as independent test accuracy.

Backplay start pools contain expert intermediate RAM only for curriculum
initialization. Remaining-enemy count20 does not imply a normal starting state:
the teacher may already have moved or modified terrain. `full_start=True`
separately identifies the ordinary native initialization reference. Midgame
starts, NOOP probes and native curriculum clears never count as original-ROM
full-stage wins; no RAM corrections are applied during stepping.

### Within-stage robustness tranche

`within_stage/` is a separately authorized twelve-hour experiment, not an
extension or reinterpretation of the closed victory study. Its new ledger,
authorization and frozen split live under `runs/20260930-within-stage/`.
Previously revealed opening phases are not claimed as new unknown tests.

The new normal-opening protocol reconstructs stage1 using legal title/menu
input timing and a fixed 150-frame post-START handoff. Geometry is still
pristine, the player is live, the base is intact, and all twenty enemies
remain. No recorded RAM is injected to initialize these environments.
Duplicate parameter settings are grouped by observed future gameplay:
112 valid configurations produced 72 distinct 128-NOOP gameplay families.
The frozen split assigns 40 families to training, 12 to development and 20
to final evaluation; every alias of a family stays in the same partition.
These are distinct measured families, not a claim of statistically IID games.

An information audit found two firing lanes whose legacy47 observations and
masks were bit-identical but whose actual ROM hit outcomes differed by a
two-pixel enemy displacement. `PrecisionObservation` therefore appends eight
current-position offset planes (enemy, player projectile, enemy projectile and
item X/Y), producing 55 planes per frame and 110 for a two-frame stack.
Legacy47 planes are preserved exactly. Overlapping entities use explicit mean
offsets; the representation does not claim complete slot-level observability.

Precision checkpoints require their own loader. The old94-to-new110 migration
maps the two chronological frames separately, initializes added input weights
to zero, copies the actor weights, creates an independent critic encoder, and
starts a fresh Adam optimizer. It is weights-only migration, not exact resume.
Offsets-off experiments zero only the appended blocks during both training and
evaluation and record that required adapter in the checkpoint.

Training actuator noise leaves the policy's intended action/log-probability
unchanged while the environment occasionally executes a different legal
action. Intended, selected and executed actions are logged independently.
Recovery stress uses one predeclared movement-only intervention; cases in
which no intervention actually happened cannot count as recovery successes.
Immediate terminal failures are retained. Ordinary model clears, perturbed
recovery clears and mixed teacher/student collection clears are separate
metrics.

Precision demonstration datasets store lossless55 frames and reconstruct
zero-padded110 history. RAM/frame audit records are separate from model inputs.
Collectors and learners authenticate episode family/configuration membership
against the frozen TRAIN partition. On-policy training samples TRAIN families
uniformly, then parameter aliases uniformly, and charges initialization work
as well as agent decisions. Final-family testing remains a separate one-shot
operation after model selection, not part of the training loop.

- [ ] Reconstruct the between-update memory side effects consumed by ROM RNG,
  then rerun independent logical-boundary comparisons. A full CPU emulator is
  not yet proven necessary.
- [ ] Validate the stochastic distribution and original-ROM transfer of any
  independently seeded training backend, including cartridge-rng-v1 with and
  without scratch replay against SplitMix64-v1. Conditional mechanics evidence
  alone does not validate either; the structural argument for the cartridge
  formula is not a measured transfer improvement.
- [ ] Extend independent long-trajectory validation for ice, stage transitions
  and terminal-state continuation. Menus, audio and rendering are not goals of
  the gameplay training backend.
- [ ] Reconcile ROM event observers with instruction-level shot creation and
  enemies that activate and die within one update. Preserve existing production
  experiment contracts until deliberately versioned.
