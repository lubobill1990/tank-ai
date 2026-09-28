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
