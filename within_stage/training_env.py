"""TRAIN-only precision110 VecEnv; legal inputs, independent opening/noise RNG.

make_training_env(...) returns TrainingVecEnv (a normal VecFrameStack). Do NOT
stack or apply the victory reward again. current_kills() returns counters from
the just-completed transition, including terminal values across DummyVecEnv's
automatic resets; current_state_kills() returns each new/current episode's
counter. initial_remaining() is always20 after checked normal resets.
No counter is inferred from optional per-step infos, and no RAM initializes a
scene. Only the wrapped original opening env charges initialization/steps.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from numbers import Real
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from learning_lab.victory_objective import FiniteVictoryVecEnv
from within_stage.opening_pool import load_pool, make_opening_env
from within_stage.observation import PrecisionObservation, OBSERVATION_VERSION
from within_stage.execution_noise import ExecutionNoiseEnv

ROOT = Path(__file__).resolve().parents[1]
VERSION = "within-stage-training-env-v1"
GLOBAL_DEADLINE = datetime.fromisoformat("2026-10-01T08:30:25.131+08:00").timestamp()

FROZEN_SPLIT_SHA256 = "a94de5882737587e06930d6d782d512228f3248958df26f188fc2b494816eebe"


def load_train_context(split_path, config_indices=None):
    path = Path(split_path).resolve()
    if _sha(path) != FROZEN_SPLIT_SHA256:
        raise ValueError("Frozen TRAIN split identity mismatch")
    split = json.loads(path.read_text(encoding="utf-8"))
    if split.get("version") != "within-stage-family-split-v1" or split.get("stage") != 1 \
            or split.get("observation_version") != OBSERVATION_VERSION \
            or split.get("channels_per_frame") != 55 or split.get("post_start_frames") != 150 \
            or split.get("actions") != 10 or split.get("n_stack") != 2 \
            or split.get("max_frames") != 12000:
        raise ValueError("Frozen precision/opening protocol differs")
    pool_path = Path(split["pool_path"])
    if _sha(pool_path) != split["pool_sha256"]:
        raise ValueError("Frozen opening pool SHA256 mismatch")
    pool = load_pool(pool_path)
    train = split["train"]
    families, indices = train["family_ids"], train["all_config_indices"]
    if len(families) != 40 or len(set(families)) != 40 or len(set(indices)) != len(indices):
        raise ValueError("Malformed TRAIN families or aliases")
    by_config = {row["config_index"]: row for row in pool["specs"]}
    groups = {int(family): [] for family in families}
    for index in indices:
        if index not in by_config or by_config[index]["family_id"] not in groups:
            raise ValueError("TRAIN configuration has an invalid family")
        groups[by_config[index]["family_id"]].append(int(index))
    actual = {r["config_index"] for r in pool["specs"] if r["family_id"] in groups}
    if actual != set(indices) or any(not members for members in groups.values()):
        raise ValueError("TRAIN aliases were not kept together")
    if config_indices is not None:
        allowed = list(config_indices)
        if not allowed or any(type(i) is not int for i in allowed) \
                or len(set(allowed)) != len(allowed) or not set(allowed) <= set(indices):
            raise ValueError("config_indices must be a nonempty distinct subset of TRAIN aliases only")
        groups = {family: [i for i in members if i in allowed] for family, members in groups.items()}
        groups = {family: members for family, members in groups.items() if members}
    # Drop excluded partitions and diagnostic future data from the per-slot context.
    runtime_pool = {
        **{k: deepcopy(pool[k]) for k in ("format", "stage", "handoff", "frozen", "partition", "reference")},
        "specs": [{"config_index": i, "spec": deepcopy(by_config[i]["spec"])}
                  for members in groups.values() for i in members],
        "families": [{"family_id": f, "config_indices": members.copy()} for f, members in groups.items()],
    }
    return dict(pool=runtime_pool, groups=groups, all_train_config_indices=indices.copy(),
                selected_config_indices=[i for members in groups.values() for i in members],
                split_path=str(path), split_sha256=FROZEN_SPLIT_SHA256,
                pool_path=str(pool_path), pool_sha256=split["pool_sha256"])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    return {name: _sha(ROOT / name) for name in (
        "within_stage\\training_env.py", "within_stage\\opening_pool.py",
        "within_stage\\early_openings.py", "within_stage\\openings.py",
        "within_stage\\observation.py", "within_stage\\execution_noise.py",
        "learning_lab\\victory_objective.py", "learning_lab\\rewards.py",
        "battle_city_env.py", "game_state.py", "heuristic.py", "battle_city.nes")}


def _seed(seed, stream):
    payload = f"{VERSION}:opening:{seed}:{stream}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


class _BoundAccount:
    """Forwards accounting unchanged, while enforcing the factory's tighter time bound."""
    def __init__(self, account, deadline):
        object.__setattr__(self, "_account", account)
        object.__setattr__(self, "_deadline", deadline)

    def __getattr__(self, name):
        return getattr(self._account, name)

    def __setattr__(self, name, value):
        setattr(self._account, name, value)

    def check(self):
        if time.monotonic() >= self._deadline or time.time() >= GLOBAL_DEADLINE:
            raise TimeoutError("TRAIN environment absolute deadline reached")
        self._account.check()

    def before(self, kind):
        self.check()
        self._account.before(kind)

    def after(self, kind):
        self._account.after(kind)

    def guard(self, phase):
        self.check()


class _OpeningSwitch(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (47, 26, 26), np.float32)
    action_space = gym.spaces.Discrete(10)
    render_mode = None

    def __init__(self, context, account, stream, opening_seed, last_kills):
        super().__init__()
        self.context, self.account, self.stream, self.last_kills = context, account, stream, last_kills
        self.rng_seed = _seed(opening_seed, stream)
        self.opening_rng = np.random.default_rng(self.rng_seed)
        self.families = tuple(context["groups"])
        self.raw, self.needs_reset, self.episode_index, self.opening = None, True, -1, None

    @property
    def state(self):
        if self.raw is None:
            raise gym.error.ResetNeeded("No active TRAIN opening")
        return self.raw.state

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Opening override/reset options are forbidden; configure a TRAIN subset")
        self.account.check()
        # Gym/SB3 seed affects no opening or noise RNG; the fixed independent
        # per-slot stream remains continuous through automatic resets.
        super().reset(seed=seed)
        self.close()
        family = self.families[int(self.opening_rng.integers(len(self.families)))]
        aliases = self.context["groups"][family]
        config = aliases[int(self.opening_rng.integers(len(aliases)))]
        self.episode_index += 1
        self.opening = dict(
            partition="train", stage=1, config_index=config, family_id=family,
            opening_rng_seed=self.rng_seed, stream=self.stream, episode_index=self.episode_index,
            reset_input_seed=0, post_start_frames=150, curriculum_start=False)
        try:
            self.raw = make_opening_env(self.context["pool"], config, account=self.account, max_frames=12000)
            if self.raw.action_space.n != 10 or self.raw.observation_space.shape != (47, 26, 26):
                raise ValueError("TRAIN opening requires canonical47/Primitive10")
            observation, info = self.raw.reset(seed=0)
            state = self.raw.state
            if state.stage != 1 or state.remaining_enemies != 20 or state.kills != 0 \
                    or state.lives != 3 or state.player is None or state.base_destroyed or state.game_over:
                raise ValueError("TRAIN opening does not preserve normal stage1 initial conditions")
            self.needs_reset = False
            result = deepcopy(info)
            result["training_opening"] = deepcopy(self.opening)
            return observation.copy(), result
        except BaseException:
            self.close()
            raise

    def action_masks(self):
        self.account.check()
        if self.needs_reset:
            raise gym.error.ResetNeeded("Reset before TRAIN action masks")
        masks = np.asarray(self.raw.action_masks())
        if masks.shape != (10,) or masks.dtype != np.bool_ or not masks.any():
            raise ValueError("Expected nonempty Primitive10 boolean masks")
        return masks.copy()

    def step(self, action):
        self.account.check()
        if self.needs_reset:
            raise gym.error.ResetNeeded("Reset before another TRAIN step")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) \
                or not 0 <= action < 10 or not self.action_masks()[action]:
            raise ValueError("Expected legal primitive action")
        try:
            observation, reward, terminated, truncated, info = self.raw.step(int(action))
        except BaseException:
            self.needs_reset = True
            raise
        self.last_kills[self.stream] = int(self.raw.state.kills)
        result = deepcopy(info)
        result["training_opening"] = deepcopy(self.opening)
        result["training_state_kills"] = int(self.raw.state.kills)
        if terminated or truncated:
            self.needs_reset = True
            result["terminal_remaining_enemies"] = int(self.raw.state.remaining_enemies)
            result["terminal_base_destroyed"] = bool(self.raw.state.base_destroyed)
        return observation.copy(), reward, terminated, truncated, result

    def close(self):
        if self.raw is not None:
            self.raw.close()
        self.raw = None
        self.needs_reset = True


class TrainingVecEnv(VecFrameStack):
    def __init__(self, wrapped, slots, last_kills, provenance):
        super().__init__(wrapped, n_stack=2, channels_order="first")
        self._slots, self._last_kills, self._provenance = slots, last_kills, provenance
        self.objective_version = wrapped.objective_version

    def reset(self):
        self._last_kills.fill(0)
        return super().reset()

    @property
    def provenance(self):
        return deepcopy(self._provenance)

    def verify_provenance(self):
        p = self._provenance
        if _sha(p["split_path"]) != p["split_sha256"] or _sha(p["pool_path"]) != p["pool_sha256"] \
                or _source_hashes() != p["sources"]:
            raise ValueError("TRAIN environment source/split/pool provenance changed")
        return True

    def current_openings(self):
        return [deepcopy(slot.opening) for slot in self._slots]

    def current_kills(self):
        return self._last_kills.copy()

    def current_state_kills(self):
        return np.array([slot.state.kills if slot.raw is not None else 0 for slot in self._slots], np.int64)

    def initial_remaining(self):
        return np.full(len(self._slots), 20, np.int64)


def make_training_env(split_path, *, n_envs=4, noise_probability=0., noise_seed=0,
                      account, deadline, config_indices=None, opening_seed=0):
    """Return a TRAIN-only stacked110 VecEnv; no emulator initialized until reset.

    config_indices optionally restricts to a nonempty TRAIN-only subset. Sampling
    stays uniform over retained families, then uniform over retained aliases.
    deadline is absolute monotonic; account is the existing shared ProbeAccounting
    or compatible counter. Neither account nor ledger budgets are extended.
    current_kills()/initial_remaining() expose the already-wired reward providers.
    """
    if type(n_envs) is not int or not 1 <= n_envs <= 32:
        raise ValueError("n_envs must be in1..32")
    if isinstance(noise_probability, (bool, np.bool_)) or not isinstance(noise_probability, Real) \
            or not np.isfinite(noise_probability) or not 0 <= noise_probability <= 1:
        raise ValueError("noise_probability must be finite in[0,1]")
    for value, name in ((noise_seed, "noise_seed"), (opening_seed, "opening_seed")):
        if type(value) is not int or not 0 <= value < 2**64:
            raise ValueError(f"{name} must be uint64")
    if isinstance(deadline, (bool, np.bool_)) or not isinstance(deadline, Real) or not np.isfinite(deadline):
        raise ValueError("deadline must be a finite absolute monotonic time")
    if not all(callable(getattr(account, name, None)) for name in ("check", "before", "after")):
        raise ValueError("account must provide ProbeAccounting-compatible check/before/after")
    context = load_train_context(split_path, config_indices)
    bounded = _BoundAccount(account, float(deadline))
    counters = np.zeros(n_envs, np.int64)
    slots = [_OpeningSwitch(context, bounded, i, opening_seed, counters) for i in range(n_envs)]
    wrappers = []
    raw_vec = None
    try:
        for i, slot in enumerate(slots):
            precise = PrecisionObservation(slot)
            noisy = ExecutionNoiseEnv(precise, probability=float(noise_probability),
                                      noise_seed=noise_seed, stream_id=i,
                                      deadline=float(deadline), guard=bounded.guard)
            wrappers.append(Monitor(noisy, filename=None))
        raw_vec = DummyVecEnv([lambda env=env: env for env in wrappers])
        victory = FiniteVictoryVecEnv(
            raw_vec, np.full(n_envs, 20, np.int64), lambda: counters.copy(), gamma=1., shaping=False)
        provenance = dict(
            version=VERSION, split_path=context["split_path"], split_sha256=context["split_sha256"],
            pool_path=context["pool_path"], pool_sha256=context["pool_sha256"],
            sources=_source_hashes(), selected_partition="train",
            selected_family_ids=list(context["groups"]), selected_config_indices=context["selected_config_indices"],
            sampling="uniform TRAIN family, then uniform legal alias; independent per-slot PCG64",
            opening_seed=opening_seed, opening_stream_seeds=[slot.rng_seed for slot in slots],
            noise_probability=float(noise_probability), noise_seed=noise_seed,
            noise_stream_ids=list(range(n_envs)), rng_independence="opening, noise and policy/reset RNG streams separate",
            observation_version=OBSERVATION_VERSION, observation_shape=[110, 26, 26], actions=10,
            objective_version=victory.objective_version, gamma=1., shaping=False,
            wrapper_order=["TRAIN opening switch(Primitive10)", "PrecisionObservation(55)",
                           "ExecutionNoiseEnv", "Monitor", "DummyVecEnv",
                           "FiniteVictoryVecEnv(shaping=False,gamma=1)", "VecFrameStack(first,2)"],
            counter_source="canonical raw.state.kills captured after each step; terminal value survives automatic reset",
            accounting="Only original opening env calls existing account.before/after; wrappers do not charge again",
            deadline_monotonic=float(deadline), global_deadline_epoch=GLOBAL_DEADLINE,
            ram_initialization=False, final_or_development_accepted=False,
        )
        return TrainingVecEnv(victory, slots, counters, provenance)
    except BaseException:
        if raw_vec is not None:
            raw_vec.close()
        else:
            for env in wrappers:
                env.close()
            for slot in slots:
                slot.close()
        raise
