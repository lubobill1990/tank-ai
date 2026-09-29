"""Ten-primitive Rust/ROM transfer evaluation, independent of training returns.

Formal full games use 12,000 frames. Explicit smaller caps are smoke tests,
not checkpoint-selection evidence. Skills retain their original 600-frame cap
(or the explicitly requested smaller smoke cap) and never count as full wins.
"""
from contextlib import contextmanager
import random
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from battle_city_env import BattleCityEnv
import game_state as gs
from heuristic import redundant_boundary_action
from rust_env import BACKEND_VERSION, RNG_VERSION, SEED_VERSION, RustVectorEnv
from skills import SKILL_NAMES, SkillEnv


MODES = ("deterministic", "sampled")
FULL_GAME_FRAMES = 12000
SKILL_FRAMES = 600
COUNTERS = ("kills", "deaths", "shots", "pickups", "frames")


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("validation deadline exhausted")


@contextmanager
def _evaluation_state(model):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    # Do not initialize CUDA in a CPU-only evaluator or enqueue deferred seeds.
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    policy = getattr(model, "policy", None)
    training = getattr(policy, "training", None)
    try:
        if training is not None:
            policy.train(False)
        with torch.no_grad():
            yield
    finally:
        try:
            if training is not None:
                policy.train(training)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)


def _seed_episode(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def _primitive(action):
    value = np.asarray(action)
    if value.size != 1 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError("prediction must be one integer primitive action in 0..9")
    action = int(value.item())
    if not 0 <= action < 10:
        raise ValueError("prediction must be one integer primitive action in 0..9")
    return action


class PrimitiveActions(gym.Wrapper):
    """Trim original masks without changing the cartridge or enabling macros."""
    def __init__(self, env, *, deadline=None):
        super().__init__(env)
        self.action_space = spaces.Discrete(10)
        self.deadline = deadline
        self._nes_step = None
        if hasattr(env, "nes"):
            self._nes_step = env.nes.step

            def checked_step(*args, **kwargs):
                _check_deadline(self.deadline)
                return self._nes_step(*args, **kwargs)

            env.nes.step = checked_step

    @property
    def state(self):
        return self.env.state

    @property
    def max_frames(self):
        return self.env.max_frames

    def action_masks(self):
        return np.asarray(self.env.action_masks(), dtype=bool)[:10].copy()

    def reset(self, *, seed=None, options=None):
        _check_deadline(self.deadline)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        action = _primitive(action)
        _check_deadline(self.deadline)
        return self.env.step(action)

    def close(self):
        if self._nes_step is not None:
            self.env.nes.step = self._nes_step
        self.env.close()


def _initial_ram(stage, deadline):
    """Exactly stage_ram(stage, seed=0), with checks inside ROM boot/reset loops."""
    _check_deadline(deadline)
    env = PrimitiveActions(BattleCityEnv(stage=stage, reward_profile="sparse"),
                           deadline=deadline)
    try:
        env.reset(seed=0)
        for _ in range(2):
            env.env.nes.step(0)
        return env.env.nes.ram.copy()[None, :]
    finally:
        env.close()


class RustEpisode:
    """Expose terminal native state before VecEnv's mandatory automatic reset."""
    def __init__(self, initial, *, stage, max_frames, deadline=None):
        _check_deadline(deadline)
        self.stage, self.max_frames, self.deadline = stage, max_frames, deadline
        self.vector = RustVectorEnv(initial, max_frames=max_frames, seed=0)
        self.observation_space = self.vector.observation_space
        self.action_space = self.vector.action_space
        self._native_step = self.vector.batch.step
        self._native_reset = self.vector.batch.reset
        self.state = None

        def checked_step(actions):
            _check_deadline(self.deadline)
            result = self._native_step(actions)
            frame, kills, deaths, shots, pickups = map(int, result[3][0])
            self._stats = dict(frames=frame, kills=kills, deaths=deaths,
                               shots=shots, pickups=pickups)
            self.state = gs.parse(self.vector.batch.ram(0), frame=frame, kills=kills)
            return result

        def checked_reset(*args, **kwargs):
            _check_deadline(self.deadline)
            return self._native_reset(*args, **kwargs)

        self.vector.batch.step = checked_step
        self.vector.batch.reset = checked_reset

    def reset(self, *, seed=None):
        _check_deadline(self.deadline)
        self.vector.seed(seed)
        observation = self.vector.reset()[0]
        self.state = gs.parse(self.vector.batch.ram(0), frame=0, kills=0)
        self._stats = dict.fromkeys(COUNTERS, 0)
        return observation, {}

    def action_masks(self):
        return self.vector.batch.masks()[0].copy()

    def step(self, action):
        action = _primitive(action)
        _check_deadline(self.deadline)
        observation, rewards, dones, infos = self.vector.step(np.array([action]))
        done, info = bool(dones[0]), dict(infos[0])
        if done:
            observation = info.pop("terminal_observation")
        else:
            observation = observation[0]
            info.update(self._stats, outcome="", lives=self.state.lives)
        info.update(stage=self.stage, objective="full")
        truncated = bool(info.pop("TimeLimit.truncated", False))
        return observation, float(rewards[0]), done and not truncated, truncated, info

    def close(self):
        self.vector.batch.step = self._native_step
        self.vector.batch.reset = self._native_reset
        self.vector.close()


class EpisodePlayer:
    """VecFrameStack(first) history, without stacking an automatic reset frame."""
    def __init__(self, raw, *, model, nstack, deterministic, deadline=None):
        if type(nstack) is not int or nstack <= 0:
            raise ValueError("nstack must be positive")
        self.raw, self.model = raw, model
        self.deterministic, self.deadline = deterministic, deadline
        self.channels = raw.observation_space.shape[0]
        self.observation = np.zeros(
            (self.channels * nstack, *raw.observation_space.shape[1:]),
            dtype=raw.observation_space.dtype)
        self.done = True
        self.last_action = None

    def reset(self, seed):
        _check_deadline(self.deadline)
        observation, _ = self.raw.reset(seed=seed)
        self.observation.fill(0)
        self.observation[-self.channels:] = observation
        self.done = False
        self.last_action = None
        return self.observation

    def step(self):
        if self.done:
            raise RuntimeError("reset the episode before stepping")
        _check_deadline(self.deadline)
        masks = self.raw.action_masks()
        if masks.shape != (10,):
            raise ValueError("evaluation requires exactly ten primitive masks")
        _check_deadline(self.deadline)
        action, _ = self.model.predict(self.observation, deterministic=self.deterministic,
                                       action_masks=masks)
        self.last_action = _primitive(action)
        _check_deadline(self.deadline)
        observation, reward, terminated, truncated, info = self.raw.step(self.last_action)
        self.observation = np.roll(self.observation, -self.channels, axis=0)
        self.observation[-self.channels:] = observation
        self.done = bool(terminated or truncated)
        return reward, self.done, info


class _Diagnostics:
    """Decision-boundary samples, not inferred frame-level cartridge events."""
    def __init__(self):
        self.values = dict.fromkeys((
            "sample_count", "sampled_frames", "player_sample_count", "player_sample_frames",
            "stationary_samples", "stationary_sampled_frames", "max_stationary_sampled_frames",
            "stagnation_frames", "max_stagnation_frames", "outward_boundary_requests",
            "boundary_waste_requests", "boundary_waste_frames", "fire_requests",
            "requested_unavailable_fire", "requested_outward_fire",
            "ineffective_fire_requests",
        ), 0)
        self.values["action_counts"] = [0] * 10
        self.stationary_streak = 0
        self.last_kills = self.last_pickups = 0

    def update(self, before, after, action, info):
        elapsed = after.frame - before.frame
        if elapsed <= 0:
            raise RuntimeError("evaluation step did not advance its frame counter")
        values = self.values
        values["sample_count"] += 1
        values["sampled_frames"] += elapsed
        values["action_counts"][action] += 1
        player, next_player = before.player, after.player
        sampled = player is not None and next_player is not None
        moved = sampled and (player.x, player.y) != (next_player.x, next_player.y)
        stationary = sampled and not moved
        if sampled:
            values["player_sample_count"] += 1
            values["player_sample_frames"] += elapsed
        if stationary:
            values["stationary_samples"] += 1
            values["stationary_sampled_frames"] += elapsed
            self.stationary_streak += elapsed
        else:
            self.stationary_streak = 0
        values["max_stationary_sampled_frames"] = max(
            values["max_stationary_sampled_frames"], self.stationary_streak)
        progress = moved or info["kills"] > self.last_kills or info["pickups"] > self.last_pickups
        values["stagnation_frames"] = 0 if progress else values["stagnation_frames"] + elapsed
        values["max_stagnation_frames"] = max(values["max_stagnation_frames"],
                                             values["stagnation_frames"])
        self.last_kills, self.last_pickups = info["kills"], info["pickups"]
        boundary = redundant_boundary_action(before, action)
        values["outward_boundary_requests"] += int(boundary)
        if boundary and stationary and player.facing == next_player.facing:
            values["boundary_waste_requests"] += 1
            values["boundary_waste_frames"] += elapsed
        if action >= 5:
            unavailable = not before.fire_ready
            values["fire_requests"] += 1
            values["requested_unavailable_fire"] += int(unavailable)
            values["requested_outward_fire"] += int(boundary)
            values["ineffective_fire_requests"] += int(unavailable or boundary)

    def result(self):
        result = dict(self.values)
        result["boundary_waste_rate"] = _ratio(result["boundary_waste_frames"],
                                                result["sampled_frames"])
        result["stationary_sample_rate"] = _ratio(result["stationary_samples"],
                                                   result["player_sample_count"])
        result["ineffective_fire_request_rate"] = _ratio(result["ineffective_fire_requests"],
                                                          result["fire_requests"])
        return result


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def play_episode(raw, *, model, nstack, deterministic, seed, deadline=None, resource_check=None):
    """Run one real episode; no timeout/deadline can be converted into success."""
    _check_deadline(deadline)
    if resource_check is not None:
        resource_check()
    _seed_episode(seed)
    player = EpisodePlayer(raw, model=model, nstack=nstack, deterministic=deterministic,
                           deadline=deadline)
    player.reset(seed)
    diagnostics, total_reward = _Diagnostics(), 0.0
    for decision in range(1, raw.max_frames + 1):
        if resource_check is not None:
            resource_check()
        before = raw.state
        reward, done, info = player.step()
        _check_deadline(deadline)
        total_reward += float(reward)
        diagnostics.update(before, raw.state, player.last_action, info)
        if done:
            break
    else:
        raise RuntimeError("evaluation environment exceeded its frame budget")
    if not np.isfinite(total_reward):
        raise ValueError("evaluation produced nonfinite reward")
    fields = ("outcome", "objective", "stage", "lives", *COUNTERS, "episode_seed",
              "skill", "variant", "skill_success", "reason", "interception")
    episode = {key: info[key] for key in fields if key in info}
    if info["objective"] == "full" and info["outcome"] not in (
            "clear", "base_lost", "lives_exhausted", "timeout"):
        raise ValueError("full games require a full-game terminal outcome")
    episode.update(seed=seed, reward=total_reward, decisions=decision,
                   diagnostics=diagnostics.result())
    return episode


def _summarize(episodes, *, skill=False):
    summary = {"episodes": len(episodes)}
    if skill:
        successes = sum(bool(e["skill_success"]) for e in episodes)
        summary.update(successes=successes, success_rate=successes / len(episodes))
    else:
        wins = sum(e["outcome"] == "clear" for e in episodes)
        summary.update(wins=wins, win_rate=wins / len(episodes))
    for key in (*COUNTERS, "decisions", "reward"):
        summary[f"mean_{key}"] = sum(e[key] for e in episodes) / len(episodes)
    summary["outcomes"] = {outcome: sum(e["outcome"] == outcome for e in episodes)
                           for outcome in sorted({e["outcome"] for e in episodes})}
    totals = {name: sum(e["diagnostics"][name] for e in episodes) for name in (
        "sample_count", "sampled_frames", "player_sample_count", "player_sample_frames",
        "stationary_samples", "stationary_sampled_frames", "outward_boundary_requests",
        "boundary_waste_requests", "boundary_waste_frames", "fire_requests",
        "requested_unavailable_fire", "requested_outward_fire", "ineffective_fire_requests",
    )}
    totals["action_counts"] = [sum(e["diagnostics"]["action_counts"][i] for e in episodes)
                               for i in range(10)]
    totals["boundary_waste_rate"] = _ratio(totals["boundary_waste_frames"], totals["sampled_frames"])
    totals["stationary_sample_rate"] = _ratio(totals["stationary_samples"],
                                              totals["player_sample_count"])
    totals["ineffective_fire_request_rate"] = _ratio(totals["ineffective_fire_requests"],
                                                     totals["fire_requests"])
    for key in ("stagnation_frames", "max_stagnation_frames", "max_stationary_sampled_frames"):
        totals[f"mean_{key}"] = sum(e["diagnostics"][key] for e in episodes) / len(episodes)
    summary["diagnostics"] = totals
    return summary


def _mode_result(episodes, *, stages=None):
    if stages is not None:
        return {
            "episodes": episodes, "summary": _summarize(episodes),
            "per_stage": {str(stage): _summarize([e for e in episodes if e["stage"] == stage])
                          for stage in stages},
        }
    per_skill = {skill: _summarize([e for e in episodes if e["skill"] == skill], skill=True)
                 for skill in SKILL_NAMES}
    summary = _summarize(episodes, skill=True)
    summary["mean_success_rate"] = sum(s["success_rate"] for s in per_skill.values()) / len(per_skill)
    return {"episodes": episodes, "per_skill": per_skill, "summary": summary}


def _validate_inputs(model, model_name, stages, seeds, max_frames, deadline):
    from rust_models import model_spec
    spec = model_spec(model_name)
    for values, name, upper, lower in ((stages, "stages", 36, 1), (seeds, "seeds", 2**32, 0)):
        if not values or any(type(value) is not int or not lower <= value < upper for value in values) \
                or len(set(values)) != len(values):
            raise ValueError(f"{name} must contain distinct valid integers")
    if type(max_frames) is not int or not 1 <= max_frames <= FULL_GAME_FRAMES:
        raise ValueError("max_frames must be an integer in 1..12000")
    if deadline is not None and (isinstance(deadline, bool) or not np.isfinite(deadline)):
        raise ValueError("deadline must be finite absolute monotonic seconds")
    if not callable(getattr(model, "predict", None)):
        raise ValueError("model must supply masked primitive predictions")
    expected = (47 * spec["n_stack"], 26, 26)
    observation_space = getattr(model, "observation_space", None)
    action_space = getattr(model, "action_space", None)
    # Lightweight prediction proxies need not expose spaces; actual checkpoints do.
    if observation_space is not None and observation_space.shape != expected:
        raise ValueError(f"checkpoint observation shape must be {expected}")
    if action_space is not None and (
            not isinstance(action_space, spaces.Discrete) or action_space.n != 10
            or action_space.start != 0):
        raise ValueError("checkpoint action space must be Discrete(10), starting at zero")
    return spec


def evaluate_suite(model, *, model_name, stages=(4, 5), seeds=(101, 202),
                   max_frames=12000, deadline=None, resource_check=None):
    """Evaluate both modes on native full games, independent ROM games and skills.

    ``deadline`` is an absolute ``time.monotonic()`` time. Partial evaluations
    raise instead of returning fabricated/partial evidence. The caller's Python,
    NumPy, torch CPU/initialized CUDA RNGs and policy training mode are restored.
    """
    stages, seeds = tuple(stages), tuple(seeds)
    with _evaluation_state(model):
        spec = _validate_inputs(model, model_name, stages, seeds, max_frames, deadline)
        _check_deadline(deadline)
        if resource_check is not None:
            resource_check()
        nstack = spec["n_stack"]
        raw_results = {backend: {mode: [] for mode in MODES} for backend in ("rust", "rom", "skills")}
        for stage in stages:
            initial = _initial_ram(stage, deadline)
            rust = RustEpisode(initial, stage=stage, max_frames=max_frames, deadline=deadline)
            try:
                for mode in MODES:
                    for seed in seeds:
                        raw_results["rust"][mode].append(play_episode(
                            rust, model=model, nstack=nstack, deterministic=mode == "deterministic",
                            seed=seed, deadline=deadline, resource_check=resource_check))
            finally:
                rust.close()
            _check_deadline(deadline)
            rom = PrimitiveActions(BattleCityEnv(stage=stage, objective="full",
                                                max_frames=max_frames, reward_profile="sparse"),
                                   deadline=deadline)
            try:
                for mode in MODES:
                    for seed in seeds:
                        raw_results["rom"][mode].append(play_episode(
                            rom, model=model, nstack=nstack, deterministic=mode == "deterministic",
                            seed=seed, deadline=deadline, resource_check=resource_check))
            finally:
                rom.close()
        skill_frames = min(SKILL_FRAMES, max_frames)
        for skill in SKILL_NAMES:
            for variant in (0, 1):
                _check_deadline(deadline)
                raw = PrimitiveActions(SkillEnv(skill, variant=variant, max_frames=skill_frames),
                                       deadline=deadline)
                try:
                    for mode in MODES:
                        for seed in seeds:
                            raw_results["skills"][mode].append(play_episode(
                                raw, model=model, nstack=nstack, deterministic=mode == "deterministic",
                                seed=seed, deadline=deadline, resource_check=resource_check))
                finally:
                    raw.close()
        result = {
            backend: {
                "objective": "skill" if backend == "skills" else "full",
                "max_frames": skill_frames if backend == "skills" else max_frames,
                "modes": {mode: _mode_result(raw_results[backend][mode],
                                             stages=None if backend == "skills" else stages)
                          for mode in MODES},
            }
            for backend in ("rust", "rom", "skills")
        }
        result["metadata"] = {
            "schema_version": 1, "model_name": model_name, "model_spec": spec,
            "n_stack": nstack, "observation_shape": [47 * nstack, 26, 26], "action_count": 10,
            "stages": list(stages), "seeds": list(seeds), "skills": list(SKILL_NAMES),
            "variants": [0, 1], "max_frames": max_frames, "skill_max_frames": skill_frames,
            "formal_max_frames": FULL_GAME_FRAMES, "smoke_only": max_frames < FULL_GAME_FRAMES,
            "rom_parity": False, "rust_backend": BACKEND_VERSION,
            "rust_rng": RNG_VERSION, "rust_seed_version": SEED_VERSION, "stage_ram_seed": 0,
            "rng_note": (
                "Independent native PRNG and original cartridge stream; identical seed labels "
                "do not imply trajectory parity. Rust stage RAM uses fixed seed 0 plus two idle "
                "frames, then episode_seed(seed, stream=0, episode=0). ROM seeds choose reset "
                "timing phases. Policy sampling is reseeded per episode, one decision at a time; "
                "its sampling unit differs from batched training inference."
            ),
            "counter_sources": {
                "rust": "Authoritative native cumulative frames/kills/deaths/shots/pickups.",
                "rom": (
                    "Canonical original-ROM StateTracker counters; shots observe sound pulses. "
                    "Known sound-pulse and same-frame-kill omissions are observer limitations, "
                    "not evidence of backend differences."
                ),
                "skills": "Actual SkillEnv primitive success predicates, never full-stage wins.",
            },
            "diagnostics_note": (
                "One pre/post decision-boundary sample per action; sample_count=decisions and "
                "sampled_frames=sum(actual elapsed frames). Stationary rates use paired live-player "
                "samples, boundary waste rates use elapsed frames, fire-request rates use selected "
                "fire actions. Boundary waste is unchanged sampled pose after a provably redundant "
                "outward request; decision sampling cannot observe transient within-step motion. "
                "Stagnation means no sampled movement, counted kill or pickup. Ineffective fire "
                "requests are pre-state unavailable or already-outward requests, not actual shots "
                "or confirmed misses; readiness can change inside a decision."
            ),
            "base_collision_note": (
                "The experimental side-model has changed self/base collision behavior; no shared "
                "self/base-hit semantics or causal explanation of outcome differences is assumed."
            ),
        }
        _check_deadline(deadline)
        return result
