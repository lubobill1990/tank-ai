"""Controlled real-ROM skill diagnostics, never full-stage win evaluations.

Only this benchmark writes scenario RAM. The cartridge still performs movement,
shooting, collisions, damage, pickups and timers; canonical StateTracker snapshots
and the normal 47-channel/15-action environment remain the sole state interface.
Variants change geometry. Seeds select reproducible cartridge timing phases.
"""
from typing import Any, Iterable

from battle_city_env import BattleCityEnv
import game_state as gs

SKILL_NAMES = ("shoot", "dodge", "counter", "demolish", "pickup", "defend")
DODGE_SURVIVAL_FRAMES = 120


class SkillEnv(BattleCityEnv):
    """A bounded, RAM-initialized cartridge scene, not a training curriculum.

    Shoot freezes one enemy. Defend freezes one enemy *after it has fired*:
    its live baseward projectile must also be dealt with before success.
    Freeze is the ROM's ordinary timer, visible in the canonical observation.
    Dodge/counter use its valid enemy-projectile slot after the owner has gone.
    No RAM is edited once reset returns.
    """

    def __init__(self, skill: str, variant: int = 0, max_frames: int = 600):
        if skill not in SKILL_NAMES:
            raise ValueError(f"skill must be one of {SKILL_NAMES}")
        if type(variant) is not int or variant not in (0, 1):
            raise ValueError("variant must be 0 or 1")
        if type(max_frames) is not int or max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")
        self.skill, self.variant = skill, variant
        self._skill_success = False
        self._skill_reason = ""
        self._interception = None
        super().__init__(stage=1, objective="full", max_frames=max_frames)

    def _setup(self) -> None:
        ram = self.nes.ram
        grid = ram[0x400:0x800].reshape(32, 32)[2:28, 2:28]
        grid[:] = 0
        grid[24:26, 12:14] = ((200, 201), (202, 203))
        ram[0xa1:0xa8] = 0
        ram[0xa8:0xb0] = 0
        ram[0xcc:0xe0] = 0
        ram[0x73:0x77] = 0
        ram[0x15:0x1c] = 0
        ram[0x7f], ram[0x80], ram[0x51], ram[0x68] = 0, 1, 3, 0x80
        ram[0x86:0x8b] = 0
        ram[0x62] = ram[0x100] = ram[0x101] = ram[0x30f] = 0
        self._gate_cells = ()
        variant = self.variant
        if self.skill == "shoot":
            player = (88, 160, gs.UP) if variant == 0 else (64, 112, gs.RIGHT)
            enemy = (88, 80, gs.DOWN) if variant == 0 else (160, 112, gs.LEFT)
            self._enemy(*enemy)
            ram[0x100] = 10
        elif self.skill in ("dodge", "counter"):
            player = (88, 144, gs.UP) if variant == 0 else (144, 88, gs.LEFT)
            offset = 8 if self.skill == "dodge" else 0
            bullet = (88 + offset, 64, gs.DOWN) if variant == 0 else (64, 88 + offset, gs.RIGHT)
            self._enemy_bullet(*bullet)
        elif self.skill == "demolish":
            if variant == 0:
                player = (88, 120, gs.RIGHT)
                grid[:, 10] = 16
                grid[12, 10], grid[13, 10] = 8, 2
                self._gate_cells = ((12, 10), (13, 10))
                ram[0x86], ram[0x87] = 144, 120
            else:
                player = (120, 88, gs.DOWN)
                grid[10, :] = 16
                grid[10, 12], grid[10, 13] = 8, 4
                self._gate_cells = ((10, 12), (10, 13))
                ram[0x86], ram[0x87] = 120, 144
            ram[0x88] = 3
        elif self.skill == "pickup":
            player = (72, 72, gs.RIGHT if variant == 0 else gs.DOWN)
            ram[0x86], ram[0x87] = (184, 72) if variant == 0 else (72, 184)
            ram[0x88] = 3
        else:
            player = (120, 192 if variant == 0 else 176, gs.UP)
            self._enemy(120, 104 if variant == 0 else 80, gs.DOWN)
            self._enemy_bullet(120, 144 if variant == 0 else 112, gs.DOWN)
            ram[0x100] = 10
        x, y, facing = player
        ram[0x90], ram[0x98], ram[0xa0] = x, y, 0x80 | gs.ROM_FACING.index(facing)

    def _enemy(self, x: int, y: int, facing: int) -> None:
        ram = self.nes.ram
        ram[0x92], ram[0x9a] = x, y
        ram[0xa2], ram[0xaa] = 0xa0 | gs.ROM_FACING.index(facing), 0x80

    def _enemy_bullet(self, x: int, y: int, facing: int) -> None:
        ram = self.nes.ram
        ram[0xba], ram[0xc4] = x, y
        ram[0xce], ram[0xd8] = 0x40 | gs.ROM_FACING.index(facing), 0

    def reset(self, *, seed=None, options=None):
        self._skill_success, self._skill_reason, self._interception = False, "", None
        super().reset(seed=seed, options=options)
        self._setup()
        self._tracker.reset(self.nes.ram)
        initial = self.state
        # Flush the ROM's input/render pipeline, and verify actual transitions
        # before establishing frame zero. No synthetic transition is substituted.
        for _ in range(2):
            self.nes.step(0)
            state, events = self._tracker.advance(self.nes.ram)
            if events.kills or events.deaths or events.shots or events.pickups:
                raise RuntimeError("skill setup caused an unexpected cartridge event")
        self._validate_setup(initial, state)
        self._tracker.reset(self.nes.ram)
        self._bot.reset()
        self._item_opportunities = int(self.state.item is not None)
        self._previous_skill_state = self.state
        self._previous_shots = 0
        self._initial_player = self.state.player
        self._target = next((b for b in self.state.bullets if b.slot == 2), None)
        self._target_passed_player = False
        return self._observation(), self._info("", {})

    def _validate_setup(self, before: gs.GameState, after: gs.GameState) -> None:
        valid = (after.player is not None and before.player == after.player
                 and not after.shield and not after.base_destroyed
                 and after.stage == 1 and after.spawn_left == 0
                 and after.remaining_enemies == 1 and after.kills == 0)
        count = 1 if self.skill in ("shoot", "defend") else 0
        valid = valid and len(after.enemies) == count and after.enemies == before.enemies
        if self.skill in ("dodge", "counter", "defend"):
            valid = valid and len(before.bullets) == len(after.bullets) == 1
            if valid:
                first, last = before.bullets[0], after.bullets[0]
                valid = (first.slot == last.slot == 2 and first.generation == last.generation
                         and (last.x - first.x, last.y - first.y) == (first.vx * 2, first.vy * 2))
        else:
            valid = valid and not after.bullets
        if self.skill in ("pickup", "demolish"):
            valid = valid and after.item == before.item and after.item is not None
        if not valid:
            raise RuntimeError(f"cartridge rejected {self.skill} variant {self.variant} setup")

    def _detect_interception(self, before: gs.GameState, after: gs.GameState) -> None:
        target = self._target
        if target is None or self._interception is not None:
            return
        enemy = next((b for b in before.bullets if b.generation == target.generation), None)
        if enemy is None or enemy.owner < 2 \
                or any(b.generation == target.generation for b in after.bullets):
            return
        ram = self.nes.ram
        if int(ram[0xcc + enemy.slot]) != 0:
            return
        surviving = {b.generation for b in after.bullets}
        for player in before.bullets:
            if player.owner != 0 or player.generation in surviving:
                continue
            if int(ram[0xcc + player.slot]) != 0:
                continue
            px, py = player.x + player.vx, player.y + player.vy
            ex, ey = enemy.x + enemy.vx, enemy.y + enemy.vy
            interior = all(gs.ARENA <= coordinate < gs.ARENA + gs.GRID * gs.CELL
                           for coordinate in (px, py, ex, ey))
            if interior and abs(px - ex) <= 4 and abs(py - ey) <= 4:
                # ROM pairwise cancellation zeros BOTH live slots directly.
                # It also cancels perpendicular shots. Wall/edge/tank impacts
                # instead enter the 0x33 explosion state.
                self._interception = {
                    "frame": after.frame,
                    "enemy_slot": enemy.slot, "enemy_generation": enemy.generation,
                    "player_slot": player.slot, "player_generation": player.generation,
                    "enemy_status_after": int(ram[0xcc + enemy.slot]),
                    "player_status_after": int(ram[0xcc + player.slot]),
                    "enemy_before": [enemy.x, enemy.y],
                    "player_before": [player.x, player.y],
                    "same_frame_shot": False,
                }
                return
        # A muzzle-distance counter can create and cancel the shot between
        # snapshots. Require the real shot pulse AND the retained ROM bullet
        # coordinates AND direct-zero cancellation, not just a vanished target.
        shooter = after.player
        if self._shots > self._previous_shots and before.fire_ready and shooter is not None \
                and not any(b.owner == 0 for b in before.bullets) and int(ram[0xcc]) == 0:
            px, py = int(ram[0xb8]), int(ram[0xc2])
            ex, ey = int(ram[0xb8 + enemy.slot]), int(ram[0xc2 + enemy.slot])
            interior = all(gs.ARENA <= coordinate < gs.ARENA + gs.GRID * gs.CELL
                           for coordinate in (px, py, ex, ey))
            if interior and abs(px - ex) <= 4 and abs(py - ey) <= 4:
                self._interception = {
                    "frame": after.frame,
                    "enemy_slot": enemy.slot, "enemy_generation": enemy.generation,
                    "player_slot": 0, "player_generation": None,
                    "enemy_status_after": int(ram[0xcc + enemy.slot]),
                    "player_status_after": int(ram[0xcc]),
                    "enemy_before": [enemy.x, enemy.y], "player_before": None,
                    "enemy_after": [ex, ey], "player_after": [px, py],
                    "same_frame_shot": True,
                }

    def _outcome(self, deaths):
        state, before = self.state, self._previous_skill_state
        self._previous_skill_state = state
        self._detect_interception(before, state)
        self._previous_shots = self._shots
        if state.base_destroyed:
            self._skill_reason = "base_lost"
            return "base_lost"
        if self._deaths or deaths or state.player is None or state.lives == 0:
            self._skill_reason = "player_died"
            return "lives_exhausted"
        if state.stage != 1:
            self._skill_reason = "stage_changed"
            return "failure"
        success = False
        if self.skill == "shoot":
            success, reason = state.kills == 1 and self._shots > 0, "target_destroyed"
        elif self.skill == "dodge":
            p = state.player
            bullet = next((b for b in state.bullets if b.generation == self._target.generation), None)
            previous = next((b for b in before.bullets if b.generation == self._target.generation), None)
            if bullet is not None and previous is not None and before.player is not None:
                lane = abs(bullet.x - p.x) if bullet.vy else abs(bullet.y - p.y)
                moved = p.x != self._initial_player.x if bullet.vy else p.y != self._initial_player.y
                old_distance = ((previous.x - before.player.x) * previous.vx
                                + (previous.y - before.player.y) * previous.vy)
                new_distance = (bullet.x - p.x) * bullet.vx + (bullet.y - p.y) * bullet.vy
                advancing = ((bullet.x - previous.x, bullet.y - previous.y)
                             == (previous.vx, previous.vy)
                             and (bullet.vx, bullet.vy) == (previous.vx, previous.vy))
                # The live projectile must actually pass outside the ROM's
                # <10px impact lane. A wall/despawn alone is not an evasion.
                if advancing and old_distance <= 0 < new_distance and moved and lane >= 10:
                    self._target_passed_player = True
            success = (state.frame >= DODGE_SURVIVAL_FRAMES and bullet is None
                       and self._target_passed_player and self._interception is None)
            reason = "evaded"
        elif self.skill == "counter":
            success, reason = self._interception is not None, "intercepted"
            if not success and not any(b.generation == self._target.generation for b in state.bullets):
                self._skill_reason = "target_not_intercepted"
                return "failure"
        elif self.skill == "demolish":
            clear = all(state.brick_masks[r, c] == 0 for r, c in self._gate_cells)
            crossed = (state.player.x if self.variant == 0 else state.player.y) >= 120
            success, reason = clear and crossed and self._shots > 0, "passage_cleared"
        elif self.skill == "pickup":
            success, reason = self._pickups > 0 and state.player.power > 0, "item_collected"
        else:
            success = state.kills == 1 and not state.enemies and not any(
                b.owner >= 2 for b in state.bullets)
            reason = "base_defended"
        if success:
            self._skill_success, self._skill_reason = True, reason
            return "objective"
        if state.frame >= self.max_frames:
            self._skill_reason = "timeout"
            return "timeout"
        return ""

    def _info(self, outcome, components):
        info = super()._info(outcome, components)
        info.update(
            objective="skill", skill=self.skill, variant=self.variant,
            skill_success=self._skill_success, reason=self._skill_reason,
            outcome="objective" if self._skill_success else ("failure" if outcome else ""),
            interception=self._interception,
        )
        return info


def _distinct(values: Iterable[Any], name: str, valid) -> tuple:
    values = tuple(values)
    if not values or any(not valid(value) for value in values) or len(set(values)) != len(values):
        raise ValueError(f"{name} must contain distinct valid values")
    return values


def evaluate_skills(*, model=None, seeds=(101, 202), variants=(0, 1),
                    nstack=2, skills=SKILL_NAMES) -> dict:
    """Measure controlled skill success, independently of full-stage wins.

    ``modes[mode]`` has ``episodes``, ``per_skill`` aggregates and ``summary``.
    ``summary.mean_success_rate`` is the unweighted mean of per-skill rates.
    Each episode retains the objective/failure outcome, reason and ROM counters.
    Imports of the torch/SB3 evaluation stack are deferred until evaluation.
    """
    seeds = _distinct(seeds, "seeds", lambda s: type(s) is int and 0 <= s < 2**32)
    variants = _distinct(variants, "variants", lambda v: type(v) is int and v in (0, 1))
    skills = _distinct(skills, "skills", lambda s: isinstance(s, str) and s in SKILL_NAMES)
    if type(nstack) is not int or nstack <= 0:
        raise ValueError("nstack must be a positive integer")
    from experiments import EpisodePlayer, evaluation_rng

    modes = {}
    for mode in ("deterministic", "sampled"):
        episodes = []
        for skill in skills:
            for variant in variants:
                env = SkillEnv(skill, variant=variant)
                try:
                    player = EpisodePlayer(env, nstack=nstack, model=model,
                                           deterministic=mode == "deterministic")
                    for seed in seeds:
                        with evaluation_rng(seed):
                            player.reset(seed)
                            for _ in range(env.max_frames):
                                _, done, info = player.step()
                                if done:
                                    break
                            else:
                                raise RuntimeError("skill environment exceeded its frame budget")
                        episode = {key: info[key] for key in (
                            "skill", "variant", "skill_success", "reason", "frames",
                            "shots", "deaths", "outcome", "kills", "pickups", "interception",
                        )}
                        episode["seed"] = seed
                        episodes.append(episode)
                finally:
                    env.close()
        per_skill = {}
        for skill in skills:
            selected = [e for e in episodes if e["skill"] == skill]
            successes = sum(e["skill_success"] for e in selected)
            per_skill[skill] = {"episodes": len(selected), "successes": successes,
                                "success_rate": successes / len(selected)}
        modes[mode] = {
            "episodes": episodes, "per_skill": per_skill,
            "summary": {
                "episodes": len(episodes),
                "successes": sum(e["skill_success"] for e in episodes),
                "mean_success_rate": sum(s["success_rate"] for s in per_skill.values()) / len(skills),
            },
        }
    return {
        "schema_version": 1, "benchmark": "controlled_real_rom_skills",
        "policy": "heuristic" if model is None else "MaskablePPO",
        "seeds": list(seeds), "variants": list(variants), "skills": list(skills),
        "nstack": nstack, "max_frames": 600, "modes": modes,
    }
