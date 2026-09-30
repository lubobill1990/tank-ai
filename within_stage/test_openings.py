"""Synthetic opening protocol tests; never start ROMs or collect learning data."""
import hashlib
from types import SimpleNamespace
import unittest

import numpy as np

import game_state as gs
from within_stage import openings


class FakeNES:
    def __init__(self, auto_game_at=None):
        self.ram = np.zeros(2048, np.uint8)
        self.screen = np.zeros((240, 256, 3), np.uint8)
        self.phase, self.stage_ticks, self.auto_game_at = "title", 0, auto_game_at
        self.actions = []
        self._has_backup = False

    def step(self, action):
        self.actions.append(action)
        if action == 8:
            if self.phase == "title":
                self.phase = "stage"
                self.ram[0x85] = 1
            elif self.phase == "stage":
                self.game()
        elif self.phase == "stage":
            self.stage_ticks += 1
            if self.auto_game_at is not None and self.stage_ticks >= self.auto_game_at:
                self.game()

    def game(self):
        self.phase = "game"
        self.ram[0x51], self.ram[0x68], self.ram[0x80] = 3, 0x80, 20
        self.ram[0xa0], self.ram[0xa2] = 0x80, 0x80
        self.ram[0x90], self.ram[0x98] = 88, 208

    def _backup(self):
        self._has_backup = True


def bare_env(spec, auto=None):
    env = openings.NaturalOpeningEnv.__new__(openings.NaturalOpeningEnv)
    env.spec, env.stage = spec, 1
    env.nes = FakeNES(auto)
    env.account = openings.ProbeAccounting(40000, deadline=float("inf"))
    env.checkpoints, env.wait_status = [], {}
    return env


class OpeningTests(unittest.TestCase):
    def test_protocol_counts_every_boot_frame_and_uses_only_legal_start_noop(self):
        env = bare_env(openings.OpeningSpec(title_wait=7, stage_wait=9))
        env._boot()
        self.assertEqual(len(env.nes.actions), 513 + 7 + 9)
        self.assertEqual(env.account.raw_initialization_completed, 529)
        self.assertEqual(set(env.nes.actions), {0, 8})
        self.assertEqual(env.nes.actions.count(8), 2)
        self.assertTrue(env.nes._has_backup)

    def test_stage_wait_auto_start_is_rejected_before_another_start_press(self):
        env = bare_env(openings.OpeningSpec(stage_wait=100), auto=40)
        with self.assertRaises(openings.InvalidOpening):
            env._boot()
        self.assertTrue(env.wait_status["stage_wait_advanced_game"])
        self.assertEqual(env.nes.actions.count(8), 1)

    def test_predeclared_grid_has_at_most24_unique_menu_specs_and_fixed_postboot(self):
        specs = openings.predeclared_specs()
        self.assertEqual(len(specs), 16)
        self.assertEqual(len({(s.title_wait, s.stage_wait) for s in specs}), 16)
        self.assertTrue(all(s.legacy_phase is None for s in specs))
        self.assertLessEqual(openings.conservative_plan_bound(), 40000)

    def test_legacy_phases_are_explicit_and_cannot_be_mixed_with_menu_waits(self):
        with self.assertRaises(ValueError):
            openings.OpeningSpec(title_wait=1, legacy_phase=3)
        with self.assertRaises(ValueError):
            openings.OpeningSpec(legacy_phase=24)
        with self.assertRaises(ValueError):
            openings.OpeningSpec(title_wait=-1)

    def test_budget_stops_before_spending_and_counts_agent_calls_separately(self):
        budget = openings.ProbeAccounting(2, deadline=float("inf"))
        budget.before("raw")
        budget.after("raw")
        budget.before("agent")
        budget.after("agent")
        with self.assertRaises(openings.ProbeStop):
            budget.before("raw")
        self.assertEqual(budget.attempted, 2)
        self.assertEqual(budget.completed, 2)

    def test_semantic_hash_ignores_cosmetic_bytes_but_detects_enemy_positions(self):
        ram = np.zeros(2048, np.uint8)
        ram[0x85], ram[0x51], ram[0x68], ram[0x80] = 1, 3, 0x80, 20
        ram[0xa0], ram[0x90], ram[0x98] = 0x80, 88, 208
        ram[0xa2], ram[0x92], ram[0x9a] = 0x80, 24, 24
        other = ram.copy()
        other[0x0f] = 123
        self.assertNotEqual(hashlib.sha256(ram).hexdigest(), hashlib.sha256(other).hexdigest())
        self.assertEqual(openings.semantic_hash(gs.parse(ram)), openings.semantic_hash(gs.parse(other)))
        other[0x92] += 1
        self.assertNotEqual(openings.semantic_hash(gs.parse(ram)), openings.semantic_hash(gs.parse(other)))


if __name__ == "__main__":
    unittest.main()
