"""CPU fake-backend tests; no real emulator or controller decisions."""
import hashlib
from types import SimpleNamespace
import unittest

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from learning_lab.victory_objective import FiniteVictoryVecEnv
from learning_lab import rom_curriculum_env as curriculum


class FakeNES:
    def __init__(self):
        self.ram = np.zeros(2048, np.uint8)
        self.screen = np.zeros((4, 4, 3), np.uint8)
        self.saved_ram, self.saved_screen = self.ram.copy(), self.screen.copy()
        self._has_backup = True
        self.backups = 0

    def step(self, action):
        self.ram[0] += 1
        self.screen.fill(self.ram[0])
        return self.screen, 0., False, {}

    def _backup(self):
        self.saved_ram, self.saved_screen = self.ram.copy(), self.screen.copy()
        self.backups += 1

    def restore(self):
        self.ram[:] = self.saved_ram
        self.screen[:] = self.saved_screen


class FakeBase(gym.Env):
    observation_space = spaces.Box(-1, 1, (47, 26, 26), np.float32)
    action_space = spaces.Discrete(15)

    def __init__(self):
        self.nes = FakeNES()
        self.max_frames = 12000
        self.frame_skip = 4
        self.reset_seeds = []
        self.corrupt_restore = False
        self.terminal_frame = None
        self.closed = False
        self._state(0)

    def _state(self, frame):
        self.state = SimpleNamespace(frame=frame, kills=5 + int(self.nes.ram[0]) // 4,
                                     remaining_enemies=5 - int(self.nes.ram[0]) // 4,
                                     stage=1, base_destroyed=False, game_over=False, player=object())

    def _observation(self):
        obs = np.zeros((47, 26, 26), np.float32)
        obs[4, 24:26, 12:14] = 1
        obs[36].fill(max(0., 1 - self.state.frame / self.max_frames))
        return obs

    def reset(self, *, seed=None, options=None):
        self.reset_seeds.append(seed)
        self.nes.restore()
        if seed != 23:
            self.nes.step(0)
            self.nes.ram[0] = 0
            self.nes.screen.fill(0)
        elif self.corrupt_restore:
            self.nes.screen[0, 0, 0] ^= 1
        self._state(0)
        return self._observation(), {'outcome': '', 'kills': self.state.kills}

    def step(self, action):
        for _ in range(4):
            self.nes.step(action)
        self._state(self.state.frame + 4)
        outcome = ('clear' if self.state.remaining_enemies <= 0 else
                   'base_lost' if self.terminal_frame and self.state.frame >= self.terminal_frame else
                   'timeout' if self.state.frame >= self.max_frames else '')
        return self._observation(), -0.002, bool(outcome and outcome != 'timeout'), outcome == 'timeout', {
            'outcome': outcome, 'kills': self.state.kills, 'frames': self.state.frame}

    def action_masks(self):
        masks = np.ones(15, bool)
        masks[5] = False
        return masks

    def close(self):
        self.closed = True


class FakeTeacher:
    def __init__(self):
        self.calls = 0

    def reset(self):
        pass

    def act(self, state, masks):
        self.calls += 1
        return 0


class RomCurriculumTests(unittest.TestCase):
    def make(self, *, source_frame=8, max_frames=8, full_start=False, **kwargs):
        ram = np.zeros(2048, np.uint8)
        ram[0] = source_frame
        spec = {'stage': 1, 'source_seed': 1, 'source_frame': source_frame,
                'ram_sha256': hashlib.sha256(ram.tobytes()).hexdigest(),
                'remaining': 5 - source_frame // 4, 'full_start': full_start}
        spec.update(kwargs.pop('spec_updates', {}))
        raw, teacher = FakeBase(), FakeTeacher()

        class FakeCurriculum(curriculum.RomCurriculumEnv):
            def _make_raw_env(self):
                return raw

            def _make_teacher(self):
                return teacher

        env = FakeCurriculum(spec, max_frames=max_frames, **kwargs)
        self.addCleanup(env.close)
        return env, raw, teacher

    def test_prepare_once_exact_restore_and_never_teacher_on_later_resets(self):
        events = []
        env, raw, teacher = self.make(initialization_counter=events.append)
        self.assertEqual(teacher.calls, 0)
        obs, info = env.reset(seed=99)
        self.assertEqual(raw.reset_seeds, [1, 23])
        self.assertEqual(teacher.calls, 2)
        self.assertEqual(raw.nes.backups, 1)
        self.assertEqual(raw.state.frame, 0)
        self.assertEqual(env.initial_remaining, 3)
        self.assertEqual(env.current_kills, 0)
        self.assertEqual(obs.shape, (47, 26, 26))
        self.assertEqual(obs[36, 0, 0], 1)
        self.assertTrue(info['curriculum_start'])
        self.assertFalse(info['full_rom_win'])
        expected_ram, expected_pixels = raw.nes.ram.copy(), raw.nes.screen.copy()
        env.step(0)
        self.assertEqual(env.current_kills, 1)
        restored, _ = env.reset()
        np.testing.assert_array_equal(raw.nes.ram, expected_ram)
        np.testing.assert_array_equal(raw.nes.screen, expected_pixels)
        np.testing.assert_array_equal(restored, obs)
        self.assertEqual(raw.reset_seeds, [1, 23, 23])
        self.assertEqual(teacher.calls, 2)
        self.assertEqual([event['kind'] for event in events],
                         ['normal_reset_substep', 'teacher_decision', 'teacher_decision'])
        self.assertEqual(env.initialization_counts['teacher_decisions_attempted'], 2)
        self.assertEqual(env.initialization_counts['teacher_decisions_completed'], 2)
        self.assertEqual(env.initialization_counts['normal_reset_substeps_attempted'], 1)
        self.assertEqual(env.learner_counts, {'attempted': 1, 'completed': 1})

    def test_hash_mismatch_past_terminal_and_bad_sources_fail_without_recovery(self):
        env, raw, _ = self.make(spec_updates={'ram_sha256': '0' * 64})
        with self.assertRaisesRegex(ValueError, 'RAM'):
            env.reset()
        self.assertEqual(raw.nes.backups, 0)
        with self.assertRaises(RuntimeError):
            env.reset()
        second, raw2, _ = self.make()
        raw2.terminal_frame = 4
        with self.assertRaisesRegex(RuntimeError, 'terminal'):
            second.reset()
        self.assertEqual(raw2.nes.backups, 0)
        with self.assertRaisesRegex(ValueError, 'source'):
            self.make(spec_updates={'source_sha256': {'learning_lab\\teacher_v2.py': '0' * 64}})

    def test_corrupt_pixel_restore_is_rejected_and_poisoned(self):
        env, raw, _ = self.make()
        env.reset()
        raw.corrupt_restore = True
        with self.assertRaisesRegex(RuntimeError, 'snapshot'):
            env.reset()
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_deadline_and_initialization_callback_stop_before_extra_actions(self):
        env, raw, teacher = self.make(deadline=0)
        with self.assertRaises(TimeoutError):
            env.reset()
        self.assertEqual(raw.reset_seeds, [])
        self.assertEqual(teacher.calls, 0)
        def stop(event):
            raise RuntimeError('bootstrap allowance exhausted')
        limited, raw2, _ = self.make(initialization_counter=stop)
        with self.assertRaisesRegex(RuntimeError, 'allowance'):
            limited.reset()
        self.assertEqual(limited.initialization_counts['normal_reset_substeps_attempted'], 0)

    def test_masks_and_victory_frame_stack_are_compatible(self):
        env, _, _ = self.make()
        vec = DummyVecEnv([lambda: Monitor(env)])
        victory = FiniteVictoryVecEnv(vec, np.array([env.initial_remaining]),
                                      lambda: np.asarray(vec.get_attr('current_kills'), np.int64),
                                      shaping=False)
        stacked = VecFrameStack(victory, n_stack=2, channels_order='first')
        self.addCleanup(stacked.close)
        observations = stacked.reset()
        self.assertEqual(observations.shape, (1, 94, 26, 26))
        self.assertEqual(stacked.env_method('action_masks')[0].shape, (10,))
        self.assertFalse(stacked.env_method('action_masks')[0][5])
        _, rewards, dones, _ = stacked.step(np.array([0]))
        self.assertEqual(rewards[0], 0)
        self.assertFalse(dones[0])
        _, rewards, dones, infos = stacked.step(np.array([0]))
        self.assertTrue(dones[0])
        self.assertEqual(rewards[0], -1)
        self.assertEqual(infos[0]['episode']['r'], -1)
        self.assertEqual(infos[0]['outcome'], 'timeout')
        self.assertTrue(infos[0]['curriculum_start'])
        self.assertFalse(infos[0]['TimeLimit.truncated'])

    def test_full_start_uses_only_fixed_training_seed_without_teacher_or_zero_delay_seed(self):
        env, raw, teacher = self.make(source_frame=0, full_start=True)
        _, info = env.reset(seed=23)
        env.step(0)
        env.reset(seed=77)
        self.assertEqual(raw.reset_seeds, [1, 1])
        self.assertEqual(teacher.calls, 0)
        self.assertEqual(raw.nes.backups, 0)
        self.assertFalse(info['curriculum_start'])
        self.assertTrue(info['normal_training_start'])

    def test_invalid_stages_seeds_frames_and_native_full_reference_are_rejected(self):
        for spec in ({'stage': 3}, {'source_seed': 23}, {'source_frame': 7},
                     {'source_frame': -1, 'full_start': True}, {'remaining': 0}):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                self.make(spec_updates=spec)


if __name__ == '__main__':
    unittest.main()
