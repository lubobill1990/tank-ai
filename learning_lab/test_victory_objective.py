import unittest
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym


class VictoryObjectiveTests(unittest.TestCase):
    def test_episode_total_is_only_win_or_loss_regardless_of_progress(self):
        from learning_lab.victory_objective import victory_transition
        for final in ('clear', 'base_lost', 'lives_exhausted', 'timeout'):
            for path in ([0, .1, .3, .6], [0, 0, 0], [0, .95, 1]):
                total = sum(victory_transition(a, b, '') for a, b in zip(path, path[1:]))
                total += victory_transition(path[-1], path[-1], final)
                self.assertAlmostEqual(total, 1 if final == 'clear' else -1)

    def test_invalid_potential_or_outcome_does_not_silently_change_objective(self):
        from learning_lab.victory_objective import victory_transition
        for args in ((0, -1, ''), (0, 1.1, ''), (float('nan'), 0, ''),
                     (0, 0, 'unknown'), (.5, .2, '')):
            with self.assertRaises(ValueError):
                victory_transition(*args)

    def test_wrapper_corrects_returns_makes_timeout_terminal_and_resets_potential(self):
        from learning_lab.victory_objective import FiniteVictoryVecEnv

        class CounterEnv(gym.Env):
            observation_space = gym.spaces.Box(-1, 1, (47, 26, 26), np.float32)
            action_space = gym.spaces.Discrete(10)

            def reset(self, *, seed=None, options=None):
                self.kills = 0
                return np.zeros(self.observation_space.shape, np.float32), {}

            def step(self, action):
                self.kills += 1
                done = self.kills == 2
                info = {'kills': self.kills}
                if done:
                    info.update(outcome='timeout', episode={'r': -0.004, 'l': 2})
                return np.zeros(self.observation_space.shape, np.float32), -0.002, False, done, info

        raw = DummyVecEnv([CounterEnv])
        wrapped = FiniteVictoryVecEnv(raw, [4], lambda: np.array([raw.envs[0].kills]))
        try:
            wrapped.reset()
            for _ in range(2):
                _, first, _, _ = wrapped.step(np.array([0]))
                _, last, dones, infos = wrapped.step(np.array([0]))
                self.assertEqual(float(first[0]), .25)
                self.assertEqual(float(last[0]), -1.25)
                self.assertTrue(dones[0])
                self.assertFalse(infos[0]['TimeLimit.truncated'])
                self.assertTrue(infos[0]['original_time_limit_truncated'])
                self.assertEqual(infos[0]['episode']['r'], -1)
                self.assertEqual(infos[0]['original_episode_return'], -0.004)
        finally:
            wrapped.close()

    def test_discounted_use_is_rejected_instead_of_claiming_exact_win_objective(self):
        from learning_lab.victory_objective import FiniteVictoryVecEnv
        with self.assertRaises(ValueError):
            FiniteVictoryVecEnv(None, [20], lambda: np.array([0]), gamma=.99)

    def test_sparse_mode_has_the_same_remaining_return_across_curriculum_starts(self):
        from learning_lab.victory_objective import victory_transition
        for progress in (0, .5, .95):
            self.assertEqual(victory_transition(progress, 1, 'clear', shaping=False), 1)
            self.assertEqual(victory_transition(progress, 1, 'timeout', shaping=False), -1)
            self.assertEqual(victory_transition(progress, 1, '', shaping=False), 0)
