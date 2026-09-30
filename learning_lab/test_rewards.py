"""Reward-only ablation tests; native acceptance uses stage 1, not held-out stages."""
from copy import deepcopy
import os
import unittest

from gymnasium import spaces
import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack

from learning_lab.rewards import RewardProfileVecEnv


def terminal(outcome, reward_return, length=1):
    return {
        'outcome': outcome, 'TimeLimit.truncated': outcome == 'timeout',
        'episode': {'r': reward_return, 'l': length, 't': 0.25},
        'frames': length * 4, 'kills': 0, 'deaths': 0, 'shots': 0,
        'pickups': 0, 'lives': 1, 'episode_seed': 123,
    }


class ScriptedVecEnv(VecEnv):
    """A small deterministic vector source that deliberately reuses output storage."""

    def __init__(self, steps):
        self.steps = deepcopy(steps)
        n = len(steps[0][0])
        self.obs = np.zeros((n, 1, 2, 2), dtype=np.float32)
        self.rewards = np.zeros(n, dtype=np.float32)
        self.dones = np.zeros(n, dtype=bool)
        self.terminals = self.obs.copy()
        self.infos = [{} for _ in range(n)]
        self.position = 0
        self.actions = None
        self.render_mode = None
        self.masks = np.tile([True, False, True], (n, 1))
        super().__init__(n, spaces.Box(-1, 1, (1, 2, 2), dtype=np.float32),
                         spaces.Discrete(3))

    def reset(self):
        self.position = 0
        self.obs.fill(0)
        self.rewards.fill(0)
        self.dones.fill(False)
        self._reset_seeds()
        return self.obs

    def step_async(self, actions):
        self.actions = np.array(actions, copy=True)

    def step_wait(self):
        rewards, dones, infos = self.steps[self.position]
        self.position += 1
        self.rewards[:] = rewards
        self.dones[:] = dones
        self.obs.fill(self.position / 10)
        self.terminals[:] = self.obs
        self.obs[self.dones] = 0
        for index, info in enumerate(infos):
            self.infos[index].clear()
            self.infos[index].update(deepcopy(info))
            if self.dones[index]:
                self.infos[index]['terminal_observation'] = self.terminals[index]
        return self.obs, self.rewards, self.dones, self.infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        if attr_name == 'render_mode':
            return [None for _ in self._get_indices(indices)]
        if attr_name == 'action_masks':
            return [lambda index=i: self.masks[index].copy()
                    for i in self._get_indices(indices)]
        raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError(attr_name)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if method_name != 'action_masks' or args or kwargs:
            raise AttributeError(method_name)
        return [self.masks[i].copy() for i in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


class RewardProfileTests(unittest.TestCase):
    def wrap(self, steps, profile='baseline'):
        source = ScriptedVecEnv(steps)
        env = RewardProfileVecEnv(source, profile=profile)
        self.addCleanup(env.close)
        env.reset()
        return source, env

    def test_baseline_preserves_reward_bits_and_original_episode_summary(self):
        values = [-0.0, -3.001, -20.002, 4.9995, -0.0005]
        outcomes = ['clear', 'lives_exhausted', 'base_lost', 'clear', 'timeout']
        infos = [terminal(outcome, 7.123456, 17) for outcome in outcomes]
        source, env = self.wrap([(values, [True] * 5, infos)])
        _, rewards, dones, result = env.step(np.zeros(5, dtype=np.int64))
        self.assertEqual(rewards.dtype, np.float32)
        self.assertEqual(rewards.tobytes(), np.asarray(values, np.float32).tobytes())
        np.testing.assert_array_equal(dones, [True] * 5)
        for index, info in enumerate(result):
            self.assertEqual(info['episode'], infos[index]['episode'])
            self.assertEqual(info['original_episode_return'], 7.123456)
            self.assertEqual(info['reward_profile'], 'baseline')
            self.assertEqual(info['original_reward'], float(source.rewards[index]))
            self.assertEqual(info['reward_adjustment'], 0.0)
            self.assertEqual(info['TimeLimit.truncated'], outcomes[index] == 'timeout')

    def test_balanced_changes_only_done_lives_exhausted(self):
        values = [-3.002, -20.002, 5.0, -0.002, -0.0]
        infos = [terminal(outcome, value) for outcome, value in
                 zip(['lives_exhausted', 'base_lost', 'clear', 'timeout'], values)]
        infos.append({'outcome': 'lives_exhausted'})
        source, env = self.wrap([(values, [True] * 4 + [False], infos)],
                                'loss_balanced_v1')
        _, rewards, _, result = env.step(np.zeros(5, dtype=np.int64))
        expected = np.array([-23.002, -20.002, 5.0, -0.002, -0.0], np.float32)
        self.assertEqual(rewards.tobytes(), expected.tobytes())
        self.assertEqual(source.rewards.tobytes(), np.asarray(values, np.float32).tobytes())
        self.assertEqual(result[0]['reward_adjustment'], -20.0)
        for index, info in enumerate(result):
            self.assertEqual(info['reward_profile'], 'loss_balanced_v1')
            self.assertEqual(info['original_reward'], float(source.rewards[index]))
            if index:
                self.assertEqual(info['reward_adjustment'], 0.0)
            if index < 4:
                self.assertEqual(info['episode']['r'], float(rewards[index]))
                self.assertEqual(info['original_episode_return'], values[index])
                self.assertEqual(info['TimeLimit.truncated'], index == 3)
        self.assertNotIn('episode', result[4])
        self.assertNotIn('original_episode_return', result[4])

    def test_mixed_autoresets_accumulate_each_environment_independently(self):
        steps = [
            ([1.0, -0.5], [False, False], [{}, {}]),
            ([-3.0, 2.0], [True, False], [terminal('lives_exhausted', -2, 2), {}]),
            ([0.25, -20.0], [False, True], [{}, terminal('base_lost', -18.5, 3)]),
            ([-3.0, -3.0], [True, True],
             [terminal('lives_exhausted', -2.75, 2), terminal('lives_exhausted', -3)]),
            ([1.0, -0.5], [True, True], [terminal('clear', 1), terminal('timeout', -0.5)]),
        ]
        _, env = self.wrap(steps, 'loss_balanced_v1')
        observed_returns, originals, adjustments = [], [], []
        for _ in steps:
            _, _, dones, infos = env.step(np.array([0, 2]))
            for index in np.flatnonzero(dones):
                observed_returns.append(infos[index]['episode']['r'])
                originals.append(infos[index]['original_episode_return'])
                adjustments.append(infos[index]['reward_adjustment'])
        self.assertEqual(observed_returns, [-22, -18.5, -22.75, -23, 1, -0.5])
        self.assertEqual(originals, [-2, -18.5, -2.75, -3, 1, -0.5])
        self.assertEqual(adjustments, [-20, 0, -20, -20, 0, 0])

    def test_explicit_reset_discards_partial_episode_return(self):
        steps = [([1], [False], [{}]),
                 ([-3], [True], [terminal('lives_exhausted', -2, 2)])]
        _, env = self.wrap(steps, 'loss_balanced_v1')
        env.step(np.array([0]))
        env.reset()
        env.step(np.array([0]))
        _, _, _, infos = env.step(np.array([0]))
        self.assertEqual(infos[0]['episode']['r'], -22)

    def test_terminal_return_sums_emitted_float32_rewards_not_unrounded_penalty(self):
        steps = [([0.1], [False], [{}]),
                 ([-3.001], [True], [terminal('lives_exhausted', -2.901, 2)])]
        _, env = self.wrap(steps, 'loss_balanced_v1')
        first = env.step(np.array([0]))[1]
        _, last, _, infos = env.step(np.array([0]))
        expected = float(np.float32(0.1)) + float(np.float32(-23.001))
        self.assertEqual(infos[0]['episode']['r'], expected)
        self.assertEqual(infos[0]['episode']['r'], float(first[0]) + float(last[0]))
        self.assertEqual(infos[0]['episode']['l'], 2)
        self.assertEqual(infos[0]['episode']['t'], 0.25)

    def test_outputs_and_metadata_own_storage_across_steps_and_caller_mutation(self):
        for profile in ('baseline', 'loss_balanced_v1'):
            with self.subTest(profile=profile):
                steps = [([-3], [True], [terminal('lives_exhausted', -3)]),
                         ([1], [False], [{}])]
                source, env = self.wrap(steps, profile)
                first = env.step(np.array([0]))
                snapshot = deepcopy(first)
                first[3][0]['episode']['l'] = 999
                self.assertEqual(source.infos[0]['episode']['l'], 1)
                first[3][0]['episode']['l'] = 1
                env.step(np.array([0]))
                for actual, expected in zip(first[:3], snapshot[:3]):
                    np.testing.assert_array_equal(actual, expected)
                for key, expected in snapshot[3][0].items():
                    if isinstance(expected, np.ndarray):
                        np.testing.assert_array_equal(first[3][0][key], expected)
                    else:
                        self.assertEqual(first[3][0][key], expected)
                self.assertFalse(np.shares_memory(first[1], source.rewards))
                self.assertFalse(np.shares_memory(first[2], source.dones))

    def test_reset_observation_is_owned(self):
        _, env = self.wrap([([1], [False], [{}])])
        reset_obs = env.reset()
        env.step(np.array([0]))
        np.testing.assert_array_equal(reset_obs, np.zeros_like(reset_obs))

    def test_rejects_unknown_profiles_without_silent_fallback(self):
        source = ScriptedVecEnv([([0], [False], [{}])])
        for profile in ('loss_balanced', '', None, [], True):
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, 'profile'):
                RewardProfileVecEnv(source, profile=profile)

    def test_rejects_missing_unknown_or_nonstring_terminal_outcome(self):
        for profile in ('baseline', 'loss_balanced_v1'):
            for outcome in (None, '', 'lost', [], 1):
                with self.subTest(profile=profile, outcome=outcome):
                    info = terminal(outcome, 0)
                    if outcome is None:
                        del info['outcome']
                    _, env = self.wrap([([0], [True], [info])], profile)
                    with self.assertRaisesRegex(ValueError, 'outcome'):
                        env.step(np.array([0]))

    def test_rejects_malformed_terminal_episode_summary(self):
        for episode in (None, {}, {'l': 1}, {'r': 'bad'}, {'r': np.nan}):
            with self.subTest(episode=episode):
                info = terminal('lives_exhausted', -3)
                info['episode'] = episode
                _, env = self.wrap([([-3], [True], [info])], 'loss_balanced_v1')
                with self.assertRaisesRegex(ValueError, 'episode'):
                    env.step(np.array([0]))

    def test_frame_stack_masks_actions_and_seed_forwarding(self):
        source, wrapper = self.wrap([
            ([0.25, 0.5], [False, False], [{}, {}]),
            ([-3, -0.5], [True, True],
             [terminal('lives_exhausted', -2.75, 2), terminal('timeout', 0, 2)]),
        ], 'loss_balanced_v1')
        env = VecFrameStack(wrapper, n_stack=2, channels_order='first')
        self.assertEqual(env.seed(123), [123, 124])
        self.assertEqual(source._seeds, [123, 124])
        np.testing.assert_array_equal(env.reset(), np.zeros((2, 2, 2, 2)))
        self.assertTrue(is_masking_supported(env))
        np.testing.assert_array_equal(get_action_masks(env), [[True, False, True]] * 2)
        env.step_async(np.array([2, 0]))
        np.testing.assert_array_equal(source.actions, [2, 0])
        first = env.step_wait()[0].copy()
        obs, _, _, infos = env.step(np.array([0, 2]))
        np.testing.assert_array_equal(obs, np.zeros_like(obs))
        for index, info in enumerate(infos):
            np.testing.assert_array_equal(info['terminal_observation'][:1], first[index, 1:])
        self.assertEqual(infos[0]['episode']['r'], -22.75)
        self.assertEqual(infos[1]['episode']['r'], 0)
        self.assertTrue(infos[1]['TimeLimit.truncated'])


@unittest.skipUnless(os.environ.get('TANK_AI_NATIVE_REWARD_TESTS') == '1',
                     'Set TANK_AI_NATIVE_REWARD_TESTS=1 for 81 native decisions on stage 1')
class NativeRewardAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rust_env import stage_ram
        cls.initial = stage_ram(1, seed=0)[None, :]

    def make_envs(self, initial, max_frames, stacked=False):
        from rust_env import RustVectorEnv
        envs = []
        for profile in (None, 'baseline', 'loss_balanced_v1'):
            env = RustVectorEnv(initial.copy(), max_frames, seed=73)
            self.addCleanup(env.close)
            if profile is not None:
                env = RewardProfileVecEnv(env, profile)
            if stacked:
                env = VecFrameStack(env, n_stack=2, channels_order='first')
            envs.append(env)
        return envs

    def test_native_timeouts_seed_masks_and_stacks_match_unwrapped(self):
        envs = self.make_envs(np.repeat(self.initial, 2, axis=0), 8, stacked=True)
        traces = []
        for _ in range(2):
            for env in envs:
                self.assertEqual(env.seed(101), [101, 102])
                self.assertTrue(is_masking_supported(env))
            resets = [env.reset() for env in envs]
            for actual in resets[1:]:
                np.testing.assert_array_equal(actual, resets[0])
            trace = []
            for actions in ([5, 1], [6, 2], [7, 3], [8, 4], [9, 5], [0, 6]):
                masks = [get_action_masks(env) for env in envs]
                for actual in masks[1:]:
                    np.testing.assert_array_equal(actual, masks[0])
                outputs = [env.step(np.asarray(actions)) for env in envs]
                trace.append(deepcopy(outputs[1]))
                control_obs, control_rewards, control_dones, control_infos = outputs[0]
                for observations, rewards, dones, infos in outputs[1:]:
                    np.testing.assert_array_equal(observations, control_obs)
                    self.assertEqual(rewards.tobytes(), control_rewards.tobytes())
                    np.testing.assert_array_equal(dones, control_dones)
                    for index in np.flatnonzero(dones):
                        self.assertEqual(infos[index]['outcome'], 'timeout')
                        self.assertTrue(infos[index]['TimeLimit.truncated'])
                        self.assertEqual(infos[index]['episode'], control_infos[index]['episode'])
                        self.assertEqual(infos[index]['original_episode_return'],
                                         control_infos[index]['episode']['r'])
                        np.testing.assert_array_equal(infos[index]['terminal_observation'],
                                                      control_infos[index]['terminal_observation'])
            traces.append(trace)
        for first, second in zip(*traces):
            for actual, expected in zip(first[:3], second[:3]):
                np.testing.assert_array_equal(actual, expected)
            for actual, expected in zip(first[3], second[3]):
                self.assertEqual(actual.get('episode_seed'), expected.get('episode_seed'))

    def test_native_lives_exhausted_penalty_repeats_once_per_autoreset(self):
        # Supported preterminal death animation; not a policy/gameplay evaluation.
        initial = self.initial.copy()
        initial[0, 0x51] = 1
        initial[0, 0xA0] = 0x11
        envs = self.make_envs(initial, 120)
        for env in envs:
            env.reset()
        seeds = []
        for _ in range(3):
            outputs = [env.step(np.array([0])) for env in envs]
            original = outputs[0][1][0]
            self.assertEqual(outputs[1][1].tobytes(), outputs[0][1].tobytes())
            self.assertEqual(outputs[2][1][0], np.float32(float(original) - 20))
            for index, (obs, rewards, dones, infos) in enumerate(outputs):
                self.assertTrue(dones[0])
                self.assertEqual(infos[0]['outcome'], 'lives_exhausted')
                self.assertFalse(infos[0]['TimeLimit.truncated'])
                self.assertEqual(infos[0]['episode']['r'], float(rewards[0]))
                np.testing.assert_array_equal(obs, outputs[0][0])
                np.testing.assert_array_equal(infos[0]['terminal_observation'],
                                              outputs[0][3][0]['terminal_observation'])
                if index:
                    self.assertEqual(infos[0]['original_reward'], float(original))
                    self.assertEqual(infos[0]['original_episode_return'],
                                     outputs[0][3][0]['episode']['r'])
            seeds.append(outputs[2][3][0]['episode_seed'])
        self.assertEqual(len(set(seeds)), 3)


if __name__ == '__main__':
    unittest.main()
