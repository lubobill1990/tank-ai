"""Fake-environment-only actuator-noise contracts; no ROM/native/GPU work."""
from dataclasses import FrozenInstanceError
import unittest

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from within_stage import execution_noise as noise


class FakeEnv(gym.Env):
    observation_space = spaces.Box(-1, 1, (1, 2, 2), np.float32)
    action_space = spaces.Discrete(10)
    render_mode = None

    def __init__(self, length=3, only_action=None):
        self.length, self.only_action = length, only_action
        self.observation = np.zeros((1, 2, 2), np.float32)
        self.mask = np.ones(10, bool)
        self.info = {}
        self.received, self.reset_arguments = [], []
        self.position = 0
        self.fail_step = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_arguments.append((seed, options))
        self.position = 0
        self.observation.fill(0)
        self.mask.fill(True)
        if self.only_action is not None:
            self.mask.fill(False); self.mask[self.only_action] = True
        self.info.clear()
        self.info['owned_test'] = {'array': self.observation}
        return self.observation, self.info

    def action_masks(self):
        return self.mask

    def step(self, action):
        self.received.append(int(action))
        if self.fail_step:
            raise RuntimeError('synthetic step failure')
        self.position += 1
        self.observation.fill(action / 10)
        self.info.clear()
        done = self.position == self.length
        self.info.update(objective='full', outcome='clear' if done else '',
                         owned_test={'array': self.observation},
                         env_draw=int(self.np_random.integers(0, 100000)),
                         full_rom_win=done)
        if done:
            self.info['episode'] = {'r': -0., 'l': self.position}
        return self.observation, np.float32(-0.), done, False, self.info


class ExecutionNoiseTests(unittest.TestCase):
    def test_probability_zero_preserves_actions_reward_bits_episode_and_opening_reset(self):
        raw = FakeEnv()
        env = noise.ExecutionNoiseEnv(raw)
        options = {'opening': 'chosen-outside-wrapper'}
        env.reset(seed=123, options=options)
        for action in (2, 4, 6):
            argument = np.asarray(action, np.int64)
            _, reward, terminated, truncated, info = env.step(argument)
            self.assertEqual(int(argument), action)
            self.assertEqual(reward.tobytes(), np.float32(-0.).tobytes())
            self.assertEqual(info['execution_noise']['intended_action'], action)
            self.assertEqual(info['execution_noise']['executed_action'], action)
            self.assertIsNone(info['execution_noise']['draw'])
            self.assertFalse(info['execution_noise']['applied'])
            self.assertFalse(truncated)
        self.assertTrue(terminated)
        self.assertEqual(raw.received, [2, 4, 6])
        self.assertEqual(raw.reset_arguments, [(123, options)])
        self.assertEqual(info['episode'], {'r': -0., 'l': 3})
        self.assertTrue(info['full_game_clear'])
        self.assertTrue(info['uninterrupted_model_clear'])

    def test_noise_seed_is_independent_of_policy_global_rng_and_rom_reset_rng(self):
        np.random.seed(94)
        expected_global = np.random.RandomState(94).random_sample(4)
        a = noise.ExecutionNoiseEnv(FakeEnv(length=12), probability=.6, noise_seed=17, stream_id=3)
        b = noise.ExecutionNoiseEnv(FakeEnv(length=12), probability=.6, noise_seed=17, stream_id=3)
        a.reset(seed=1); b.reset(seed=999)
        a_events, b_events = [], []
        expected_base = np.random.default_rng(1)
        for _ in range(10):
            ia, ib = a.step(0)[4], b.step(0)[4]
            self.assertEqual(ia['env_draw'], int(expected_base.integers(0, 100000)))
            a_events.append(ia['execution_noise'])
            b_events.append(ib['execution_noise'])
        self.assertEqual(a_events, b_events)
        np.testing.assert_array_equal(np.random.random_sample(4), expected_global)
        first = a.episode_provenance
        a.reset(seed=1); b.reset(seed=7)
        self.assertEqual(a.episode_provenance, b.episode_provenance)
        self.assertNotEqual(first.episode_noise_seed, a.episode_provenance.episode_noise_seed)
        other = noise.ExecutionNoiseEnv(FakeEnv(), noise_seed=17, stream_id=4)
        other.reset(seed=1)
        self.assertNotEqual(first.episode_noise_seed, other.episode_provenance.episode_noise_seed)

    def test_requested_noise_selects_only_different_legal_primitive(self):
        raw = FakeEnv(length=20)
        env = noise.ExecutionNoiseEnv(raw, probability=1, noise_seed=31)
        env.reset(seed=0)
        raw.mask[:] = False; raw.mask[[0, 3, 8]] = True
        for _ in range(20):
            action = np.asarray(3, np.int64)
            _, _, _, _, info = env.step(action)
            event = info['execution_noise']
            self.assertEqual(int(action), 3)
            self.assertIn(event['executed_action'], (0, 8))
            self.assertTrue(event['requested'])
            self.assertTrue(event['applied'])
            self.assertTrue(0 <= event['draw'] < 1)
        self.assertEqual(env.last_episode.applied_interventions, 20)
        self.assertTrue(info['full_game_clear'])
        self.assertFalse(info['uninterrupted_model_clear'])
        self.assertFalse(info['full_rom_win'])
        self.assertFalse(env.last_episode.recovery_rate_valid)

    def test_recovery_schedule_requests_exactly_once_and_cycles_only_between_episodes(self):
        schedule = [1, 0]
        raw = FakeEnv(length=3)
        env = noise.RecoveryStressEnv(raw, schedule, noise_seed=42, stream_id=1)
        schedule[0] = 99
        env.reset(seed=0)
        events = [env.step(0)[4]['execution_noise'] for _ in range(3)]
        self.assertEqual([event['requested'] for event in events], [False, True, False])
        self.assertEqual(sum(event['applied'] for event in events), 1)
        self.assertTrue(env.last_episode.recovery_rate_valid)
        self.assertTrue(env.last_episode.recovery_success)
        self.assertFalse(env.last_episode.uninterrupted_model_clear)
        first = env.last_episode
        env.reset(seed=999)
        self.assertEqual(env.episode_provenance.scheduled_decision_index, 0)
        self.assertEqual(env.step(0)[4]['execution_noise']['requested'], True)
        self.assertEqual(first.provenance.scheduled_decision_index, 1)
        with self.assertRaises(FrozenInstanceError):
            first.applied_interventions = 10
        with self.assertRaises(FrozenInstanceError):
            env.episode_provenance.episode_noise_seed = 0

    def test_stress_without_actual_intervention_is_invalid_not_an_ordinary_model_win(self):
        raw = FakeEnv(length=2, only_action=0)
        env = noise.RecoveryStressEnv(raw, [0])
        env.reset(seed=0)
        first = env.step(0)[4]
        raw.mask.fill(True)
        last = env.step(0)[4]
        self.assertTrue(first['execution_noise']['requested'])
        self.assertFalse(first['execution_noise']['applied'])
        self.assertFalse(last['execution_noise']['requested'])
        self.assertEqual(env.last_episode.requested_interventions, 1)
        self.assertEqual(env.last_episode.applied_interventions, 0)
        self.assertEqual(env.last_episode.invalid_reason, 'no_legal_alternative')
        self.assertFalse(env.last_episode.recovery_rate_valid)
        self.assertFalse(last['uninterrupted_model_clear'])
        self.assertFalse(last['full_rom_win'])
        self.assertTrue(last['full_game_clear'])
        missed = noise.RecoveryStressEnv(FakeEnv(length=2), [8])
        missed.reset(seed=0)
        missed.step(0); info = missed.step(0)[4]
        self.assertEqual(missed.last_episode.requested_interventions, 0)
        self.assertEqual(missed.last_episode.invalid_reason, 'scheduled_index_not_reached')
        self.assertFalse(info['execution_noise_episode']['recovery_success'])

    def test_data_ownership_and_early_reset_do_not_mutate_past_provenance_or_history(self):
        raw = FakeEnv(length=5)
        env = noise.RecoveryStressEnv(raw, [0])
        reset_obs, reset_info = env.reset(seed=0)
        obs, _, _, _, info = env.step(0)
        snapshot, nested = obs.copy(), info['owned_test']['array'].copy()
        info['execution_noise']['executed_action'] = 99
        env.step(1)
        np.testing.assert_array_equal(obs, snapshot)
        np.testing.assert_array_equal(info['owned_test']['array'], nested)
        np.testing.assert_array_equal(reset_obs, 0)
        np.testing.assert_array_equal(reset_info['owned_test']['array'], 0)
        masks = env.action_masks(); masks.fill(False)
        self.assertTrue(raw.mask.any())
        env.reset(seed=0)
        self.assertFalse(env.last_episode.completed)
        self.assertEqual(env.last_episode.invalid_reason, 'reset_before_termination')

    def test_monitor_dummy_vec_and_frame_stack_keep_episode_boundaries(self):
        env = noise.ExecutionNoiseEnv(FakeEnv(length=2), probability=1)
        vec = VecFrameStack(DummyVecEnv([lambda: Monitor(env)]), n_stack=2, channels_order='first')
        self.addCleanup(vec.close)
        self.assertEqual(vec.reset().shape, (1, 2, 2, 2))
        vec.step(np.array([0]))
        obs, _, dones, infos = vec.step(np.array([0]))
        self.assertTrue(dones[0])
        self.assertEqual(infos[0]['episode']['l'], 2)
        self.assertEqual(infos[0]['execution_noise_episode']['applied_interventions'], 2)
        np.testing.assert_array_equal(obs, 0)
        self.assertEqual(env.episode_provenance.episode_index, 1)
        self.assertEqual(vec.env_method('action_masks')[0].shape, (10,))

    def test_invalid_inputs_deadline_guards_and_step_failure_are_explicit(self):
        for probability in (-.1, 1.1, np.nan, np.inf, True, '0.1'):
            with self.assertRaises(ValueError):
                noise.ExecutionNoiseEnv(FakeEnv(), probability=probability)
        for seed in (-1, 2**64, True, 1.5):
            with self.assertRaises(ValueError):
                noise.ExecutionNoiseEnv(FakeEnv(), noise_seed=seed)
        for schedule in ([], [-1], [True], [1.5]):
            with self.assertRaises(ValueError):
                noise.RecoveryStressEnv(FakeEnv(), schedule)
        expired_raw = FakeEnv()
        expired = noise.ExecutionNoiseEnv(expired_raw, deadline=0)
        with self.assertRaises(TimeoutError):
            expired.reset()
        self.assertEqual(expired_raw.reset_arguments, [])
        calls = []
        raw = FakeEnv()
        env = noise.ExecutionNoiseEnv(raw, probability=1, guard=calls.append)
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset(seed=0)
        for action in (-1, 10, True, 1.5, [0]):
            with self.assertRaises(ValueError):
                env.step(action)
        raw.mask[2] = False
        with self.assertRaises(ValueError):
            env.step(2)
        raw.fail_step = True
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            env.step(0)
        self.assertFalse(env.last_episode.completed)
        self.assertEqual(env.last_episode.applied_interventions, 0)
        self.assertEqual(env.last_episode.attempted_interventions, 1)
        with self.assertRaises(RuntimeError):
            env.step(0)
        self.assertTrue(calls)

    def test_training_requests_without_alternatives_do_not_replace_the_action(self):
        env = noise.ExecutionNoiseEnv(FakeEnv(only_action=0), probability=1.)
        env.reset(seed=0)
        for _ in range(3):
            info = env.step(0)[4]
            self.assertTrue(info['execution_noise']['requested'])
            self.assertFalse(info['execution_noise']['applied'])
            self.assertEqual(info['execution_noise']['executed_action'], 0)
        self.assertEqual(env.last_episode.requested_interventions, 3)
        self.assertEqual(env.last_episode.applied_interventions, 0)
        self.assertTrue(info['uninterrupted_model_clear'])

    def test_wrong_action_space_nested_noise_and_malformed_masks_are_rejected(self):
        for space in (spaces.Discrete(15), spaces.Discrete(10, start=1)):
            raw = FakeEnv(); raw.action_space = space
            with self.assertRaises(ValueError):
                noise.ExecutionNoiseEnv(raw)
        env = noise.ExecutionNoiseEnv(FakeEnv())
        with self.assertRaisesRegex(ValueError, 'nest'):
            noise.RecoveryStressEnv(Monitor(env), [0])
        raw = FakeEnv()
        wrapped = noise.ExecutionNoiseEnv(raw)
        wrapped.reset(seed=0)
        for mask in (np.zeros(10, bool), np.ones(10, np.float32), np.ones((1, 10), bool)):
            raw.mask = mask
            with self.assertRaises(ValueError):
                wrapped.step(0)
        self.assertEqual(raw.received, [])

    def test_stress_subset_is_owned_recorded_and_intersected_with_current_legal_actions(self):
        allowed = [1, 4]
        raw = FakeEnv(length=2)
        env = noise.RecoveryStressEnv(raw, [0], allowed_perturbation_actions=allowed)
        allowed.append(9)
        _, info = env.reset(seed=0)
        self.assertEqual(env.episode_provenance.allowed_perturbation_actions, (1, 4))
        self.assertEqual(info['execution_noise_provenance']['allowed_perturbation_actions'], (1, 4))
        raw.mask.fill(False); raw.mask[[0, 4, 8, 9]] = True
        event = env.step(8)[4]['execution_noise']
        self.assertEqual(event['intended_action'], 8)
        self.assertEqual(event['executed_action'], 4)
        self.assertEqual(event['legal_alternatives'], (4,))
        self.assertTrue(event['applied'])

    def test_default_stress_excludes_fire_but_training_noise_still_allows_it(self):
        raw = FakeEnv(length=2)
        stress = noise.RecoveryStressEnv(raw, [0])
        stress.reset(seed=0)
        raw.mask.fill(False); raw.mask[[0, 5]] = True
        event = stress.step(0)[4]['execution_noise']
        self.assertEqual(event['allowed_perturbation_actions'], (0, 1, 2, 3, 4))
        self.assertTrue(event['requested'])
        self.assertFalse(event['applied'])
        raw.mask.fill(True)
        last = stress.step(0)[4]
        self.assertFalse(last['execution_noise']['requested'])
        self.assertFalse(last['recovery_rate_valid'])
        self.assertFalse(last['recovery_success'])
        self.assertFalse(last['immediate_terminal_after_intervention'])
        training_raw = FakeEnv(length=1)
        training = noise.ExecutionNoiseEnv(training_raw, probability=1.)
        training.reset(seed=0)
        training_raw.mask.fill(False); training_raw.mask[[0, 5]] = True
        event = training.step(0)[4]['execution_noise']
        self.assertEqual(event['executed_action'], 5)
        self.assertEqual(event['allowed_perturbation_actions'], tuple(range(10)))

    def test_immediate_terminal_failure_is_retained_as_a_valid_failed_stress_case(self):
        class TerminalFailure(FakeEnv):
            def step(self, action):
                obs, _, terminated, truncated, info = super().step(action)
                info['outcome'] = 'base_lost'
                info['full_rom_win'] = False
                return obs, np.float32(-20), terminated, truncated, info
        raw = TerminalFailure(length=1)
        env = noise.RecoveryStressEnv(raw, [0], allowed_perturbation_actions=(4,))
        env.reset(seed=0)
        _, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(raw.received, [4])
        self.assertEqual(reward, -20)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['outcome'], 'base_lost')
        self.assertTrue(info['immediate_terminal_after_intervention'])
        self.assertTrue(info['execution_noise']['immediate_terminal_after_intervention'])
        self.assertTrue(env.last_episode.immediate_terminal_after_intervention)
        self.assertTrue(env.last_episode.recovery_rate_valid)
        self.assertFalse(env.last_episode.recovery_success)
        self.assertIsNone(env.last_episode.invalid_reason)
        self.assertEqual(env.last_episode.applied_interventions, 1)
        env.reset(seed=0)
        self.assertEqual(env.episode_provenance.episode_index, 1)

    def test_invalid_stress_subsets_fail_explicitly(self):
        for allowed in ((), (-1,), (10,), (True,), (1.5,), (1, 1), None):
            with self.subTest(allowed=allowed), self.assertRaises(ValueError):
                noise.RecoveryStressEnv(FakeEnv(), [0], allowed_perturbation_actions=allowed)
        with self.assertRaises(TypeError):
            noise.ExecutionNoiseEnv(FakeEnv(), allowed_perturbation_actions=(0, 1))


if __name__ == '__main__':
    unittest.main()
