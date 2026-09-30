"""One-variable reward ablation for the seeded native VecEnv.

``baseline`` preserves reward bits and upstream episode summaries.
``loss_balanced_v1`` subtracts 20 only on terminal ``lives_exhausted``;
all other rewards and timeout truncation flags are unchanged.
"""
from copy import deepcopy
from numbers import Real

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class RewardProfileVecEnv(VecEnvWrapper):
    """Wrap native rewards without changing observations, actions, or termination.

    Every info contains ``reward_profile``, ``original_reward`` (the upstream
    step reward), and ``reward_adjustment`` (the nominal adjustment, 0 or -20).
    Terminal infos also contain ``original_episode_return`` (upstream
    ``episode['r']``). For the balanced profile, ``episode['r']`` is the float64
    sum of the actual emitted rewards, including their native dtype rounding.
    Baseline leaves the upstream episode dictionary's values unchanged.

    Terminal infos must have an explicit native outcome and a finite numeric
    ``episode['r']``; malformed contracts raise ValueError, not a fallback.
    Outputs and metadata own their storage, including terminal observations.
    Standard VecEnvWrapper forwarding preserves masks, seed, and reset options.
    """

    def __init__(self, venv: VecEnv, profile: str = 'baseline'):
        if not isinstance(profile, str) or profile not in ('baseline', 'loss_balanced_v1'):
            raise ValueError(f'Unknown reward profile: {profile!r}')
        super().__init__(venv)
        self.profile = profile
        self._profile_returns = np.zeros(self.num_envs, dtype=np.float64)

    def reset(self):
        observations = self.venv.reset()
        self._profile_returns.fill(0)
        return deepcopy(observations)

    def step_wait(self):
        observations, upstream_rewards, upstream_dones, upstream_infos = self.venv.step_wait()
        rewards = upstream_rewards.copy()
        dones = upstream_dones.copy()
        infos = deepcopy(upstream_infos)
        terminal_indices = np.flatnonzero(dones)
        for index in terminal_indices:
            info = infos[index]
            outcome = info.get('outcome')
            if not isinstance(outcome, str) or outcome not in (
                    'base_lost', 'lives_exhausted', 'clear', 'timeout'):
                raise ValueError(f'Invalid terminal outcome for environment {index}: {outcome!r}')
            episode = info.get('episode')
            if not isinstance(episode, dict) or not isinstance(episode.get('r'), Real) \
                    or isinstance(episode['r'], (bool, np.bool_)) or not np.isfinite(episode['r']):
                raise ValueError(f'Invalid terminal episode return for environment {index}')

        for index, info in enumerate(infos):
            adjustment = -20.0 if (
                self.profile == 'loss_balanced_v1' and dones[index]
                and info['outcome'] == 'lives_exhausted'
            ) else 0.0
            info['reward_profile'] = self.profile
            info['original_reward'] = float(rewards[index])
            info['reward_adjustment'] = adjustment
            if adjustment:
                rewards[index] += adjustment

        self._profile_returns += rewards
        for index in terminal_indices:
            info = infos[index]
            info['original_episode_return'] = float(info['episode']['r'])
            if self.profile == 'loss_balanced_v1':
                info['episode']['r'] = float(self._profile_returns[index])
            self._profile_returns[index] = 0
        return deepcopy(observations), rewards, dones, infos
