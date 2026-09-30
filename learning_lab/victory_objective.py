"""Finite-horizon win probability with telescoping progress shaping, gamma=1."""
import numpy as np

from learning_lab.rewards import RewardProfileVecEnv

OBJECTIVE_VERSION = 'finite-victory-potential-v1'
SPARSE_OBJECTIVE_VERSION = 'finite-victory-sparse-v1'
FAILURES = frozenset(('base_lost', 'lives_exhausted', 'timeout'))


def victory_transition(previous_potential, current_potential, outcome, *, shaping=True):
    if (not np.isfinite(previous_potential) or not np.isfinite(current_potential)
            or not 0 <= previous_potential <= 1 or not 0 <= current_potential <= 1
            or outcome not in ('', 'clear', *FAILURES)):
        raise ValueError('Invalid progress potential or terminal outcome')
    if outcome:
        return (1.0 if outcome == 'clear' else -1.0) - (previous_potential if shaping else 0.0)
    if current_potential < previous_potential:
        raise ValueError('Kill progress cannot decrease inside an episode')
    return current_potential - previous_potential if shaping else 0.0


class FiniteVictoryVecEnv(RewardProfileVecEnv):
    objective_version = OBJECTIVE_VERSION

    def __init__(self, venv, initial_remaining, read_kills, *, gamma=1.0, shaping=True):
        if isinstance(gamma, bool) or gamma != 1.0:
            raise ValueError('The exact finite victory objective requires gamma=1')
        if type(shaping) is not bool:
            raise ValueError('shaping must be an explicit boolean')
        self._shaping = shaping
        self.objective_version = OBJECTIVE_VERSION if shaping else SPARSE_OBJECTIVE_VERSION
        initial = np.asarray(initial_remaining)
        if (initial.shape != (venv.num_envs,) or not np.issubdtype(initial.dtype, np.integer)
                or (initial < 1).any() or (initial > 20).any() or not callable(read_kills)):
            raise ValueError('Expected per-environment initial remaining counts in1..20 and a counter reader')
        self._initial_remaining = initial.astype(np.int64, copy=True)
        self._read_kills = read_kills
        self._potential = np.zeros(venv.num_envs, dtype=np.float64)
        self._victory_returns = np.zeros(venv.num_envs, dtype=np.float64)
        super().__init__(venv, profile='baseline')

    def reset(self):
        observation = super().reset()
        self._potential.fill(0)
        self._victory_returns.fill(0)
        return observation

    def step_wait(self):
        observation, original_rewards, dones, infos = super().step_wait()
        kills = np.asarray(self._read_kills())
        if kills.shape != (self.num_envs,) or not np.issubdtype(kills.dtype, np.integer):
            raise ValueError('Expected an integer cumulative kill counter for every environment')
        rewards = np.empty(self.num_envs, dtype=np.float32)
        for index in range(self.num_envs):
            outcome = infos[index]['outcome'] if dones[index] else ''
            if not dones[index] and not 0 <= kills[index] <= self._initial_remaining[index]:
                raise ValueError('Kill counter exceeds the episode remaining-enemy contract')
            current = self._potential[index] if dones[index] else kills[index] / self._initial_remaining[index]
            rewards[index] = victory_transition(self._potential[index], current, outcome, shaping=self._shaping)
            self._victory_returns[index] += float(rewards[index])
            infos[index].update(
                reward_profile=self.objective_version,
                reward_adjustment=float(rewards[index]) - float(original_rewards[index]),
                initial_remaining_enemies=int(self._initial_remaining[index]),
                finite_horizon=True,
            )
            if dones[index]:
                infos[index]['original_time_limit_truncated'] = bool(infos[index].get('TimeLimit.truncated', False))
                infos[index]['TimeLimit.truncated'] = False
                infos[index]['episode']['r'] = float(self._victory_returns[index])
                self._potential[index] = self._victory_returns[index] = 0
            else:
                self._potential[index] = current
        return observation, rewards, dones, infos
