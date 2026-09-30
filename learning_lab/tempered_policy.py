"""Explicit lower-temperature sampling; no teacher or game-state rules."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO


def tempered_probabilities(probabilities, temperature):
    p = np.asarray(probabilities, dtype=np.float64)
    if (p.shape != (10,) or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0
            or isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError('Expected ten finite nonnegative probabilities and a positive temperature')
    positive = p > 0
    logits = np.log(p[positive]) / temperature
    logits -= logits.max()
    q = np.zeros(10, dtype=np.float64)
    q[positive] = np.exp(logits)
    return q / q.sum()


class TemperedPolicy:
    learned_model = True

    def __init__(self, model, temperature):
        if not isinstance(model, MaskablePPO) or model.action_space.n != 10:
            raise ValueError('Temperature calibration requires a ten-action learned policy')
        tempered_probabilities(np.ones(10), temperature)
        self.model, self.temperature = model, float(temperature)
        self.decoder_config = {
            'name': 'temperature-sampling-v1', 'temperature': self.temperature,
            'deterministic_mode': 'unchanged-argmax',
            'uses_teacher_or_game_state_rules': False,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict(self, observation, *, deterministic, action_masks):
        if deterministic:
            return self.model.predict(observation, deterministic=True, action_masks=action_masks)
        self.model.policy.set_training_mode(False)
        tensor, vectorized = self.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(tensor, action_masks=action_masks)
            logits = distribution.distribution.logits / self.temperature
            masks = torch.as_tensor(action_masks, dtype=torch.bool, device=logits.device).reshape(logits.shape)
            logits = logits.masked_fill(~masks, float('-inf'))
            actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        return (actions if vectorized else actions[0]), None
