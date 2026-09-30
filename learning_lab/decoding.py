"""Explicit inference calibration using only learned action probabilities and masks."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO


def choose_marginal_action(probabilities, mask, fire_threshold):
    p, mask = np.asarray(probabilities, dtype=np.float64), np.asarray(mask)
    if (p.shape != (10,) or not np.isfinite(p).all() or (p < 0).any()
            or mask.shape != (10,) or mask.dtype != np.bool_ or not mask.any()):
        raise ValueError('Invalid ten-action distribution or boolean mask')
    if (isinstance(fire_threshold, bool) or not isinstance(fire_threshold, (int, float))
            or not 0 <= fire_threshold <= 1):
        raise ValueError('fire_threshold must be finite in0..1')
    p = np.where(mask, p, 0)
    if p.sum() <= 0:
        raise ValueError('No positive probability on a legal action')
    p /= p.sum()
    movement_mass = p[:5] + p[5:]
    movement = int(np.argmax(movement_mass))
    fire = bool(p[movement + 5] / movement_mass[movement] >= fire_threshold)
    proposal = movement + 5 * fire
    if mask[proposal]:
        return proposal
    return movement + 5 * (not fire)


class MarginalPolicy:
    learned_model = True

    def __init__(self, model, *, fire_threshold=.1):
        if not isinstance(model, MaskablePPO) or model.action_space.n != 10:
            raise ValueError('Marginal decoding requires a ten-action learned MaskablePPO policy')
        choose_marginal_action(np.ones(10), np.ones(10, dtype=bool), fire_threshold)
        self.model = model
        self.decoder_config = {
            'name': 'movement-marginal-and-conditional-fire-threshold-v1',
            'fire_threshold': fire_threshold,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'sampled_mode': 'unchanged-joint-distribution',
            'uses_teacher_or_game_state_rules': False,
        }

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict(self, observation, *, deterministic, action_masks):
        if not deterministic:
            return self.model.predict(observation, deterministic=False, action_masks=action_masks)
        self.model.policy.set_training_mode(False)
        tensor, vectorized = self.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(tensor, action_masks=action_masks)
            probabilities = distribution.distribution.probs.detach().cpu().numpy()
        masks = np.asarray(action_masks)
        if masks.ndim == 1:
            masks = masks[None]
        if masks.shape != probabilities.shape:
            raise ValueError('Batched masks do not match learned distributions')
        actions = np.asarray([
            choose_marginal_action(p, m, self.decoder_config['fire_threshold'])
            for p, m in zip(probabilities, masks)
        ], dtype=np.int64)
        return (actions if vectorized else actions[0]), None
