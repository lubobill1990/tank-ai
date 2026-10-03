"""NN-guided MPC: the trained ego policy replaces random rollout tails.

Random tails are the planner's noise floor - candidate scores differ mostly
by rollout luck, which is why deeper search never helped. Here steps 1..D of
every rollout are driven by the bc-ego policy (sampled, mask-legal), so a
candidate's score reflects how the position plays out under competent
continuation. Step 0 remains the candidate action under evaluation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mpc_teacher import MpcVetoTeacher, base_threat, load_student

DEFAULT_POLICY = Path(r'I:\tank-ai\runs\generalization-20261001\models\bc-ego-v2-final.zip')


class GuidedPlanner(MpcVetoTeacher):
    VERSION = 'guided-mpc-v1'

    def __init__(self, policy_path=DEFAULT_POLICY, device='cuda', **kwargs):
        super().__init__(**kwargs)
        import torch
        self.torch = torch
        self.device = device
        self.policy = load_student(policy_path, device)
        self.policy.policy.set_training_mode(False)
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(int(self.rng.integers(2**31)))

    def _tail_actions(self, stacked, masks):
        """Sampled mask-legal actions straight from the actor network.

        SB3's predict() spends ~25ms per call on wrapping; the raw forward
        for this batch is under a millisecond, so drive torch directly.
        """
        torch = self.torch
        with torch.no_grad():
            observations = torch.from_numpy(stacked).to(self.device)
            logits = self.policy.policy.get_distribution(
                observations).distribution.logits
            legal = torch.from_numpy(masks).to(self.device)
            logits = logits.masked_fill(~legal, float('-inf'))
            probs = torch.softmax(logits, dim=1)
            actions = torch.multinomial(probs, 1, generator=self._generator)
        return actions.squeeze(1).cpu().numpy()

    def plan(self, rams):
        from rust_env import NativeBatch
        L = len(rams)
        A, S = 10, self.plan_seeds
        plan_init = np.repeat(np.asarray(rams, np.uint8), A * S, axis=0)
        total = L * A * S
        seeds = [int(self.rng.integers(2**63)) for _ in range(total)]
        plan = NativeBatch(plan_init, 1 << 20, seeds=seeds)
        try:
            previous = plan.observe().copy()
            first = np.tile(np.repeat(np.arange(A), S), L)
            legal_first = plan.masks()[np.arange(total), first].copy()
            acts0 = np.where(legal_first, first, 0)
            score = np.zeros(total, np.float32)
            failed = np.zeros(total, bool)
            alive = np.ones(total, bool)
            prev_stats = np.zeros((total, 5), np.float32)
            factor = 1.0
            for d in range(self.depth):
                if d == 0:
                    acts = acts0
                else:
                    stacked = np.ascontiguousarray(
                        np.concatenate([previous, plan.obs], axis=1))
                    masks = plan.masks().copy()
                    masks[~masks.any(axis=1), 0] = True
                    acts = self._tail_actions(stacked, masks)
                previous = plan.obs.copy()
                _, rewards, dones, stats = plan.step(acts)
                sp = stats.astype(np.float32)
                deaths = sp[:, 2] - prev_stats[:, 2]
                score += factor * alive * rewards
                failed |= alive & ((deaths > 0) | (rewards <= -19.0))
                alive &= ~dones
                prev_stats = sp
                factor *= self.discount
            threat = np.zeros(total, np.float32)
            for j in np.flatnonzero(alive):
                threat[j] = base_threat(plan.ram(int(j)))
            score -= self.threat_weight * factor * threat
        finally:
            plan.close()
        score = np.where(legal_first, score, np.float32(-1e9))
        score = score.reshape(L, A, S).mean(axis=2)
        unsafe = (failed | ~legal_first).reshape(L, A, S).any(axis=2)
        return score, unsafe
