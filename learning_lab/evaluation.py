"""Budgeted development evaluation; sealed stages are deliberately inaccessible."""
import hashlib
from pathlib import Path
import time
import traceback

import numpy as np
from gymnasium import spaces
from sb3_contrib import MaskablePPO

from learning_lab.budget import BudgetExhausted
from rust_learning import _hash, _write_new, runtime_signature
from rust_resources import ResourceMonitor
from rust_validation import evaluate_suite

SEALED_STAGES = frozenset((8, 18, 28))


class BaselinePolicy:
    learned_model = False
    observation_space = spaces.Box(-1, 1, (94, 26, 26), np.float32)
    action_space = spaces.Discrete(10)

    def __init__(self, name):
        if name not in ('up_fire', 'uniform_random', 'noop'):
            raise ValueError('Unknown primitive baseline')
        self.name = name

    def predict(self, observation, *, deterministic, action_masks):
        mask = np.asarray(action_masks)
        if mask.shape != (10,) or mask.dtype != np.bool_ or not mask.any():
            raise ValueError('Expected a nonempty ten-action boolean mask')
        allowed = np.flatnonzero(mask)
        if self.name == 'up_fire':
            for action in (6, 5, 0):
                if mask[action]:
                    return np.asarray(action), None
            return np.asarray(allowed[0]), None
        if self.name == 'noop':
            return np.asarray(allowed[0]), None
        # Deterministic argmax of a uniform distribution breaks ties by index.
        action = allowed[0] if deterministic else np.random.choice(allowed)
        return np.asarray(action), None


class _CountedPolicy:
    def __init__(self, model, limit, deadline):
        self.model, self.limit, self.deadline = model, limit, deadline
        self.attempts = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict(self, *args, **kwargs):
        if self.attempts >= self.limit or time.monotonic() >= self.deadline:
            raise BudgetExhausted('Evaluation interaction or wall-time allowance exhausted')
        self.attempts += 1
        return self.model.predict(*args, **kwargs)


def evaluate_candidate(model, *, model_name, output, budget, stages=(1, 2, 3),
                       seeds=(0, 1), max_frames=12000):
    stages, seeds = tuple(stages), tuple(seeds)
    if not stages or set(stages) - {1, 2, 3}:
        raise ValueError('Development evaluation is restricted to stages1/2/3; sealed stages stay unused')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Expected distinct nonnegative development seeds')
    if type(max_frames) is not int or not 1 <= max_frames <= 12000:
        raise ValueError('max_frames must be in1..12000; shorter evaluations are smoke only')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # One decision cannot consume less than one frame. Reserve this conservative
    # upper bound; settlement releases unused decisions only after a normal return.
    allowance = 4 * len(stages) * len(seeds) * max_frames + 24 * len(seeds) * min(600, max_frames)
    label = 'eval-' + hashlib.sha256(str(output).encode()).hexdigest()[:16] + '-' + output.stem
    token = budget.reserve(label, allowance)
    deadline = time.monotonic() + budget.snapshot()['remaining_seconds']
    counted = _CountedPolicy(model, allowance, deadline)
    result = {
        'lab_evaluation_version': 'development-v1', 'role': 'development-not-confirmation',
        'learned_model': isinstance(model, MaskablePPO),
        'baseline': model.name if isinstance(model, BaselinePolicy) else None,
        'reservation': token, 'status': 'running', 'stages': list(stages), 'seeds': list(seeds),
        'evaluation_reward': 'Unchanged sparse diagnostic returns; task outcomes are authoritative.',
    }
    try:
        resources = ResourceMonitor(output.with_suffix('.resources.jsonl'),
                                    device=str(getattr(model, 'device', 'cpu')).split(':')[0])
        resources.check('evaluation-start', force=True)
        result['provenance'] = {
            'runtime': runtime_signature(),
            'lab_sources': {name: _hash(Path(__file__).parent / name)
                            for name in ('evaluation.py', 'budget.py')},
        }
        result['metrics'] = evaluate_suite(
            counted, model_name=model_name, stages=stages, seeds=seeds,
            max_frames=max_frames, deadline=deadline,
            resource_check=lambda: resources.check('evaluation'))
        result.update(status='completed', decision_attempts=counted.attempts)
        _write_new(output, result)
    except Exception as error:
        result.update(status='failed', decision_attempts=counted.attempts,
                      error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        _write_new(output.with_suffix('.failure.json'), result)
        raise
    finally:
        budget.settle(token, counted.attempts)
    return result
