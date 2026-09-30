from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from gymnasium import spaces


class Policy:
    observation_space = spaces.Box(-1, 1, (110, 26, 26), np.float32)
    action_space = spaces.Discrete(10)

    def predict(self, observation, *, deterministic, action_masks):
        return np.asarray(0), None


class EvaluationTests(unittest.TestCase):
    def test_development_interface_refuses_final_and_unassigned_configs_before_reserving(self):
        from within_stage.evaluation import validate_development_request
        split = {'stage': 1, 'max_frames': 12000, 'development': {'all_config_indices': [2, 3]},
                 'train': {'all_config_indices': [0, 1]}, 'final': {'all_config_indices': [4, 5]}}
        for index in (0, 4, 99):
            with self.assertRaises(ValueError):
                validate_development_request(split, index, 12000)
        validate_development_request(split, 2, 12000)
        with self.assertRaises(ValueError):
            validate_development_request(split, 2, 12001)

    def test_legacy_plane_adapter_zeros_only_the_two_new_offset_blocks(self):
        from within_stage.evaluation import ObservationAdapter

        class Spy(Policy):
            def predict(self, observation, **kwargs):
                self.seen = observation.copy()
                return np.asarray(0), None

        raw = Spy()
        adapter = ObservationAdapter(raw, offsets_enabled=False)
        observation = np.arange(110 * 26 * 26, dtype=np.float32).reshape(110, 26, 26)
        saved = observation.copy()
        adapter.predict(observation, deterministic=True, action_masks=np.ones(10, bool))
        np.testing.assert_array_equal(raw.seen[:47], observation[:47])
        np.testing.assert_array_equal(raw.seen[55:102], observation[55:102])
        self.assertFalse(raw.seen[47:55].any() or raw.seen[102:].any())
        np.testing.assert_array_equal(observation, saved)

    def test_wrong_precision_shape_is_not_accepted_as_a_legacy_model(self):
        from within_stage.evaluation import ObservationAdapter
        policy = Policy()
        policy.observation_space = spaces.Box(-1, 1, (94, 26, 26), np.float32)
        with self.assertRaises(ValueError):
            ObservationAdapter(policy, offsets_enabled=True)
