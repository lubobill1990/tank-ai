"""Wider egocentric extractor: same centering geometry, more capacity."""
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from learning_lab.ego_encoder import center_observations


class EgocentricWideCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        if observation_space.shape != (94, 26, 26) or observation_space.dtype != np.float32 \
                or features_dim != 512:
            raise ValueError('Wide egocentric CNN requires float32 (94,26,26) and 512 features')
        super().__init__(observation_space, features_dim)
        self.local_cnn = nn.Sequential(
            nn.Conv2d(94, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(),
            nn.Conv2d(96, 96, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(128 * 7 * 7, 384), nn.ReLU(),
        )
        self.global_cnn = nn.Sequential(
            nn.Conv2d(94, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(384 + 128 + 4, features_dim), nn.ReLU())

    def forward(self, observations):
        centered, scalars = center_observations(observations)
        local = self.local_cnn(centered)
        global_map = self.global_cnn(observations.detach())
        return self.combine(torch.cat((local, global_map, scalars), dim=1))
