"""Observation-only local/global CNN, with fixed input-only translation.

The centroid is rounded to its nearest grid point (half upward), avoiding
parity-dependent nearest-sampling ties. Both frames use the same anchor.
Geometry and sampling are detached; learnable convolutions still backpropagate.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


@torch.no_grad()
def center_observations(observations):
    """Return centered (B,94,25,25) and observed normalized XY/fire/lives (B,4)."""
    if not isinstance(observations, torch.Tensor) or observations.ndim != 4 \
            or observations.shape[1:] != (94, 26, 26) or observations.shape[0] == 0 \
            or observations.dtype != torch.float32:
        raise ValueError("Expected nonempty float32 (B,94,26,26) observations")
    frames = observations.detach()
    current = frames[:, 52].clamp(0, 1)
    previous = frames[:, 5].clamp(0, 1)
    current_count = current.sum(dim=(1, 2))
    previous_count = previous.sum(dim=(1, 2))
    player = torch.where((current_count > 0)[:, None, None], current, previous)
    count = player.sum(dim=(1, 2)).clamp_min(1)
    axis = torch.arange(26, device=frames.device, dtype=frames.dtype)
    x = (player * axis[None, None, :]).sum(dim=(1, 2)) / count
    y = (player * axis[None, :, None]).sum(dim=(1, 2)) / count
    present = (current_count > 0) | (previous_count > 0)
    x = torch.where(present, x, torch.full_like(x, 12.5))
    y = torch.where(present, y, torch.full_like(y, 12.5))
    offset = torch.arange(-12, 13, device=frames.device, dtype=frames.dtype)
    oy, ox = torch.meshgrid(offset, offset, indexing="ij")
    sample_x = torch.floor(x + .5)[:, None, None] + ox[None]
    sample_y = torch.floor(y + .5)[:, None, None] + oy[None]
    grid = torch.stack((sample_x * (2 / 25) - 1, sample_y * (2 / 25) - 1), dim=-1)
    centered = F.grid_sample(frames, grid, mode="nearest", padding_mode="zeros",
                             align_corners=True)
    scalars = torch.stack((x * (2 / 25) - 1, y * (2 / 25) - 1,
                           frames[:, 71, 0, 0], frames[:, 67, 0, 0]), dim=1)
    return centered, scalars


class EgocentricGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        if observation_space.shape != (94, 26, 26) or observation_space.dtype != np.float32 \
                or features_dim != 256:
            raise ValueError("Egocentric CNN requires float32 (94,26,26) and 256 features")
        super().__init__(observation_space, features_dim)
        self.crop_size = 25
        self.local_cnn = nn.Sequential(
            nn.Conv2d(94, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 192), nn.ReLU(),
        )
        self.global_cnn = nn.Sequential(
            nn.Conv2d(94, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.ReLU(),
        )
        self.combine = nn.Sequential(nn.Linear(192 + 64 + 4, features_dim), nn.ReLU())

    def forward(self, observations):
        centered, scalars = center_observations(observations)
        local = self.local_cnn(centered)
        global_map = self.global_cnn(observations.detach())
        return self.combine(torch.cat((local, global_map, scalars), dim=1))
