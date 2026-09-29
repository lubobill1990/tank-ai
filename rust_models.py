"""Versioned extractors for the bounded experimental Rust learning study.

History means the latest 16 raw decision observations (about one second at
15 decisions/s), not recurrent memory. VecFrameStack owns chronological stacking
and episode resets. Rollout inference and PPO replay use exactly the same input;
these extractors retain neither hidden state nor a key/value cache.
"""
import math

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from policy import Grid26CNN


MODEL_NAMES = ("cnn", "spatial", "history")
_VERSION = "rust-models-v1"
_CHANNELS_PER_FRAME = 47


def model_spec(name: str) -> dict:
    """Return a fresh JSON-safe architecture/history contract."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown Rust model {name!r}; expected one of {MODEL_NAMES}")
    return {"name": name, "n_stack": 16 if name == "history" else 2,
            "features_dim": 256, "version": _VERSION}


def extractor_class(name: str):
    """Return a module-level class so standard MaskablePPO serialization works."""
    model_spec(name)
    return {"cnn": Grid26CNN, "spatial": SpatialTransformer,
            "history": HistoryTransformer}[name]


def _positions(length, width):
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, width, 2, dtype=torch.float32)
                            * (-math.log(10000.0) / width))
    result = torch.empty(length, width)
    result[:, 0::2] = torch.sin(positions * frequencies)
    result[:, 1::2] = torch.cos(positions * frequencies)
    return result


def _stem(channels):
    return nn.Sequential(
        nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
    )


def _layers():
    return nn.ModuleList([
        nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256,
                                   dropout=0.0, activation="gelu", batch_first=True,
                                   norm_first=True)
        for _ in range(2)
    ])


def _check_space(observation_space, stack, features_dim):
    expected = (stack * _CHANNELS_PER_FRAME, 26, 26)
    if observation_space.shape != expected:
        raise ValueError(f"Expected channels-first observation space {expected}, "
                         f"got {observation_space.shape}")
    if features_dim != 256:
        raise ValueError("Versioned Rust transformer extractors require features_dim=256")


def _check_input(observations, stack):
    expected = (stack * _CHANNELS_PER_FRAME, 26, 26)
    if observations.ndim != 4 or tuple(observations.shape[1:]) != expected:
        raise ValueError(f"Expected batched observations (batch, {expected}), "
                         f"got {tuple(observations.shape)}")


class SpatialTransformer(BaseFeaturesExtractor):
    """Two-frame CNN stem, 169 positioned spatial tokens, mean-pooled attention."""
    def __init__(self, observation_space, features_dim=256):
        _check_space(observation_space, 2, features_dim)
        super().__init__(observation_space, features_dim)
        self.stem = _stem(94)
        axis = _positions(13, 32)
        positions = torch.cat((axis[:, None, :].expand(13, 13, 32),
                               axis[None, :, :].expand(13, 13, 32)), dim=-1)
        self.register_buffer("positions", positions.reshape(1, 169, 64))
        self.layers = _layers()
        self.norm = nn.LayerNorm(64)
        self.projection = nn.Sequential(nn.Linear(64, features_dim), nn.ReLU())

    def forward(self, observations):
        _check_input(observations, 2)
        tokens = self.stem(observations).flatten(2).transpose(1, 2) + self.positions
        for layer in self.layers:
            tokens = layer(tokens)
        return self.projection(self.norm(tokens).mean(dim=1))


class HistoryTransformer(BaseFeaturesExtractor):
    """Causal attention over exactly 16 independently encoded raw frames.

    A canonical real frame contains the base landmark, so an entirely zero frame
    unambiguously identifies VecFrameStack prefix padding. Padding contributes no
    positional features and cannot become a key for a valid query.
    """
    def __init__(self, observation_space, features_dim=256):
        _check_space(observation_space, 16, features_dim)
        super().__init__(observation_space, features_dim)
        self.stem = _stem(_CHANNELS_PER_FRAME)
        self.frame_projection = nn.Sequential(
            # For 13 -> 3, these are exactly the adaptive bins [0:5, 4:9, 8:13],
            # but fixed pooling also supports deterministic CUDA backward.
            nn.AvgPool2d(kernel_size=5, stride=4), nn.Flatten(),
            nn.Linear(64 * 3 * 3, 64), nn.GELU(),
        )
        self.register_buffer("positions", _positions(16, 64).unsqueeze(0))
        self.register_buffer("_causal", torch.ones(16, 16, dtype=torch.bool).tril(),
                             persistent=False)
        self.register_buffer("_diagonal", torch.eye(16, dtype=torch.bool), persistent=False)
        self.layers = _layers()
        self.norm = nn.LayerNorm(64)
        self.projection = nn.Sequential(nn.Linear(64, features_dim), nn.ReLU())

    def encode_sequence(self, observations):
        """Return (batch, 16, features_dim) causal features, zero at padded steps."""
        _check_input(observations, 16)
        batch = observations.shape[0]
        frames = observations.reshape(batch, 16, _CHANNELS_PER_FRAME, 26, 26)
        valid = frames.ne(0).flatten(2).any(dim=2)
        tokens = self.frame_projection(self.stem(
            frames.reshape(batch * 16, _CHANNELS_PER_FRAME, 26, 26)
        )).reshape(batch, 16, 64)
        invalid = ~valid.unsqueeze(-1)
        tokens = (tokens + self.positions).masked_fill(invalid, 0)
        allowed = self._causal.unsqueeze(0) & valid.unsqueeze(1)
        # Empty-prefix query rows need one finite softmax entry. This self-only
        # fallback is never visible to real queries, and its output is discarded.
        allowed = allowed | (~allowed.any(dim=-1, keepdim=True) & self._diagonal)
        blocked = (~allowed).repeat_interleave(4, dim=0)
        for layer in self.layers:
            tokens = layer(tokens, src_mask=blocked).masked_fill(invalid, 0)
        return self.projection(self.norm(tokens)).masked_fill(invalid, 0)

    def forward(self, observations):
        return self.encode_sequence(observations)[:, -1]
