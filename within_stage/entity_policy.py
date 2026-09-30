"""Fresh entity-relative MultiInputPolicy candidate; no legacy migration or trainer.

A small precision-grid CNN and two64-dimensional/four-head attention blocks
produce256 features. A single learned query pools unordered entity tokens.
No entity slot-position embeddings exist. Actor/critic extractors are separate.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import platform

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from within_stage.entity_observation import entity_schema, entity_space
from within_stage.precision_policy import file_sha256, state_sha256, _json_bytes, _json_hash


ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'within-stage-entity-policy-v1'


def entity_contract():
    return {
        'schema': entity_schema(), 'algorithm': 'MaskablePPO', 'policy': 'MultiInputPolicy',
        'extractor': 'EntityGridExtractor-v1', 'features_dim': 256,
        'grid_cnn_channels': [16, 32, 32], 'grid_embedding_dim': 128,
        'token_embedding_dim': 64, 'attention_blocks': 2, 'attention_heads': 4,
        'attention_feedforward_dim': 128, 'attention_dropout': 0.,
        'pooling': 'single-learned-query-always-unmasked',
        'entity_slot_position_embedding': False, 'key_padding_mask': 'valid==0',
        'normalize_images': False, 'net_arch': {'pi': [64, 64], 'vf': [64, 64]},
        'activation': 'Tanh', 'share_features_extractor': False, 'actions': 10,
        'fresh_initialization': True, 'legacy_checkpoint_reuse': False,
        'exact_simulator_resume': False}


def _sources():
    return {name: file_sha256(ROOT / name) for name in (
        'within_stage\\entity_policy.py', 'within_stage\\entity_observation.py',
        'within_stage\\observation.py', 'within_stage\\precision_policy.py',
        'game_state.py', 'battle_city_env.py')}


def _validate_spaces(env):
    expected = entity_space()
    if env.observation_space != expected or not isinstance(env.action_space, gym.spaces.Discrete) \
            or env.action_space.n != 10 or env.action_space.start != 0:
        raise ValueError('Expected entity dict {grid:(110,26,26),entities:(18,16)} and Discrete(10)')


class EntityGridExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        if observation_space != entity_space() or features_dim != 256:
            raise ValueError('EntityGridExtractor requires the exact v1 dict schema and256 features')
        super().__init__(observation_space, features_dim)
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(110, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU())
        self.token_embedding = nn.Linear(16, 64)
        self.query = nn.Parameter(torch.zeros(1, 1, 64))
        self.attention = nn.ModuleList([
            nn.TransformerEncoderLayer(64, 4, dim_feedforward=128, dropout=0.,
                                       activation='gelu', batch_first=True, norm_first=True)
            for _ in range(2)])
        self.token_norm = nn.LayerNorm(64)
        self.fusion = nn.Sequential(nn.Linear(128 + 64, features_dim), nn.ReLU())

    def forward(self, observations):
        grid, entities = observations['grid'], observations['entities']
        if grid.ndim != 4 or tuple(grid.shape[1:]) != (110, 26, 26) \
                or entities.shape != (len(grid), 18, 16) \
                or grid.dtype != torch.float32 or entities.dtype != torch.float32:
            raise ValueError('Entity extractor input shape/dtype mismatch')
        flags = entities[:, :, 0]
        if not ((flags == 0) | (flags == 1)).all() or not torch.isfinite(grid).all():
            raise ValueError('Entity valid flags must be0/1 and grids finite')
        valid = flags == 1
        clean = entities.masked_fill(~valid.unsqueeze(-1), 0.)
        if not torch.isfinite(clean).all():
            raise ValueError('Valid entity fields must be finite')
        embedded = self.token_embedding(clean)
        query = self.query.expand(len(grid), -1, -1)
        tokens = torch.cat((query, embedded), dim=1)
        mask = torch.cat((torch.zeros((len(grid), 1), dtype=torch.bool, device=grid.device), ~valid), dim=1)
        for block in self.attention:
            tokens = block(tokens, src_key_padding_mask=mask)
            tokens = tokens.masked_fill(mask.unsqueeze(-1), 0.)
        pooled = self.token_norm(tokens[:, 0])
        return self.fusion(torch.cat((self.grid_cnn(grid), pooled), dim=1))


class SpaceOnlyEntityEnv(gym.Env):
    render_mode = None
    def __init__(self):
        self.observation_space, self.action_space = entity_space(), gym.spaces.Discrete(10)
    def reset(self, *, seed=None, options=None):
        raise RuntimeError('SpaceOnlyEntityEnv cannot reset or simulate')
    def step(self, action):
        raise RuntimeError('SpaceOnlyEntityEnv cannot step')


def make_entity_model(*, device='cpu', seed=0, lr=3e-4, env=None):
    """Return a fresh model with empty Adam; no environment reset or step occurs."""
    if type(seed) is not int or not 0 <= seed < 2**32 or isinstance(lr, bool) \
            or not isinstance(lr, (float, int)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Use uint32 seed and positive finite lr')
    if str(device) == 'auto' or torch.device(device).type not in ('cpu', 'cuda'):
        raise ValueError('Use an explicit CPU/CUDA device')
    env = SpaceOnlyEntityEnv() if env is None else env
    _validate_spaces(env)
    model = MaskablePPO(
        'MultiInputPolicy', env, seed=seed, learning_rate=lr, device=device,
        n_steps=2, batch_size=2, n_epochs=1, verbose=0,
        policy_kwargs={'features_extractor_class': EntityGridExtractor,
                       'features_extractor_kwargs': {'features_dim': 256},
                       'normalize_images': False, 'share_features_extractor': False,
                       'net_arch': {'pi': [64, 64], 'vf': [64, 64]}, 'activation_fn': nn.Tanh})
    if str(device).startswith('cuda') and model.device.type != 'cuda':
        raise RuntimeError('Requested CUDA must not silently fall back')
    model.entity_initialization = {
        'kind': 'fresh-independent-actor-critic-entity-policy', 'seed': seed, 'learning_rate': lr,
        'sources': _sources(), 'contract': entity_contract(),
        'initial_policy_sha256': state_sha256(model.policy.state_dict()),
        'initial_adam_state_entries': len(model.policy.optimizer.state),
        'environment_interactions': 0, 'legacy_checkpoint_reuse': False}
    return model


def _validate_model(model):
    _validate_spaces(model)
    policy, options = model.policy, model.policy_kwargs
    if type(policy) is not MaskableMultiInputActorCriticPolicy \
            or type(policy.pi_features_extractor) is not EntityGridExtractor \
            or type(policy.vf_features_extractor) is not EntityGridExtractor \
            or policy.features_extractor is not policy.pi_features_extractor \
            or policy.share_features_extractor is not False or policy.normalize_images is not False \
            or policy.net_arch != {'pi': [64, 64], 'vf': [64, 64]} or policy.activation_fn is not nn.Tanh \
            or options.get('features_extractor_class') is not EntityGridExtractor \
            or options.get('share_features_extractor') is not False or options.get('normalize_images') is not False \
            or options.get('net_arch') != policy.net_arch or options.get('activation_fn') is not nn.Tanh \
            or options.get('features_extractor_kwargs') != {'features_dim': 256} \
            or not isinstance(policy.optimizer, torch.optim.Adam):
        raise ValueError('Entity policy architecture/configuration mismatch')
    for extractor in (policy.pi_features_extractor, policy.vf_features_extractor):
        if extractor.features_dim != 256 or tuple(extractor.query.shape) != (1, 1, 64) \
                or type(extractor.token_embedding) is not nn.Linear \
                or (extractor.token_embedding.in_features, extractor.token_embedding.out_features) != (16, 64) \
                or len(extractor.attention) != 2:
            raise ValueError('Entity attention/query architecture mismatch')
        for block in extractor.attention:
            if type(block) is not nn.TransformerEncoderLayer or block.norm_first is not True \
                    or block.self_attn.embed_dim != 64 or block.self_attn.num_heads != 4 \
                    or block.self_attn.batch_first is not True or block.self_attn.dropout != 0 \
                    or (block.linear1.in_features, block.linear1.out_features) != (64, 128) \
                    or (block.linear2.in_features, block.linear2.out_features) != (128, 64) \
                    or any(dropout.p != 0 for dropout in (block.dropout, block.dropout1, block.dropout2)):
                raise ValueError('Entity attention architecture differs from the two64d/four-head contract')
        for index, inputs, outputs, stride in ((0, 110, 16, 1), (2, 16, 32, 2), (4, 32, 32, 2)):
            convolution = extractor.grid_cnn[index]
            if type(convolution) is not nn.Conv2d or convolution.in_channels != inputs \
                    or convolution.out_channels != outputs or convolution.kernel_size != (3, 3) \
                    or convolution.stride != (stride, stride) or convolution.padding != (1, 1):
                raise ValueError('Entity grid CNN architecture mismatch')
    actor = {p.data_ptr() for p in policy.pi_features_extractor.parameters()}
    critic = {p.data_ptr() for p in policy.vf_features_extractor.parameters()}
    if actor & critic:
        raise ValueError('Actor/critic extractors must have disjoint parameter storage')
    if any(p.dtype != torch.float32 or not torch.isfinite(p).all() for p in policy.parameters()):
        raise ValueError('Entity policy weights must be finite float32')
    all_parameters = list(policy.parameters())
    optimizer_parameters = [p for group in policy.optimizer.param_groups for p in group['params']]
    if len(all_parameters) != len(optimizer_parameters) \
            or {id(p) for p in all_parameters} != {id(p) for p in optimizer_parameters}:
        raise ValueError('Entity Adam must own every policy parameter exactly once')
    for state in policy.optimizer.state.values():
        if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all() for value in state.values()):
            raise ValueError('Entity Adam state must be finite')


def _dataset_config(config):
    try:
        config = json.loads(_json_bytes(config))
        if _json_bytes(config['schema']) != _json_bytes(entity_schema()) \
                or not isinstance(config['datasets'], list):
            raise ValueError('Explicit entity dataset schema/identities are required')
        for identity in config['datasets']:
            if not isinstance(identity.get('path'), str) or not identity['path'] \
                    or not isinstance(identity.get('sha256'), str) or len(identity['sha256']) != 64 \
                    or any(c not in '0123456789abcdef' for c in identity['sha256']):
                raise ValueError('Each dataset requires a path and SHA256')
        return config
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed entity dataset configuration') from error


def _write(path, value):
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def save_entity_checkpoint(model, path, *, dataset_config):
    """Save a fresh .zip plus strict schema/source/weights/optimizer/provenance sidecar."""
    _validate_model(model)
    creation = deepcopy(getattr(model, 'entity_initialization', None))
    if not isinstance(creation, dict) or creation.get('sources') != _sources() \
            or _json_bytes(creation.get('contract')) != _json_bytes(entity_contract()) \
            or creation.get('legacy_checkpoint_reuse') is not False \
            or creation.get('initial_adam_state_entries') != 0:
        raise ValueError('Model lacks its original fresh entity initialization identity')
    config = _dataset_config(dataset_config)
    path = Path(path).resolve()
    if path.suffix != '.zip':
        raise ValueError('Use an explicit .zip checkpoint path')
    sidecar = path.with_suffix('.manifest.json')
    if path.exists() or sidecar.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path, exclude=['_last_obs', '_last_episode_starts', '_last_original_obs',
                              'entity_initialization', 'entity_dataset_config'])
    manifest = {
        'format': FORMAT, 'contract': entity_contract(), 'checkpoint_file': path.name,
        'checkpoint_sha256': file_sha256(path), 'sources': _sources(),
        'policy_sha256': state_sha256(model.policy.state_dict()),
        'optimizer_sha256': state_sha256(model.policy.optimizer.state_dict()),
        'initialization': creation, 'initialization_sha256': _json_hash(creation),
        'dataset_config': config, 'dataset_config_sha256': _json_hash(config),
        'num_timesteps': int(model.num_timesteps), 'ppo_epoch_updates': int(model._n_updates),
        'exact_simulator_resume': False, 'versions': {
            'python': platform.python_version(), 'numpy': np.__version__, 'torch': str(torch.__version__)},
        'warning': 'Entity relational bias is an experimental representation, not demonstrated game competence.',
    }
    _write(sidecar, manifest)
    return {'path': str(path), 'manifest': str(sidecar), 'sha256': manifest['checkpoint_sha256']}


def load_entity_checkpoint(path, device='cpu', *, env=None, expected_dataset_config=None):
    """Strict trusted-local new-format load; no legacy loader/schema bypass."""
    path = Path(path).resolve()
    if env is not None:
        _validate_spaces(env)
    try:
        manifest = json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
        if manifest['format'] != FORMAT or _json_bytes(manifest['contract']) != _json_bytes(entity_contract()) \
                or manifest['sources'] != _sources() or manifest['checkpoint_file'] != path.name \
                or manifest['exact_simulator_resume'] is not False:
            raise ValueError('Entity checkpoint schema/architecture/source mismatch')
        creation = manifest['initialization']
        if creation['sources'] != _sources() or creation['contract'] != entity_contract() \
                or creation['legacy_checkpoint_reuse'] is not False or creation['initial_adam_state_entries'] != 0 \
                or _json_hash(creation) != manifest['initialization_sha256']:
            raise ValueError('Entity initialization provenance mismatch')
        config = _dataset_config(manifest['dataset_config'])
        if _json_hash(config) != manifest['dataset_config_sha256']:
            raise ValueError('Entity dataset configuration hash mismatch')
        if expected_dataset_config is not None \
                and _json_bytes(_dataset_config(expected_dataset_config)) != _json_bytes(config):
            raise ValueError('Unexpected entity dataset provenance')
        if file_sha256(path) != manifest['checkpoint_sha256']:
            raise ValueError('Entity checkpoint SHA256 mismatch')
    except (KeyError, TypeError, AttributeError, OSError) as error:
        raise ValueError('Missing or malformed entity checkpoint manifest') from error
    model = MaskablePPO.load(path, device=device, env=env,
                             custom_objects={'n_steps': 2, 'batch_size': 2, 'n_envs': 1})
    _validate_model(model)
    if state_sha256(model.policy.state_dict()) != manifest['policy_sha256'] \
            or state_sha256(model.policy.optimizer.state_dict()) != manifest['optimizer_sha256'] \
            or model.num_timesteps != manifest['num_timesteps'] \
            or model._n_updates != manifest['ppo_epoch_updates']:
        raise ValueError('Serialized entity policy/Adam/counter integrity mismatch')
    model.entity_initialization = deepcopy(creation)
    model.entity_dataset_config = deepcopy(config)
    return model
