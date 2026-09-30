"""Explicit weights-only 94->110 precision migration, with independent critic CNN.

Legacy frame0 channels0..46 map to0..46; frame1 channels47..93 map to55..101.
New offset channels47..54 and102..109 start at exactly zero weight in BOTH
encoders. Other compatible parameters are copied. Adam is newly constructed:
legacy optimizer/rollout/RNG/simulator state is never resumed.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
from stable_baselines3.common.preprocessing import preprocess_obs

from learning_lab import imitation
from policy import Grid26CNN
from within_stage.observation import OBSERVATION_VERSION, PRECISION_CHANNEL_NAMES


ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'within-stage-precision-policy-v1'
MIGRATION_FORMAT = 'within-stage-precision-weights-migration-v1'
_ENCODERS = ('features_extractor.', 'pi_features_extractor.', 'vf_features_extractor.')
_MAPPING = [
    {'old_channels': [0, 47], 'new_channels': [0, 47]},
    {'old_channels': [47, 94], 'new_channels': [55, 102]},
]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str(item.dtype).encode() + str(tuple(item.shape)).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _json_copy(value):
    return json.loads(_json_bytes(value))


def _json_hash(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def precision_schema():
    """Canonical dataset/model schema; callers put this under dataset_config['schema']."""
    if len(PRECISION_CHANNEL_NAMES) != 55:
        raise ValueError('Precision encoder must expose exactly55 channels')
    return {
        'observation_version': OBSERVATION_VERSION, 'frame_shape': [55, 26, 26],
        'observation_shape': [110, 26, 26], 'observation_dtype': 'float32',
        'n_stack': 2, 'channels_order': 'first', 'history_order': ['previous', 'current'],
        'frame_channel_names': list(PRECISION_CHANNEL_NAMES), 'actions': 10,
        'actions_dtype': 'int64', 'action_masks_shape': [10], 'action_masks_dtype': 'bool',
        'episode_ids_dtype': 'int64',
        'geometry_note': 'Current offsets with anonymous per-cell mean aggregation; not full observability.',
    }


def precision_contract():
    return {'schema': precision_schema(), 'algorithm': 'MaskablePPO', 'policy': 'CnnPolicy',
            'extractor': 'Grid26CNN', 'features_dim': 256, 'normalize_images': False,
            'net_arch': {'pi': [64, 64], 'vf': [64, 64]}, 'activation': 'Tanh',
            'share_features_extractor': False, 'actor_critic_parameter_storage_disjoint': True,
            'optimizer': 'Adam', 'legacy_optimizer_resumed': False, 'exact_simulator_resume': False}


def _sources():
    return {name: file_sha256(ROOT / name) for name in (
        'within_stage\\precision_policy.py', 'within_stage\\observation.py',
        'policy.py', 'rust_models.py', 'learning_lab\\imitation.py',
        'battle_city_env.py', 'game_state.py')}


def _validate_spaces(env, channels):
    obs, actions = env.observation_space, env.action_space
    if not isinstance(obs, spaces.Box) or obs.shape != (channels, 26, 26) \
            or obs.dtype != np.float32 or not np.all(obs.low == -1) or not np.all(obs.high == 1) \
            or not isinstance(actions, spaces.Discrete) or actions.n != 10 or actions.start != 0:
        raise ValueError(f'Expected canonical float32({channels},26,26) Box and Discrete(10)')


def _validate_policy(model, *, precision):
    _validate_spaces(model, 110 if precision else 94)
    policy = model.policy
    if type(policy) is not MaskableActorCriticCnnPolicy \
            or type(policy.features_extractor) is not Grid26CNN \
            or type(policy.pi_features_extractor) is not Grid26CNN \
            or type(policy.vf_features_extractor) is not Grid26CNN \
            or policy.normalize_images is not False or policy.activation_fn is not nn.Tanh \
            or policy.net_arch != {'pi': [64, 64], 'vf': [64, 64]} \
            or policy.features_dim != 256 \
            or policy.features_extractor is not policy.pi_features_extractor \
            or not isinstance(policy.optimizer, torch.optim.Adam):
        raise ValueError('Wrong CNN, activation, heads, normalization or optimizer architecture')
    actor, critic = policy.pi_features_extractor, policy.vf_features_extractor
    options = model.policy_kwargs
    if options.get('features_extractor_class') is not Grid26CNN \
            or options.get('normalize_images') is not False \
            or options.get('net_arch') != {'pi': [64, 64], 'vf': [64, 64]} \
            or options.get('activation_fn', nn.Tanh) is not nn.Tanh \
            or options.get('share_features_extractor', True) is not (not precision) \
            or options.get('features_extractor_kwargs', {}).get('features_dim', 256) != 256:
        raise ValueError('Serialized policy configuration disagrees with the live architecture')
    if precision:
        if policy.share_features_extractor is not False or actor is critic \
                or {p.data_ptr() for p in actor.parameters()} & {p.data_ptr() for p in critic.parameters()}:
            raise ValueError('Precision actor and critic encoders must have independent storage')
    elif policy.share_features_extractor is not True or actor is not critic:
        raise ValueError('Legacy migration requires the verified shared encoder')
    for extractor in (actor, critic):
        first = extractor.cnn[0]
        if extractor.features_dim != 256 or type(first) is not nn.Conv2d \
                or first.in_channels != (110 if precision else 94) or first.out_channels != 32 \
                or first.kernel_size != (3, 3) or first.stride != (1, 1) or first.padding != (1, 1):
            raise ValueError('Unsupported first-convolution/features architecture')
    parameters = list(policy.parameters())
    if any(p.dtype != torch.float32 or not torch.isfinite(p).all() for p in parameters):
        raise ValueError('Policy parameters must be finite float32')
    optimized = [p for group in policy.optimizer.param_groups for p in group['params']]
    if len(optimized) != len(parameters) or {id(p) for p in optimized} != {id(p) for p in parameters}:
        raise ValueError('Adam must contain the complete independent actor/critic policy')
    for state in policy.optimizer.state.values():
        if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
               for value in state.values()):
            raise ValueError('Adam state must contain finite tensors')


def validate_legacy_model(model):
    """Additional shared-encoder checks after the unchanged legacy strict loader."""
    imitation._verify_model(model)
    _validate_policy(model, precision=False)


class SpaceOnlyEnv(gym.Env):
    """Construction spaces only: invoking reset/step is an explicit error."""
    render_mode = None
    def __init__(self):
        self.observation_space = spaces.Box(-1, 1, (110, 26, 26), np.float32)
        self.action_space = spaces.Discrete(10)
    def reset(self, *, seed=None, options=None):
        raise RuntimeError('SpaceOnlyEnv cannot reset; attach a precision environment before training')
    def step(self, action):
        raise RuntimeError('SpaceOnlyEnv cannot step')


def actor_logits(model, observations):
    """Raw action-head logits, using only the actor encoder; supports gradients."""
    tensor = torch.as_tensor(observations, device=model.device)
    if tensor.dtype != torch.float32 or tensor.ndim != 4 \
            or tuple(tensor.shape[1:]) != model.observation_space.shape or not torch.isfinite(tensor).all():
        raise ValueError('Expected finite batched float32 observations matching the model space')
    policy = model.policy
    features = policy.pi_features_extractor(
        preprocess_obs(tensor, model.observation_space, normalize_images=False))
    return policy.action_net(policy.mlp_extractor.forward_actor(features))


def make_precision_model(legacy_checkpoint, *, device='cpu', learning_rate=3e-4, seed=0, env=None):
    """Return (model, migration_record), with copied weights and EMPTY new Adam state.

    No environment reset/step occurs. If env is omitted the model receives a
    SpaceOnlyEnv that cannot run. CPU equivalence is mathematical up to float
    roundoff for finite extra planes; validate CUDA numerics separately.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be uint32')
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (float, int)) \
            or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if str(device) == 'auto' or torch.device(device).type not in ('cpu', 'cuda'):
        raise ValueError('Choose an explicit CPU or CUDA device')
    if env is not None:
        _validate_spaces(env, 110)
    path = Path(legacy_checkpoint).resolve()
    if not path.is_file():
        raise ValueError('Provide an explicit verified legacy checkpoint file')
    legacy = imitation.load_imitation_checkpoint(path, device='cpu')
    validate_legacy_model(legacy)
    parent_manifest = json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    model = MaskablePPO(
        'CnnPolicy', SpaceOnlyEnv() if env is None else env, learning_rate=learning_rate,
        seed=seed, device=device, n_steps=2, batch_size=2, n_epochs=1, verbose=0,
        policy_kwargs={'features_extractor_class': Grid26CNN,
                       'features_extractor_kwargs': {'features_dim': 256},
                       'net_arch': {'pi': [64, 64], 'vf': [64, 64]},
                       'normalize_images': False, 'activation_fn': nn.Tanh,
                       'share_features_extractor': False})
    if str(device).startswith('cuda') and model.device.type != 'cuda':
        raise RuntimeError('Requested CUDA may not silently fall back to CPU')
    before, target = legacy.policy.state_dict(), model.policy.state_dict()
    if set(before) != set(target):
        raise ValueError('Legacy/precision policy parameter keys differ')
    mapped, expanded, copied = {}, [], []
    for name, value in target.items():
        source_name = ('features_extractor.' + name.split('.', 1)[1]
                       if name.startswith(_ENCODERS) else name)
        original = before[source_name]
        if name.startswith(_ENCODERS) and name.endswith('cnn.0.weight'):
            if tuple(original.shape) != (32, 94, 3, 3) or tuple(value.shape) != (32, 110, 3, 3):
                raise ValueError('Unsupported first-convolution migration shape')
            result = torch.zeros_like(value)
            result[:, :47].copy_(original[:, :47])
            result[:, 55:102].copy_(original[:, 47:94])
            mapped[name] = result
            expanded.append(name)
        else:
            if value.shape != original.shape or value.dtype != original.dtype:
                raise ValueError(f'Incompatible legacy parameter: {name}')
            mapped[name] = original.detach().to(value.device).clone()
            copied.append(name)
    model.policy.load_state_dict(mapped, strict=True)
    _validate_policy(model, precision=True)
    if model.policy.optimizer.state_dict()['state']:
        raise ValueError('Precision migration must start with fresh Adam')
    record = {
        'format': MIGRATION_FORMAT, 'kind': 'weights-only', 'contract': precision_contract(),
        'parent': {'path': str(path), 'sha256': file_sha256(path),
                   'manifest_sha256': file_sha256(path.with_suffix('.manifest.json')),
                   'manifest': parent_manifest},
        'channel_mapping': deepcopy(_MAPPING), 'initial_zero_channel_ranges': [[47, 55], [102, 110]],
        'encoder_copy': 'old shared encoder copied into independent actor and critic encoders',
        'expanded_parameters': expanded, 'copied_parameters': copied,
        'fresh_adam': True, 'legacy_optimizer_resumed': False, 'exact_simulator_resume': False,
        'optimizer_state_entries_at_creation': 0, 'learning_rate': float(learning_rate), 'seed': seed,
        'legacy_policy_sha256': state_sha256(before),
        'initial_policy_sha256': state_sha256(model.policy.state_dict()),
        'initial_actor_encoder_sha256': state_sha256(model.policy.pi_features_extractor.state_dict()),
        'initial_critic_encoder_sha256': state_sha256(model.policy.vf_features_extractor.state_dict()),
        'sources': _sources(),
        'numerics': 'Zero new weights preserve the actor mathematically for finite offsets. '
                    'CPU tests use rtol1e-5/atol2e-6; CUDA verification belongs to the caller.',
    }
    model.precision_migration_record = deepcopy(record)
    return model, record


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _validate_migration(record):
    try:
        parent = record['parent']
        if record['format'] != MIGRATION_FORMAT or record['kind'] != 'weights-only' \
                or _json_bytes(record['contract']) != _json_bytes(precision_contract()) \
                or record['sources'] != _sources() \
                or record['channel_mapping'] != _MAPPING \
                or record['initial_zero_channel_ranges'] != [[47, 55], [102, 110]] \
                or record['fresh_adam'] is not True or record['legacy_optimizer_resumed'] is not False \
                or record['exact_simulator_resume'] is not False \
                or record['optimizer_state_entries_at_creation'] != 0 \
                or parent['manifest']['format'] != imitation._FORMAT \
                or parent['manifest']['contract'] != imitation._CONTRACT \
                or parent['manifest']['sources'] != imitation._source_hashes() \
                or parent['sha256'] != parent['manifest']['checkpoint_sha256']:
            raise ValueError('Invalid precision migration/legacy-parent contract')
        for value in (parent['sha256'], parent['manifest_sha256'], record['legacy_policy_sha256'],
                      record['initial_policy_sha256'], record['initial_actor_encoder_sha256'],
                      record['initial_critic_encoder_sha256']):
            if not _valid_hash(value):
                raise ValueError('Invalid migration SHA256')
        if record['initial_actor_encoder_sha256'] != record['initial_critic_encoder_sha256']:
            raise ValueError('Initial encoder copies must be identical')
        imitation._verify_provenance(parent['manifest'])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed precision migration record') from error


def _validate_dataset_config(config):
    try:
        config = _json_copy(config)
        if not isinstance(config, dict) or _json_bytes(config['schema']) != _json_bytes(precision_schema()) \
                or not isinstance(config['datasets'], list):
            raise ValueError('Dataset config must explicitly declare the precision55/two-stack schema')
        if config.get('split') is not None and not isinstance(config['split'], dict):
            raise ValueError('Dataset split must be a mapping or null')
        for dataset in config['datasets']:
            if not isinstance(dataset, dict) or not isinstance(dataset.get('path'), str) \
                    or not dataset['path'] or not _valid_hash(dataset.get('sha256')):
                raise ValueError('Every dataset identity requires path and SHA256')
            metadata = dataset.get('metadata', {})
            for name, expected in (('frame_shape', [55, 26, 26]), ('observation_shape', [110, 26, 26]),
                                   ('observation_version', OBSERVATION_VERSION)):
                if name in metadata and metadata[name] != expected:
                    raise ValueError(f'Dataset metadata violates precision {name}')
        return config
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed precision dataset configuration') from error


def save_precision_checkpoint(model, path, *, migration_record, dataset_config):
    """Save a NEW .zip and strict sidecar; explicit dataset config is mandatory.

    For migration-only artifacts use {'schema': precision_schema(), 'datasets': [],
    'split': None, 'purpose': 'migration-only'}. Later saves retain the PRECISION
    optimizer state, not the discarded legacy Adam. No simulator resume is implied.
    """
    _validate_policy(model, precision=True)
    _validate_migration(migration_record)
    if getattr(model, 'precision_migration_record', None) != migration_record:
        raise ValueError('Model is not bound to the supplied original migration record')
    config = _validate_dataset_config(dataset_config)
    path = Path(path).resolve()
    if path.suffix.lower() != '.zip':
        raise ValueError('Precision checkpoints require an explicit .zip path')
    sidecar = path.with_suffix('.manifest.json')
    if path.exists() or sidecar.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path, exclude=['_last_obs', '_last_episode_starts', '_last_original_obs',
                              'precision_migration_record', 'precision_dataset_config'])
    manifest = {
        'format': FORMAT, 'contract': precision_contract(), 'checkpoint_file': path.name,
        'checkpoint_sha256': file_sha256(path), 'sources': _sources(),
        'migration': deepcopy(migration_record), 'migration_sha256': _json_hash(migration_record),
        'policy_sha256': state_sha256(model.policy.state_dict()),
        'precision_optimizer_sha256': state_sha256(model.policy.optimizer.state_dict()),
        'dataset_config': config, 'dataset_config_sha256': _json_hash(config),
        'num_timesteps': int(model.num_timesteps), 'ppo_epoch_updates': int(model._n_updates),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': str(torch.__version__)},
        'legacy_optimizer_resumed': False, 'precision_optimizer_saved': True, 'exact_simulator_resume': False,
    }
    _write_json(sidecar, manifest)
    return {'path': str(path), 'manifest': str(sidecar), 'sha256': manifest['checkpoint_sha256']}


def load_precision_checkpoint(path, device='cpu', *, env=None, expected_dataset_config=None):
    """Strict new-format loader; restore precision Adam, never call a legacy bypass.

    Returns the model with owned precision_migration_record and
    precision_dataset_config attributes for later explicit saves. Uses a tiny
    inference buffer; a future trainer must configure its own rollout settings.
    """
    path = Path(path).resolve()
    if env is not None:
        _validate_spaces(env, 110)
    try:
        manifest = json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
        if manifest['format'] != FORMAT \
                or _json_bytes(manifest['contract']) != _json_bytes(precision_contract()) \
                or manifest['checkpoint_file'] != path.name or manifest['sources'] != _sources() \
                or manifest['legacy_optimizer_resumed'] is not False \
                or manifest['precision_optimizer_saved'] is not True \
                or manifest['exact_simulator_resume'] is not False:
            raise ValueError('Wrong precision checkpoint schema/source/architecture')
        _validate_migration(manifest['migration'])
        config = _validate_dataset_config(manifest['dataset_config'])
        if _json_hash(config) != manifest['dataset_config_sha256'] \
                or _json_hash(manifest['migration']) != manifest['migration_sha256']:
            raise ValueError('Precision dataset/migration identity mismatch')
        if expected_dataset_config is not None \
                and _json_bytes(_validate_dataset_config(expected_dataset_config)) != _json_bytes(config):
            raise ValueError('Checkpoint dataset configuration differs from the expected contract')
        if file_sha256(path) != manifest['checkpoint_sha256']:
            raise ValueError('Precision checkpoint SHA256 mismatch')
    except (OSError, KeyError, TypeError, AttributeError) as error:
        raise ValueError('Missing or malformed precision checkpoint manifest') from error
    model = MaskablePPO.load(path, device=device, env=env,
                             custom_objects={'n_steps': 2, 'batch_size': 2, 'n_envs': 1})
    _validate_policy(model, precision=True)
    if state_sha256(model.policy.state_dict()) != manifest['policy_sha256'] \
            or state_sha256(model.policy.optimizer.state_dict()) != manifest['precision_optimizer_sha256'] \
            or model.num_timesteps != manifest['num_timesteps'] \
            or model._n_updates != manifest['ppo_epoch_updates']:
        raise ValueError('Serialized precision weights/optimizer/counters do not match manifest')
    model.precision_migration_record = deepcopy(manifest['migration'])
    model.precision_dataset_config = deepcopy(config)
    return model
