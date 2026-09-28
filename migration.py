"""Reviewed weights-only expansion or explicit same-layout full-state continuation."""
import copy
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
from stable_baselines3.common.save_util import load_from_zip_file
from torch import nn

from experiments import RECOVERY_RUNTIME_FILES, RUNTIME_FILES, checkpoint_manifest, validate_manifest
from policy import Grid26CNN


V3_SOURCE_HASHES = {
    "battle_city_env.py": "5d35e4e52113ae06ee2b75bc51b3f386d27490a20378fbac27114384ef75f319",
    "eval.py": "3366688012b4239f2026ea349004ee40eb683ee2c533f2838f8ba4b30b3b838c",
    "experiments.py": "4463ebb58550350d72ac5492ba943a0ada037186efe614fe068727fd46eb4721",
    "game_state.py": "d9ea6739d9653b4502aba5dfe12f2fd3b5ec56ac57fd03be08fbc6dc793b31d6",
    "heuristic.py": "6eddf3f3aa4b6cfab0d7492141caf8bc44a32f3f9ded2b9536e5621dee00ca82",
    "policy.py": "b9a5d30bc3e2f0bd591d84ded7f52d9b470a0ce7578244d0574e6abf64c64b19",
    "recordings.py": "bd99a4b532cfb020fe861c85e62bfa96a1e22c984ea6bb5142b4c1e096ae797d",
    "train.py": "308e65027686c2828db9b12450479345b46784e9b873a1af767add9d6b4cd9a2",
    "watch.py": "ccb7b02e73955f254e5d3b458a45b2b3f54567f894548d927280350751159eca",
}
SOURCE_HASHES = {
    "battle_city_env.py": "68777edd5614bf3343bfba7ce9666eb25d3bd6217c2c38f39acce02e49ed421c",
    "eval.py": "3366688012b4239f2026ea349004ee40eb683ee2c533f2838f8ba4b30b3b838c",
    "experiments.py": "f81ed9d20cd6bbf45a8b7bde3a6309de7e3e62e3e5fcc4d3161f6fe8272b0352",
    "game_state.py": "aac9b13995d64932d41da39a1ddad168cc86879c204612ed2ae416cd49f621f7",
    "heuristic.py": "9318dffa324891c2d54404f3cb3004ca3779f56f73e5dd9d6f574943f2cefe1e",
    "migration.py": "97cbf0d543f4525d9a4f9ae369164731afade76710a7af9766b17ffae8e8c94a",
    "policy.py": "b9a5d30bc3e2f0bd591d84ded7f52d9b470a0ce7578244d0574e6abf64c64b19",
    "recordings.py": "dbf521d152e96f1fe5977d37845fa6d3e0b3daec703af60aeca8022d4d9c11fe",
    "train.py": "318e497daaf49b9a27f466492b6e8b9ff4a187215deb7387ef4d2e851e977bd6",
    "watch.py": "ccb7b02e73955f254e5d3b458a45b2b3f54567f894548d927280350751159eca",
}
V5_SOURCE_HASHES = {
    "battle_city_env.py": "6fa10fe40c4284469ecd93c29b5cccbde6324dae025b00e0428335ac93491d4a",
    "eval.py": "3366688012b4239f2026ea349004ee40eb683ee2c533f2838f8ba4b30b3b838c",
    "experiments.py": "31eea375337093cf67054baf4b8207237dd6e9a6190d84696164901d70d224b8",
    "game_state.py": "8553d9a163760c4b7b2d91e31f2e796cb69f47eff9b5602f08d3c288db593a52",
    "heuristic.py": "3170dfa8391a7f29c806cfe3e29c02100e11e0be34ea95a15532c2b9c6c8fda8",
    "migration.py": "1e616f06315a4ff33fabaff4b8dd36ce143822e5d30afe6b61923a6692f148c3",
    "policy.py": "b9a5d30bc3e2f0bd591d84ded7f52d9b470a0ce7578244d0574e6abf64c64b19",
    "recordings.py": "dbf521d152e96f1fe5977d37845fa6d3e0b3daec703af60aeca8022d4d9c11fe",
    "train.py": "318e497daaf49b9a27f466492b6e8b9ff4a187215deb7387ef4d2e851e977bd6",
    "watch.py": "ccb7b02e73955f254e5d3b458a45b2b3f54567f894548d927280350751159eca",
}
V6_SOURCE_HASHES = {
    "battle_city_env.py": "27424310c85daa186721eadf07114edfee55fc5324000a6ef2f90b5b015a07cf",
    "eval.py": "3366688012b4239f2026ea349004ee40eb683ee2c533f2838f8ba4b30b3b838c",
    "experiments.py": "4c63d2482ececf214301bf094815ddbc538c30a6c3f5f61ac8fd0eeca5346b26",
    "game_state.py": "8553d9a163760c4b7b2d91e31f2e796cb69f47eff9b5602f08d3c288db593a52",
    "heuristic.py": "a63c1b321149e0b71e4fe0e9c574355c5cc10c55cf227f96bcbf94f7637238af",
    "migration.py": "4765fa8bb52354bfeaa2a9d9f324da5252730c996a74e9c6c586bcba9d11b413",
    "policy.py": "b9a5d30bc3e2f0bd591d84ded7f52d9b470a0ce7578244d0574e6abf64c64b19",
    "recordings.py": "dbf521d152e96f1fe5977d37845fa6d3e0b3daec703af60aeca8022d4d9c11fe",
    "train.py": "a5e67d7b92ff2e500e62b8cd650b71d89649b5c7418d5d54b83cc1d81bf41f6c",
    "watch.py": "ccb7b02e73955f254e5d3b458a45b2b3f54567f894548d927280350751159eca",
}
ROM_SHA256 = "a869aead5b6957fc62002ff9636e048cc34baf0100d629b07dc51aa18f220c0c"
OLD_CHANNEL_NAMES = (
    "brick", "steel", "water", "ice", "base",
    "player", "player_fx", "player_fy", "enemy", "enemy_fx", "enemy_fy", "armor",
    "pbullet", "pbullet_vx", "pbullet_vy", "ebullet", "ebullet_vx", "ebullet_vy",
    "remaining", "spawn_left", "lives", "power", "shield", "freeze", "fire_ready",
    "item_shield", "item_freeze", "item_shovel", "item_star", "item_grenade", "item_life",
    "last_hunt", "last_protect", "last_dodge", "last_counter", "last_item",
    "time_left", "objective_combat", "objective_defense",
)
NEW_CHANNEL_NAMES = OLD_CHANNEL_NAMES + (
    "brick_tl", "brick_tr", "brick_bl", "brick_br", "player_offset_x", "player_offset_y",
)
TACTICS_CHANNEL_NAMES = NEW_CHANNEL_NAMES + ("bullet_danger", "enemy_drops_item")
EXTRACTORS = ("features_extractor", "pi_features_extractor", "vf_features_extractor")
INPUT_WEIGHTS = tuple(f"{name}.cnn.0.weight" for name in EXTRACTORS)


def load_recovery_weights(path, signature, nstack=2):
    """Load only reviewed v6 weights for a fresh v7 comparison; not continuation."""
    if type(nstack) is not int or nstack != 2:
        raise ValueError("recovery requires the reviewed two-frame layout")
    manifest = checkpoint_manifest(path)
    expected = {
        "env_version": "grid26-events-v6-boundary", "obs_channels": 47,
        "channel_names": list(TACTICS_CHANNEL_NAMES),
        "observation_shape": [47, 26, 26], "observation_dtype": "float32",
        "stacked_observation_shape": [94, 26, 26], "stack_axis": "first",
        "action_space": {"type": "Discrete", "shape": [], "n": 15},
        "source_sha256": V6_SOURCE_HASHES, "rom_sha256": ROM_SHA256,
    }
    validate_manifest(manifest, expected)
    hashes = signature.get("source_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(RECOVERY_RUNTIME_FILES) \
            or any(not isinstance(value, str) or len(value) != 64
                   or any(c not in "0123456789abcdef" for c in value) for value in hashes.values()):
        raise ValueError("invalid recovery destination source hashes")
    allowed = {"game_state.py", "battle_city_env.py", "experiments.py", "migration.py"}
    if any(hashes[name] != value for name, value in V6_SOURCE_HASHES.items() if name not in allowed):
        raise ValueError("unreviewed recovery destination source changes")
    if signature.get("reward_profile") not in ("shaped", "sparse"):
        raise ValueError("invalid recovery reward profile")
    target = dict(expected, env_version="grid26-events-v7-clean-terrain",
                  source_sha256=hashes, reward_profile=signature["reward_profile"])
    if target != signature:
        raise ValueError("incompatible recovery observation/runtime layout")
    source = MaskablePPO.load(path, device="cpu", env=None)
    _validate_architecture(source, 94)
    _, parameters, _ = load_from_zip_file(path, device="cpu", load_data=False)
    if not _same_saved_state(parameters.get("policy"), source.policy.state_dict()):
        raise ValueError("incompatible recovery policy tensors or shared aliases")
    if manifest.get("timesteps") != source.num_timesteps:
        raise ValueError("recovery source timestep mismatch")
    provenance = {
        "method": "v6-to-v7-clean-terrain-weights-only-comparison",
        "source_checkpoint": str(Path(path).resolve()),
        "source_checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_runtime": copy.deepcopy(manifest["runtime"]),
        "source_num_timesteps": source.num_timesteps,
        "source_n_updates": source._n_updates,
        "reset": {"optimizer": True, "num_timesteps": 0, "n_updates": 0,
                  "curriculum_progress": {"combat": 0, "defense": 0, "full": 0}},
    }
    for key in ("warm_start", "continuation"):
        if key in manifest:
            provenance["source_" + key] = copy.deepcopy(manifest[key])
    return source, provenance


def validate_runtime_continuation_manifest(manifest, signature, *, nstack, resume_config=None):
    """Allow only the reviewed v5 -> v6 boundary fix, not a generic runtime bypass."""
    if type(nstack) is not int or nstack != 2:
        raise ValueError("runtime continuation requires the reviewed nstack=2 layout")
    expected = {
        "env_version": "grid26-events-v5-tactics", "obs_channels": 47,
        "channel_names": list(TACTICS_CHANNEL_NAMES),
        "observation_shape": [47, 26, 26], "observation_dtype": "float32",
        "stacked_observation_shape": [94, 26, 26], "stack_axis": "first",
        "action_space": {"type": "Discrete", "shape": [], "n": 15},
        "source_sha256": V5_SOURCE_HASHES, "rom_sha256": ROM_SHA256,
    }
    validate_manifest(manifest, expected, resume_config=resume_config)
    if manifest["training"]["nstack"] != nstack:
        raise ValueError("runtime continuation cannot change nstack")
    hashes = signature.get("source_sha256") if isinstance(signature, dict) else None
    if not isinstance(hashes, dict) or set(hashes) != set(V5_SOURCE_HASHES) \
            or set(hashes) != set(RUNTIME_FILES) or any(
                not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value) for value in hashes.values()
            ):
        raise ValueError("incompatible runtime continuation destination sources")
    allowed = {"migration.py", "train.py", "experiments.py", "heuristic.py", "battle_city_env.py"}
    if any(hashes[name] != value for name, value in V5_SOURCE_HASHES.items() if name not in allowed):
        raise ValueError("unreviewed runtime continuation destination source changes")
    target = dict(expected, env_version="grid26-events-v6-boundary", source_sha256=hashes)
    if signature != target:
        raise ValueError("incompatible runtime continuation destination runtime or channel layout")


def _same_saved_state(saved, loaded):
    if type(saved) is not type(loaded):
        return False
    if isinstance(saved, torch.Tensor):
        return saved.dtype == loaded.dtype and torch.equal(saved.cpu(), loaded.cpu())
    if isinstance(saved, dict):
        return saved.keys() == loaded.keys() and all(
            _same_saved_state(value, loaded[key]) for key, value in saved.items())
    if isinstance(saved, (list, tuple)):
        return len(saved) == len(loaded) and all(
            _same_saved_state(left, right) for left, right in zip(saved, loaded))
    return saved == loaded


def load_runtime_continuation(path, signature, *, nstack, device, resume_config=None):
    """Return (full PPO, source manifest, provenance), preserving learned/curriculum state.

    No environment is attached and no checkpoint is rewritten. The caller attaches a
    fresh environment with set_env(force_reset=True) and copies the source progress
    and warm_start lineage into its new run; ordinary load_checkpoint stays strict.
    """
    manifest = checkpoint_manifest(path)
    validate_runtime_continuation_manifest(manifest, signature, nstack=nstack,
                                          resume_config=resume_config)
    _, parameters, _ = load_from_zip_file(path, device="cpu", load_data=False)
    if not isinstance(parameters, dict) or set(parameters) != {"policy", "policy.optimizer"}:
        raise ValueError("runtime continuation requires complete policy and optimizer state")
    source = MaskablePPO.load(path, device=device, env=None, force_reset=True)
    _validate_architecture(source, 94)
    if not _same_saved_state(parameters["policy"], source.policy.state_dict()):
        raise ValueError("incompatible continuation checkpoint tensors or shared-extractor aliases")
    if type(source.policy.optimizer) is not torch.optim.Adam or not _same_saved_state(
        parameters["policy.optimizer"], source.policy.optimizer.state_dict()
    ):
        raise ValueError("incompatible continuation optimizer state")
    if type(source.num_timesteps) is not int or source.num_timesteps < 0 \
            or type(manifest.get("timesteps")) is not int \
            or manifest["timesteps"] != source.num_timesteps:
        raise ValueError("runtime continuation source timestep provenance mismatch")
    progress = manifest.get("progress")
    if not isinstance(progress, dict) or set(progress) != {"combat", "defense", "full"} or any(
        type(value) is not int or value < 0 for value in progress.values()
    ) or sum(progress.values()) != source.num_timesteps:
        raise ValueError("runtime continuation has invalid curriculum progress")
    if type(source._n_updates) is not int or source._n_updates < 0 or (
        source.num_timesteps > 0 and (
            source._n_updates == 0
            or len(source.policy.optimizer.state) != len(list(source.policy.parameters()))
        )
    ):
        raise ValueError("runtime continuation requires an optimized source checkpoint")
    if source.num_timesteps > 0:
        for group in source.policy.optimizer.param_groups:
            moments = {"exp_avg", "exp_avg_sq"}
            if group["amsgrad"]:
                moments.add("max_exp_avg_sq")
            for parameter in group["params"]:
                state = source.policy.optimizer.state.get(parameter, {})
                step = state.get("step")
                if set(state) != moments | {"step"} or not isinstance(step, torch.Tensor) \
                        or step.numel() != 1 or not torch.isfinite(step).item() or step.item() <= 0 \
                        or any(not isinstance(state[name], torch.Tensor)
                               or state[name].shape != parameter.shape
                               or state[name].dtype != parameter.dtype for name in moments):
                    raise ValueError("incomplete or unoptimized continuation Adam state")
    for name, config_name in (
        ("n_steps", "n_steps"), ("n_envs", "envs"), ("batch_size", "batch_size"),
        ("n_epochs", "n_epochs"), ("gamma", "gamma"), ("seed", "seed"),
    ):
        if getattr(source, name) != manifest["training"][config_name]:
            raise ValueError(f"runtime continuation model/config mismatch: {config_name}")
    provenance = {
        "method": "v5-to-v6-same-layout-full-state-continuation",
        "source_checkpoint": str(Path(path).resolve()),
        "source_checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_runtime": copy.deepcopy(manifest["runtime"]),
        "source_num_timesteps": source.num_timesteps,
        "source_n_updates": source._n_updates,
        "source_progress": copy.deepcopy(progress),
        "preserved": ["policy", "optimizer", "num_timesteps", "n_updates",
                      "learning_rate", "curriculum_progress"],
        "environment_reset": True,
    }
    # SB3's load(force_reset=True) only clears observations when env is supplied.
    source._last_obs = None
    return source, manifest, provenance


def validate_warm_start_manifest(manifest, signature, *, nstack):
    """Only either frozen source runtime may cross its reviewed compatibility boundary."""
    if type(nstack) is not int or nstack <= 0:
        raise ValueError("warm-start nstack must be a positive integer")
    runtime = manifest.get("runtime") if isinstance(manifest, dict) else None
    version = runtime.get("env_version") if isinstance(runtime, dict) else None
    if version == "grid26-events-v3":
        channels, hashes = 39, V3_SOURCE_HASHES
        destination_version = "grid26-events-v4-residual"
        destination_names = NEW_CHANNEL_NAMES
    elif version == "grid26-events-v4-residual":
        channels, hashes = 45, SOURCE_HASHES
        destination_version = "grid26-events-v5-tactics"
        destination_names = TACTICS_CHANNEL_NAMES
    else:
        raise ValueError("unreviewed warm-start source runtime version")
    expected = {
        "env_version": version, "obs_channels": channels,
        "observation_shape": [channels, 26, 26], "observation_dtype": "float32",
        "stacked_observation_shape": [channels * nstack, 26, 26], "stack_axis": "first",
        "action_space": {"type": "Discrete", "shape": [], "n": 15},
        "source_sha256": hashes, "rom_sha256": ROM_SHA256,
    }
    if channels == 45:
        expected["channel_names"] = list(NEW_CHANNEL_NAMES)
    validate_manifest(manifest, expected)
    if manifest["training"]["nstack"] != nstack:
        raise ValueError("warm-start cannot change nstack")
    destination_channels = len(destination_names)
    target = dict(expected, env_version=destination_version, obs_channels=destination_channels,
                  observation_shape=[destination_channels, 26, 26],
                  stacked_observation_shape=[destination_channels * nstack, 26, 26],
                  channel_names=list(destination_names))
    hashes = signature.get("source_sha256") if isinstance(signature, dict) else None
    if not isinstance(hashes, dict) or set(hashes) != set(RUNTIME_FILES) or any(
        not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        for value in hashes.values()
    ) or hashes.get("policy.py") != SOURCE_HASHES["policy.py"]:
        raise ValueError("incompatible warm-start destination runtime sources/architecture")
    if channels == 45 and any(
        hashes[name] != value for name, value in SOURCE_HASHES.items()
        if name not in ("battle_city_env.py", "heuristic.py", "migration.py", "game_state.py", "experiments.py")
    ):
        raise ValueError("unreviewed warm-start destination runtime source changes")
    target["source_sha256"] = hashes
    if signature != target:
        raise ValueError("incompatible warm-start destination runtime or channel layout")


def _validate_architecture(model, channels):
    policy = model.policy
    if type(model) is not MaskablePPO or type(policy) is not MaskableActorCriticCnnPolicy:
        raise ValueError("unsupported warm-start policy architecture")
    if not isinstance(model.observation_space, spaces.Box) \
            or model.observation_space.shape != (channels, 26, 26) \
            or model.observation_space.dtype != np.dtype("float32"):
        raise ValueError("incompatible warm-start model observation space")
    if not isinstance(model.action_space, spaces.Discrete) or model.action_space.n != 15 \
            or model.action_space.start != 0:
        raise ValueError("incompatible warm-start model action space")
    extractor = policy.features_extractor
    if type(extractor) is not Grid26CNN or policy.features_extractor_class is not Grid26CNN \
            or not policy.share_features_extractor \
            or any(getattr(policy, name) is not extractor for name in EXTRACTORS) \
            or policy.normalize_images or extractor.features_dim != 256 \
            or policy.net_arch != {"pi": [64, 64], "vf": [64, 64]} \
            or policy.activation_fn is not nn.Tanh:
        raise ValueError("unsupported warm-start Grid26CNN architecture")
    if [type(layer) for layer in extractor.cnn] != [
        nn.Conv2d, nn.ReLU, nn.Conv2d, nn.ReLU, nn.Conv2d, nn.ReLU, nn.Flatten,
    ] or [type(layer) for layer in extractor.linear] != [nn.Linear, nn.ReLU]:
        raise ValueError("unsupported warm-start Grid26CNN architecture")
    for layer, inputs, outputs, stride in (
        (extractor.cnn[0], channels, 32, 1), (extractor.cnn[2], 32, 64, 2),
        (extractor.cnn[4], 64, 64, 1),
    ):
        if (layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride,
            layer.padding, layer.dilation, layer.groups, layer.padding_mode, layer.bias is not None) != (
            inputs, outputs, (3, 3), (stride, stride), (1, 1), (1, 1), 1, "zeros", True,
        ):
            raise ValueError("unsupported warm-start convolution architecture")
    if (extractor.linear[0].in_features, extractor.linear[0].out_features) != (10816, 256):
        raise ValueError("unsupported warm-start projection architecture")
    if (extractor.cnn[6].start_dim, extractor.cnn[6].end_dim) != (1, -1):
        raise ValueError("unsupported warm-start flatten architecture")
    for network in (policy.mlp_extractor.policy_net, policy.mlp_extractor.value_net):
        if [type(layer) for layer in network] != [nn.Linear, nn.Tanh, nn.Linear, nn.Tanh] \
                or (network[0].in_features, network[0].out_features,
                    network[2].in_features, network[2].out_features) != (256, 64, 64, 64):
            raise ValueError("unsupported warm-start actor/critic architecture")
    for head, outputs in ((policy.action_net, 15), (policy.value_net, 1)):
        if type(head) is not nn.Linear or (head.in_features, head.out_features) != (64, outputs):
            raise ValueError("unsupported warm-start output architecture")


def expand_policy_state_dict(source, target, *, nstack):
    """Expand only 39 -> 45 or 45 -> 47, zeroing new inputs in every frame/alias."""
    if type(nstack) is not int or nstack <= 0 or set(source) != set(target) \
            or not set(INPUT_WEIGHTS).issubset(source):
        raise ValueError("incompatible warm-start state keys or nstack")
    channels = 45 if source[INPUT_WEIGHTS[0]].shape == (32, 45 * nstack, 3, 3) else 39
    destination_channels = 47 if channels == 45 else 45
    for name in source:
        if name.startswith("features_extractor."):
            suffix = name.removeprefix("features_extractor.")
            if any(f"{alias}.{suffix}" not in source or not torch.equal(
                source[name], source[f"{alias}.{suffix}"]) for alias in EXTRACTORS[1:]):
                raise ValueError("inconsistent shared-extractor checkpoint aliases")
    result = {}
    for name, weight in source.items():
        if weight.dtype != target[name].dtype:
            raise ValueError(f"incompatible warm-start tensor dtype: {name}")
        if name in INPUT_WEIGHTS:
            if tuple(weight.shape) != (32, channels * nstack, 3, 3) \
                    or tuple(target[name].shape) != (32, destination_channels * nstack, 3, 3):
                raise ValueError(f"incompatible warm-start input shape: {name}")
            expanded = torch.zeros_like(target[name])
            for frame in range(nstack):
                expanded[:, frame * destination_channels:frame * destination_channels + channels].copy_(
                    weight[:, frame * channels:(frame + 1) * channels])
            result[name] = expanded
            continue
        if weight.shape != target[name].shape:
            raise ValueError(f"incompatible warm-start tensor shape: {name}")
        result[name] = weight.detach().to(target[name].device).clone()
    return result


def copy_warm_start_weights(source, target, *, nstack):
    """Require a newly constructed PPO; its optimizer and counters stay untouched."""
    if type(nstack) is not int or nstack <= 0:
        raise ValueError("warm-start nstack must be a positive integer")
    channels = 45 if source.observation_space.shape == (45 * nstack, 26, 26) else 39
    _validate_architecture(source, channels * nstack)
    _validate_architecture(target, (47 if channels == 45 else 45) * nstack)
    if target.num_timesteps != 0 or target._n_updates != 0 \
            or target._current_progress_remaining != 1 or target.policy.optimizer.state \
            or target._last_obs is not None:
        raise ValueError("warm-start destination must be a fresh PPO with an unused optimizer")
    state = expand_policy_state_dict(source.policy.state_dict(), target.policy.state_dict(), nstack=nstack)
    target.policy.load_state_dict(state, strict=True)


def load_warm_start(path, signature, *, nstack):
    """Load the approved source on CPU without attaching a destination environment."""
    manifest = checkpoint_manifest(path)
    validate_warm_start_manifest(manifest, signature, nstack=nstack)
    source = MaskablePPO.load(path, device="cpu", env=None)
    channels = manifest["runtime"]["obs_channels"]
    _validate_architecture(source, channels * nstack)
    # Module loading can silently cast dtypes or overwrite inconsistent shared aliases.
    _, parameters, _ = load_from_zip_file(path, device="cpu", load_data=False)
    saved = parameters.get("policy")
    loaded = source.policy.state_dict()
    if not isinstance(saved, dict) or saved.keys() != loaded.keys() or any(
        not isinstance(weight, torch.Tensor) or weight.dtype != loaded[name].dtype
        or not torch.equal(weight, loaded[name]) for name, weight in saved.items()
    ):
        raise ValueError("incompatible warm-start checkpoint tensors or shared-extractor aliases")
    if type(manifest.get("timesteps")) is not int or manifest["timesteps"] < 0 \
            or manifest["timesteps"] != source.num_timesteps:
        raise ValueError("warm-start source timestep provenance mismatch")
    destination_channels = signature["obs_channels"]
    provenance = {
        "method": ("v4-to-v5-per-frame-tactics-input-expansion-weights-only" if channels == 45 else
                   "v3-to-v4-per-frame-input-expansion-zero-new-channels-weights-only"),
        "source_checkpoint": str(Path(path).resolve()),
        "source_checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_runtime": copy.deepcopy(manifest["runtime"]),
        "source_num_timesteps": int(source.num_timesteps),
        "source_n_updates": int(source._n_updates),
        "old_channel_names": list(NEW_CHANNEL_NAMES if channels == 45 else OLD_CHANNEL_NAMES),
        "new_channel_names": list(signature["channel_names"]),
        "stack_axis": "first", "nstack": nstack,
        "frame_mapping": [
            {"source_slice": [frame * channels, (frame + 1) * channels],
             "destination_slice": [frame * destination_channels, frame * destination_channels + channels],
             "zero_slice": [frame * destination_channels + channels, (frame + 1) * destination_channels]}
            for frame in range(nstack)
        ],
        "reset": {
            "optimizer": True, "num_timesteps": 0, "n_updates": 0,
            "curriculum_progress": {"combat": 0, "defense": 0, "full": 0},
        },
    }
    if "warm_start" in manifest:
        provenance["source_warm_start"] = copy.deepcopy(manifest["warm_start"])
    return source, provenance
