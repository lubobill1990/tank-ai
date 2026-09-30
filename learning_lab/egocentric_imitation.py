"""Observation-only egocentric BC, with its own strict lab checkpoint contract.

``uv run --no-sync python -m learning_lab.egocentric_imitation --datasets CORPUS
--output NEW_DIR --device cpu``. Deadlines are absolute time.monotonic() values.
This module never steps an environment. Development classification metrics do
not establish a ROM clear or generalization. Frozen imitation helpers provide
identical corpus validation, whole-episode splits, masked loss and monitoring.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import platform
import time

import gymnasium as gym
import numpy as np
import torch

from learning_lab.ego_encoder import EgocentricGridCNN
from learning_lab import imitation as base


_FORMAT = "tank-ai-egocentric-imitation-checkpoint-v1"
_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT = {
    "algorithm": "MaskablePPO", "policy": "CnnPolicy", "extractor": "egocentric-cnn-v1",
    "observation_shape": [94, 26, 26], "observation_dtype": "float32",
    "n_stack": 2, "channels_order": "first", "actions": 10, "features_dim": 256,
    "normalize_images": False, "share_features_extractor": True,
    "net_arch": {"pi": [64, 64], "vf": [64, 64]},
    "centering": {"crop_size": 25, "mode": "nearest", "padding_mode": "zeros",
                  "align_corners": True, "anchor_rounding": "floor(centroid+0.5)",
                  "fallback": "current_player_then_previous_player_then_arena_center",
                  "input_only_no_grad": True, "both_frames_same_anchor": True},
    "local_channels": [32, 48, 64], "local_features": 192,
    "global_channels": [16, 24, 32], "global_features": 64,
    "extra_scalars": ["normalized_player_x", "normalized_player_y", "fire_ready", "lives"],
    "pooling": "none; fixed-stride convolutions and flatten/linear",
    "objective": "masked_teacher_cross_entropy", "value_function_supervision": False,
}


def _source_hashes():
    return {name: base._sha256(_ROOT.joinpath(*name.split("/"))) for name in (
        "learning_lab/egocentric_imitation.py", "learning_lab/ego_encoder.py",
        "learning_lab/imitation.py", "learning_lab/demonstrations.py", "native_codec.py")}


def _new_model(learning_rate, seed, device):
    from sb3_contrib import MaskablePPO
    return MaskablePPO(
        "CnnPolicy", base._OfflineSpaces(), learning_rate=learning_rate, seed=seed,
        device=device, n_steps=2, batch_size=2, n_epochs=1, verbose=0,
        policy_kwargs={"features_extractor_class": EgocentricGridCNN,
                       "features_extractor_kwargs": {"features_dim": 256},
                       "normalize_images": False, "share_features_extractor": True,
                       "net_arch": deepcopy(_CONTRACT["net_arch"])},
    )


def _save_checkpoint(model, path, report, role):
    path = Path(path)
    pending = path.with_name(path.stem + ".pending.zip")
    model.save(pending)
    pending.replace(path)
    digest = base._sha256(path)
    manifest = {
        "format": _FORMAT, "checkpoint_file": path.name, "checkpoint_sha256": digest,
        "contract": deepcopy(_CONTRACT), "role": role, "warning": base._WARNING,
        "sources": report["sources"], "versions": report["versions"],
        "datasets": report["datasets"], "split": report["split"],
        "training_config": report["training_config"], "actual_device": report["actual_device"],
        "exact_simulator_resume": False, "parent_checkpoint": None,
        "progress": {key: report[key] for key in (
            "status", "stop_reason", "epochs_completed", "current_epoch", "supervised_samples",
            "validation_samples", "optimizer_updates", "environment_interactions")},
    }
    base._json_write(path.with_suffix(".manifest.json"), manifest)
    return {"path": str(path), "sha256": digest, "epoch": report["epochs_completed"]}


def _verify_model(model):
    from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
    policy, space = model.policy, model.observation_space
    if not isinstance(space, gym.spaces.Box) or space.shape != (94, 26, 26) \
            or space.dtype != np.float32 or not np.all(space.low == -1) \
            or not np.all(space.high == 1) \
            or not isinstance(model.action_space, gym.spaces.Discrete) \
            or model.action_space.n != 10 or model.action_space.start != 0 \
            or type(policy) is not MaskableActorCriticCnnPolicy \
            or type(policy.features_extractor) is not EgocentricGridCNN \
            or policy.features_extractor.features_dim != 256 \
            or policy.features_extractor.crop_size != 25 \
            or policy.normalize_images or not policy.share_features_extractor \
            or policy.net_arch != _CONTRACT["net_arch"]:
        raise ValueError("Loaded policy violates the egocentric architecture contract")
    if not isinstance(policy.optimizer, torch.optim.Adam):
        raise ValueError("Egocentric checkpoint requires the full Adam optimizer")
    if any(isinstance(layer, (torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveMaxPool2d))
           for layer in policy.features_extractor.modules()):
        raise ValueError("Egocentric CNN cannot contain adaptive pooling")


def train_egocentric_imitation(dataset_paths, output, *, epochs=10, batch_size=256,
                              learning_rate=3e-4, seed=0, device="cpu", deadline=None):
    """Train a fresh egocentric policy using the frozen baseline's episode split.

    Output must be fresh. Dataset/runtime failures are reported; deadlines and
    resource pressure return partial progress. Final checkpoint serialization may
    outlast the deadline, but no new optimizer step begins after its check.
    """
    for name, value, minimum in (("epochs", epochs, 1), ("batch_size", batch_size, 1), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) \
            or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if deadline is not None and (
        isinstance(deadline, bool) or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
    ):
        raise ValueError("deadline must be an absolute finite monotonic time")
    paths = [dataset_paths] if isinstance(dataset_paths, (str, Path)) else list(dataset_paths)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "format": "tank-ai-egocentric-imitation-report-v1", "status": "running", "stop_reason": None,
        "warning": base._WARNING, "architecture": deepcopy(_CONTRACT),
        "training_config": {"epochs": int(epochs), "batch_size": int(batch_size),
                            "learning_rate": learning_rate, "seed": int(seed),
                            "device": str(device), "deadline_monotonic": deadline},
        "epochs_completed": 0, "current_epoch": 0, "supervised_samples": 0,
        "validation_samples": 0, "optimizer_updates": 0, "environment_interactions": 0,
        "datasets": [], "split": None, "parent_checkpoint": None, "actual_device": None,
        "initial_validation_ce": None, "best_validation_ce": None, "history": [],
        "checkpoints": {"epochs": [], "best": None, "final": None},
        "sources": {}, "versions": {}, "resource_monitor": {"status": "not_started"},
    }
    model, corpus = None, None
    try:
        base._check_deadline(deadline)
        monitor = base._start_monitor(output, device, report)
        import sb3_contrib
        import stable_baselines3
        report["versions"] = {
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": str(torch.__version__), "sb3_contrib": sb3_contrib.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        }
        report["sources"] = _source_hashes()
        paths = [Path(path).resolve() for path in paths]
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("Provide one or more datasets without duplicate paths")
        corpus = base._Corpus(paths, output, deadline)
        report["datasets"] = corpus.identities
        train_indices, validation_indices, report["split"] = base._split_episodes(corpus.episode_ids, seed)
        base._check_deadline(deadline)
        model = _new_model(learning_rate, seed, device)
        _verify_model(model)
        report["actual_device"] = str(model.device)
        initial = base._evaluate(model, corpus, validation_indices, batch_size, deadline, monitor, report)
        report["initial_validation_ce"] = report["best_validation_ce"] = initial["ce"]
        report["checkpoints"]["best"] = _save_checkpoint(model, output / "best.zip", report, "best")
        rng = np.random.default_rng(seed)
        for epoch in range(1, epochs + 1):
            report["current_epoch"] = epoch
            order = rng.permutation(train_indices)
            model.policy.set_training_mode(True)
            total, seen = 0., 0
            for first in range(0, len(order), batch_size):
                base._check_deadline(deadline)
                if monitor is not None:
                    monitor.check("egocentric_minibatch")
                selection = order[first:first + batch_size]
                batch = corpus.batch(selection)
                base._check_deadline(deadline)
                model.policy.optimizer.zero_grad(set_to_none=True)
                loss, _ = base._batch_loss(model, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm,
                                               error_if_nonfinite=True)
                base._check_deadline(deadline)
                model.policy.optimizer.step()
                report["optimizer_updates"] += 1
                report["supervised_samples"] += len(selection)
                total += float(loss.item()) * len(selection)
                seen += len(selection)
            validation = base._evaluate(model, corpus, validation_indices, batch_size,
                                        deadline, monitor, report)
            report["epochs_completed"] = epoch
            report["history"].append({
                "epoch": epoch, "training_ce": total / seen, "validation_ce": validation["ce"],
                "validation_accuracy": validation["accuracy"],
            })
            checkpoint = _save_checkpoint(model, output / f"epoch-{epoch:04d}.zip", report, "completed_epoch")
            report["checkpoints"]["epochs"].append(checkpoint)
            if validation["ce"] < report["best_validation_ce"]:
                report["best_validation_ce"] = validation["ce"]
                report["checkpoints"]["best"] = _save_checkpoint(model, output / "best.zip", report, "best")
            base._json_write(output / "report.json", report)
        report["status"], report["stop_reason"] = "completed", "epochs_completed"
    except (base._DeadlineReached, KeyboardInterrupt) as error:
        report["status"] = "partial"
        report["stop_reason"] = "deadline" if isinstance(error, base._DeadlineReached) else "interrupted"
    except Exception as error:
        pressure = type(error).__name__ == "ResourcePressureError"
        report["status"] = "partial" if pressure else "failed"
        report["stop_reason"] = "resource_pressure" if pressure else "error"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if corpus is not None:
            try:
                corpus.close()
            except Exception as error:
                report["status"], report["stop_reason"] = "failed", "cleanup_error"
                report["cleanup_error"] = f"{type(error).__name__}: {error}"
        if model is not None:
            try:
                report["checkpoints"]["final"] = _save_checkpoint(model, output / "final.zip", report, "final")
            except Exception as error:
                report["status"], report["stop_reason"] = "failed", "checkpoint_error"
                report["checkpoint_error"] = f"{type(error).__name__}: {error}"
        report["elapsed_seconds"] = max(0., time.monotonic() - started)
        base._json_write(output / "report.json", report)
    return report


def load_egocentric_checkpoint(path, device="cpu"):
    """Verify this candidate's manifest, hashes and contract before trusted-local load."""
    from sb3_contrib import MaskablePPO
    path = Path(path).resolve()
    if path.is_dir():
        path = path / ("best.zip" if (path / "best.zip").is_file() else "final.zip")
    sidecar = path.with_suffix(".manifest.json")
    try:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Missing or invalid egocentric manifest: {sidecar}") from error
    if not isinstance(manifest, dict) or manifest.get("format") != _FORMAT \
            or manifest.get("contract") != _CONTRACT or manifest.get("checkpoint_file") != path.name \
            or manifest.get("sources") != _source_hashes() or not manifest.get("datasets") \
            or not isinstance(manifest.get("split"), dict) \
            or not isinstance(manifest.get("training_config"), dict) \
            or manifest.get("exact_simulator_resume") is not False:
        raise ValueError("Invalid egocentric manifest contract or source identity")
    base._verify_provenance(manifest)
    if not path.is_file() or base._sha256(path) != manifest.get("checkpoint_sha256"):
        raise ValueError("Egocentric checkpoint SHA256 mismatch")
    model = MaskablePPO.load(path, device=device)
    _verify_model(model)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deadline", type=float, help="Absolute time.monotonic() deadline")
    args = vars(parser.parse_args(argv))
    args["dataset_paths"] = args.pop("datasets")
    report = train_egocentric_imitation(**args)
    print(json.dumps(report, indent=2, allow_nan=False))
    return {"completed": 0, "failed": 1, "partial": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
