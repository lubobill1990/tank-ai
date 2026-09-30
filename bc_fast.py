"""Masked behavior cloning with threaded frame decode and batch prefetch.

Same objective and model family as learning_lab.imitation (masked teacher
cross-entropy over MaskablePPO logits); the data path decodes compressed
frames with a thread pool (the native codec releases the GIL) and overlaps
decode with GPU steps. Splits hold out whole episodes, never frames.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Stream:
    """One demonstration stream: metadata arrays in RAM, payload via memmap."""

    def __init__(self, path, codec):
        self.path = Path(path)
        self.codec = codec
        self.offsets = np.load(self.path / 'offsets.npy', allow_pickle=False)
        self.actions = np.load(self.path / 'teacher_actions.npy', allow_pickle=False)
        self.masks = np.load(self.path / 'action_masks.npy', allow_pickle=False)
        self.starts = np.load(self.path / 'episode_starts.npy', allow_pickle=False)
        self.episode_ids = np.load(self.path / 'episode_ids.npy', allow_pickle=False)
        self.payload = np.memmap(self.path / 'frames.bin', mode='r', dtype=np.uint8)

    def __len__(self):
        return len(self.actions)

    def frame(self, index):
        lo, hi = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.codec.decode(self.payload[lo:hi].tobytes())


class Corpus:
    def __init__(self, roots, threads=8):
        from native_codec import NativeRunCodec
        # decode() is stateless (only encode uses the scratch buffer), so one
        # shared codec instance is thread-safe for reading.
        self.codec = NativeRunCodec((47, 26, 26))
        self.streams = []
        for root in roots:
            for meta in sorted(Path(root).rglob('metadata.json')):
                self.streams.append(Stream(meta.parent, self.codec))
        if not self.streams:
            raise ValueError('No demonstration streams found')
        self.stream_of = []
        self.local_of = []
        self.episode_key = []
        for si, stream in enumerate(self.streams):
            n = len(stream)
            self.stream_of.append(np.full(n, si, np.int32))
            self.local_of.append(np.arange(n, dtype=np.int64))
            self.episode_key.append(stream.episode_ids + si * 1_000_000)
        self.stream_of = np.concatenate(self.stream_of)
        self.local_of = np.concatenate(self.local_of)
        self.episode_key = np.concatenate(self.episode_key)
        self.actions = np.concatenate([s.actions for s in self.streams])
        self.masks = np.concatenate([s.masks for s in self.streams])
        self.starts = np.concatenate([s.starts for s in self.streams])
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self.prefetcher = ThreadPoolExecutor(max_workers=1)

    def __len__(self):
        return len(self.actions)

    def split(self, seed, holdout=0.1):
        episodes = np.unique(self.episode_key)
        rng = np.random.default_rng(seed)
        rng.shuffle(episodes)
        cut = max(2, int(np.ceil(len(episodes) * holdout)))
        held = set(episodes[:cut].tolist())
        validation = np.flatnonzero(np.isin(self.episode_key, list(held)))
        training = np.flatnonzero(~np.isin(self.episode_key, list(held)))
        return training, validation, {'episodes': int(len(episodes)),
                                      'held_out_episodes': int(cut)}

    def _decode_one(self, task):
        row, index = task
        stream = self.streams[self.stream_of[index]]
        local = int(self.local_of[index])
        frame = stream.frame(local)
        previous = stream.frame(local - 1) if not stream.starts[local] else None
        return row, frame, previous

    def batch(self, indices):
        count = len(indices)
        observations = np.zeros((count, 94, 26, 26), np.float32)
        tasks = [(row, int(index)) for row, index in enumerate(indices)]
        for row, frame, previous in self.pool.map(self._decode_one, tasks):
            observations[row, 47:] = frame
            if previous is not None:
                observations[row, :47] = previous
        return {'observations': observations,
                'actions': self.actions[indices],
                'action_masks': self.masks[indices]}


def evaluate(model, corpus, indices, batch_size, to_tensors):
    import torch
    from learning_lab.imitation import _masked_cross_entropy
    model.policy.set_training_mode(False)
    total, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = corpus.batch(indices[start:start + batch_size])
            observations, actions, masks = to_tensors(batch)
            logits = model.policy.get_distribution(observations).distribution.logits
            loss = _masked_cross_entropy(logits, actions, masks)
            total += float(loss.item()) * len(actions)
            correct += int((logits.masked_fill(~masks, float('-inf'))
                            .argmax(dim=1) == actions).sum().item())
            count += len(actions)
    return {'ce': total / count, 'accuracy': correct / count}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=10)
    parser.add_argument('--initial-checkpoint', type=Path, default=None)
    parser.add_argument('--max-seconds', type=float, default=None)
    parser.add_argument('--extractor', choices=('cnn', 'ego'), default='cnn')
    args = parser.parse_args(argv)

    import torch
    from learning_lab.imitation import (_masked_cross_entropy, _new_model,
                                        load_imitation_checkpoint)

    def new_ego_model(learning_rate, seed, device):
        from sb3_contrib import MaskablePPO
        from learning_lab.ego_encoder import EgocentricGridCNN
        from learning_lab.imitation import _OfflineSpaces
        return MaskablePPO(
            'CnnPolicy', _OfflineSpaces(), learning_rate=learning_rate, seed=seed,
            device=device, n_steps=2, batch_size=2, n_epochs=1, verbose=0,
            policy_kwargs={'features_extractor_class': EgocentricGridCNN,
                           'normalize_images': False,
                           'net_arch': {'pi': [64, 64], 'vf': [64, 64]}})
    args.output.mkdir(parents=True, exist_ok=False)
    corpus = Corpus(args.roots, threads=args.threads)
    training, validation, split = corpus.split(args.seed)
    print(json.dumps({'samples': len(corpus), 'train': len(training),
                      'validation': len(validation), **split}), flush=True)
    if args.initial_checkpoint:
        try:
            model = load_imitation_checkpoint(args.initial_checkpoint, device=args.device)
        except Exception:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(
                args.initial_checkpoint, device=args.device,
                custom_objects={'n_steps': 2, 'n_envs': 1, 'batch_size': 2})
        for group in model.policy.optimizer.param_groups:
            group['lr'] = args.learning_rate
    elif args.extractor == 'ego':
        model = new_ego_model(args.learning_rate, args.seed, args.device)
    else:
        model = _new_model(args.learning_rate, args.seed, args.device)

    def to_tensors(batch):
        observations = torch.as_tensor(batch['observations'], device=model.device)
        actions = torch.as_tensor(batch['actions'], device=model.device)
        masks = torch.as_tensor(batch['action_masks'], device=model.device)
        return observations, actions, masks

    deadline = time.monotonic() + args.max_seconds if args.max_seconds else None
    rng = np.random.default_rng(args.seed)
    history = []
    best_ce = float('inf')
    start_time = time.monotonic()
    stopped = None
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(training)
        model.policy.set_training_mode(True)
        total, seen = 0.0, 0
        pending = None
        epoch_start = time.monotonic()
        for cursor in range(0, len(order), args.batch_size):
            selection = order[cursor:cursor + args.batch_size]
            if pending is None:
                batch = corpus.batch(selection)
            else:
                batch = pending.result()
            following = order[cursor + args.batch_size:cursor + 2 * args.batch_size]
            pending = (corpus.prefetcher.submit(corpus.batch, following)
                       if len(following) else None)
            observations, actions, masks = to_tensors(batch)
            model.policy.optimizer.zero_grad(set_to_none=True)
            logits = model.policy.get_distribution(observations).distribution.logits
            loss = _masked_cross_entropy(logits, actions, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5,
                                           error_if_nonfinite=True)
            model.policy.optimizer.step()
            total += float(loss.item()) * len(selection)
            seen += len(selection)
            if deadline and time.monotonic() > deadline:
                stopped = 'deadline'
                break
        metrics = evaluate(model, corpus, validation, args.batch_size, to_tensors)
        history.append({'epoch': epoch, 'training_ce': total / max(seen, 1),
                        'validation_ce': metrics['ce'],
                        'validation_accuracy': metrics['accuracy'],
                        'epoch_seconds': time.monotonic() - epoch_start})
        print(json.dumps(history[-1]), flush=True)
        if metrics['ce'] < best_ce:
            best_ce = metrics['ce']
            model.save(args.output / 'best.zip')
        model.save(args.output / 'final.zip')
        (args.output / 'report.json').write_text(json.dumps({
            'history': history, 'best_validation_ce': best_ce,
            'roots': [str(r) for r in args.roots], 'split': split,
            'config': {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in vars(args).items()},
            'elapsed_seconds': time.monotonic() - start_time,
            'stop_reason': stopped}, indent=2), encoding='utf-8')
        if stopped:
            break


if __name__ == '__main__':
    main()
