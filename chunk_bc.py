"""Action-chunking behaviour cloning over the demonstration corpus.

The network predicts the next K primitive actions from one stacked
observation (ACT-style), attacking the compounding-error failure mode of
single-step cloning. Future-step labels stop at the episode boundary and are
masked out of the loss. The checkpoint is a plain torch payload executed by
chunk_eval's receding-horizon runner; no SB3 machinery is involved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bc_fast import Corpus


def build_model(chunk):
    import torch
    import torch.nn as nn
    from learning_lab.ego_encoder import center_observations

    class ChunkPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.chunk = chunk
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
            self.trunk = nn.Sequential(nn.Linear(384 + 128 + 4, 512), nn.ReLU())
            self.head = nn.Linear(512, chunk * 10)

        def forward(self, observations):
            centered, scalars = center_observations(observations)
            local = self.local_cnn(centered)
            global_map = self.global_cnn(observations.detach())
            h = self.trunk(torch.cat((local, global_map, scalars), dim=1))
            return self.head(h).view(-1, self.chunk, 10)

    return ChunkPolicy()


def chunk_labels(corpus, indices, chunk):
    """actions[t:t+K] along the same episode; valid mask marks in-episode."""
    n = len(corpus.actions)
    labels = np.zeros((len(indices), chunk), np.int64)
    valid = np.zeros((len(indices), chunk), bool)
    for row, start in enumerate(indices):
        episode = corpus.episode_key[start]
        for j in range(chunk):
            k = start + j
            if k >= n or corpus.episode_key[k] != episode:
                break
            labels[row, j] = corpus.actions[k]
            valid[row, j] = True
    return labels, valid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=384)
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--initial-checkpoint', type=Path, default=None)
    args = parser.parse_args(argv)

    import torch
    import torch.nn.functional as F
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    corpus = Corpus(args.roots, threads=args.threads)
    training, validation, split = corpus.split(args.seed)
    print(json.dumps({'samples': len(corpus), 'train': len(training),
                      'validation': len(validation), **split}), flush=True)
    model = build_model(args.chunk).to(args.device)
    if args.initial_checkpoint is not None:
        payload = torch.load(args.initial_checkpoint, map_location=args.device,
                             weights_only=False)
        if payload['chunk'] != args.chunk:
            raise SystemExit('chunk size mismatch with the initial checkpoint')
        model.load_state_dict(payload['state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    history = []
    start_time = time.monotonic()

    def run_batch(indices, train):
        batch = corpus.batch(indices)
        labels, valid = chunk_labels(corpus, indices, args.chunk)
        observations = torch.as_tensor(batch['observations'], device=args.device)
        y = torch.as_tensor(labels, device=args.device)
        m = torch.as_tensor(valid, device=args.device)
        logits = model(observations)
        loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1),
                               reduction='none').reshape(y.shape)
        loss = (loss * m).sum() / m.sum().clamp_min(1)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        first_acc = ((logits[:, 0].argmax(1) == y[:, 0]) & m[:, 0]).sum() \
            / m[:, 0].sum().clamp_min(1)
        return float(loss.item()), float(first_acc)

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(training)
        model.train()
        total, batches = 0.0, 0
        epoch_start = time.monotonic()
        for cursor in range(0, len(order), args.batch_size):
            loss, _ = run_batch(order[cursor:cursor + args.batch_size], True)
            total += loss
            batches += 1
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = 0.0, 0.0
            val_batches = 0
            for cursor in range(0, len(validation), args.batch_size):
                loss, acc = run_batch(validation[cursor:cursor + args.batch_size], False)
                val_loss += loss
                val_acc += acc
                val_batches += 1
        history.append({'epoch': epoch, 'train_ce': total / batches,
                        'val_ce': val_loss / val_batches,
                        'val_first_acc': val_acc / val_batches,
                        'epoch_seconds': time.monotonic() - epoch_start})
        print(json.dumps(history[-1]), flush=True)
        torch.save({'state_dict': model.state_dict(), 'chunk': args.chunk,
                    'history': history}, args.output / 'final.pt')
    (args.output / 'report.json').write_text(json.dumps({
        'history': history, 'roots': [str(r) for r in args.roots],
        'config': {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
        'elapsed_seconds': time.monotonic() - start_time}, indent=2),
        encoding='utf-8')


if __name__ == '__main__':
    main()
