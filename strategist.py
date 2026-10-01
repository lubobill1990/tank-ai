"""Distill LLM strategic orders into a tiny MLP; run it as the commander.

Training data comes from strategy.jsonl rows logged during LLM-commander
collection (features + chosen order + episode outcome). The runtime
StrategistCommander mirrors the Commander interface but needs no network
calls: executor + MPC veto + this head is a real-time, LLM-free system.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODES = ('hunt', 'protect', 'item', 'goto', 'hold')
FEATURE_SIZE = 13 + 4 + 6 * 9 + 9


def encode_features(features):
    """strategy_features dict -> fixed float vector."""
    vector = np.zeros(FEATURE_SIZE, np.float32)
    player = features.get('player')
    cursor = 0
    if player is not None:
        pr, pc, facing, power = player
        vector[0:2] = pr / 25.0, pc / 25.0
        vector[2] = 1.0
        vector[3] = power / 3.0
        vector[13 + int(facing)] = 1.0
    vector[4] = features.get('lives', 0) / 5.0
    vector[5] = features.get('remaining', 0) / 20.0
    vector[6] = features.get('spawn_left', 0) / 20.0
    vector[7] = features.get('enemy_bullets', 0) / 5.0
    vector[8] = float(features.get('fire_ready', False))
    vector[9] = float(features.get('freeze', False))
    vector[10] = float(features.get('shield', False))
    vector[11] = features.get('player_base_dist', 272) / 272.0
    vector[12] = features.get('enemy_base_dist', 272) / 272.0
    cursor = 17
    enemies = features.get('enemies', [])
    enemies = sorted(enemies, key=lambda e: abs(e[0] - 25) + abs(e[1] - 12))[:6]
    for enemy in enemies:
        er, ec, facing, armor, drops = enemy
        vector[cursor:cursor + 9] = [
            1.0, er / 25.0, ec / 25.0,
            float(facing == 0), float(facing == 1), float(facing == 2),
            float(facing == 3), armor / 4.0, float(drops)]
        cursor += 9
    cursor = 17 + 54
    item = features.get('item')
    if item is not None:
        ir, ic, kind = item
        vector[cursor] = 1.0
        vector[cursor + 1:cursor + 3] = ir / 25.0, ic / 25.0
        if 0 <= kind < 6:
            vector[cursor + 3 + kind] = 1.0
    return vector


def build_model():
    import torch.nn as nn

    class StrategistNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(FEATURE_SIZE, 192), nn.ReLU(),
                nn.Linear(192, 128), nn.ReLU())
            self.mode_head = nn.Linear(128, len(MODES))
            self.target_head = nn.Linear(128, 2)

        def forward(self, x):
            h = self.trunk(x)
            return self.mode_head(h), self.target_head(h)

    return StrategistNet()


def load_rows(roots, *, success_only=True):
    rows = []
    for root in roots:
        for path in Path(root).rglob('strategy.jsonl'):
            for line in path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                if success_only and row.get('outcome') != 'clear':
                    continue
                rows.append(row)
    return rows


def train(args):
    import torch
    import torch.nn.functional as F
    rows = load_rows(args.roots, success_only=not args.all_outcomes)
    if len(rows) < 200:
        raise SystemExit(f'Too few strategy rows: {len(rows)}')
    features = np.stack([encode_features(r['features']) for r in rows])
    modes = np.array([MODES.index(r['order']['mode']) for r in rows], np.int64)
    targets = np.zeros((len(rows), 2), np.float32)
    has_target = np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        t = r['order'].get('target')
        if r['order']['mode'] == 'goto' and t:
            targets[i] = [t[0] / 25.0, t[1] / 25.0]
            has_target[i] = True
    device = args.device
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.as_tensor(features, device=device)
    y = torch.as_tensor(modes, device=device)
    t = torch.as_tensor(targets, device=device)
    m = torch.as_tensor(has_target, device=device)
    counts = np.bincount(modes, minlength=len(MODES)).astype(np.float64)
    weights = torch.as_tensor((counts.sum() / np.maximum(counts, 1.0)) ** 0.5,
                              dtype=torch.float32, device=device)
    rng = np.random.default_rng(0)
    split = rng.permutation(len(rows))
    val_n = max(64, len(rows) // 10)
    val_idx = torch.as_tensor(split[:val_n], device=device)
    train_idx = torch.as_tensor(split[val_n:], device=device)
    best_acc, best_state = 0.0, None
    for epoch in range(args.epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        for start in range(0, len(perm), 512):
            idx = perm[start:start + 512]
            logits, pred_t = model(x[idx])
            loss = F.cross_entropy(logits, y[idx], weight=weights)
            sel = m[idx]
            if sel.any():
                loss = loss + F.mse_loss(pred_t[sel], t[idx][sel])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits, _ = model(x[val_idx])
            acc = float((logits.argmax(1) == y[val_idx]).float().mean())
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        print(json.dumps({'epoch': epoch + 1, 'val_mode_acc': round(acc, 4)}), flush=True)
    model.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'modes': MODES,
                'feature_size': FEATURE_SIZE, 'rows': len(rows),
                'best_val_mode_acc': best_acc}, args.output)
    print(json.dumps({'saved': str(args.output), 'rows': len(rows),
                      'best_val_mode_acc': round(best_acc, 4)}))


class StrategistCommander:
    """Drop-in commander: local MLP instead of an API call."""

    def __init__(self, checkpoint, device='cpu'):
        import torch
        self.torch = torch
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model = build_model().to(device)
        self.model.load_state_dict(payload['state_dict'])
        self.model.eval()
        self.device = device
        self.calls = 0
        self.cost = 0.0
        self.budget = float('inf')
        self.model_name = f'strategist:{Path(checkpoint).name}'

    def order(self, state_text, state=None):
        from llm_rom_player import strategy_features
        if state is None:
            return {'mode': 'hunt'}, 'no-state'
        vector = encode_features(strategy_features(state))
        with self.torch.no_grad():
            logits, target = self.model(
                self.torch.as_tensor(vector, device=self.device)[None])
        self.calls += 1
        mode = MODES[int(logits.argmax())]
        order = {'mode': mode}
        if mode == 'goto':
            row = int(round(float(target[0, 0]) * 25))
            col = int(round(float(target[0, 1]) * 25))
            order['target'] = [min(max(row, 0), 25), min(max(col, 0), 25)]
        return order, mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--all-outcomes', action='store_true')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
