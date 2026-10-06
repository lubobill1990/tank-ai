"""Capture non-gameplay zero-page scratch rows from the local ROM.

Produces the cartridge-rng-v1 scratch replay trace: per-frame boundary values
of zero-page bytes that change during real play but are never read or written
by the Rust gameplay mirror (sound/NMI/render scratch). The trace is derived
from the local ROM and stays out of Git, like the ROM itself.

Rows are end-of-frame snapshots; the mirror replays row r at the start of its
logical frame r+phase. This is a boundary approximation of instruction-
interleaved audio/NMI writes, not emulator equivalence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from battle_city_env import BattleCityEnv
from rust_env import ScratchTrace, overlay_allowed

CAPTURE_VERSION = 'scratch-capture-v1'
# The same ten primitive controller masks as the mirror ABI, applied with the
# mirror's 4-frame repeat and fire release on the trailing two frames.
MASKS = (0, 16, 32, 64, 128, 1, 17, 33, 65, 129)


def capture_episode(env, rng, max_frames):
    env.reset(seed=int(rng.integers(2**31)))
    nes = env.nes
    rows = []
    while len(rows) < max_frames:
        mask = MASKS[int(rng.integers(len(MASKS)))]
        for phase in range(4):
            nes.step(mask if phase < 2 else mask & ~1)
            rows.append(np.asarray(nes.ram[:256], dtype=np.uint8).copy())
            ram = nes.ram
            if ram[0x68] != 0x80 or ram[0x51] == 0 or ram[0x80] == 0:
                return rows
            if len(rows) >= max_frames:
                break
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', required=True, help='trace .npz path (meta .json beside it)')
    parser.add_argument('--stages', default='1,5,10,20,35')
    parser.add_argument('--episodes', type=int, default=2, help='episodes per stage')
    parser.add_argument('--max-frames', type=int, default=3000, help='frame cap per episode')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    stages = sorted({int(s) for s in args.stages.split(',')})
    if not all(1 <= s <= 35 for s in stages) or args.episodes < 1 or args.max_frames < 1:
        raise SystemExit('stages must be in 1..35 with positive episodes/max-frames')
    output = Path(args.output)
    if output.suffix != '.npz':
        raise SystemExit('output must be a .npz path')

    rng = np.random.default_rng(args.seed)
    rows = []
    segments = []
    for stage in stages:
        env = BattleCityEnv(stage=stage, reward_profile='sparse')
        try:
            for episode in range(args.episodes):
                captured = capture_episode(env, rng, args.max_frames)
                rows.extend(captured)
                segments.append({'stage': stage, 'episode': episode, 'frames': len(captured)})
                print(f'stage {stage} episode {episode}: {len(captured)} frames')
        finally:
            env.close()
    data = np.stack(rows)

    distinct = np.array([len(np.unique(data[:, a])) for a in range(256)])
    changed = {int(a) for a in np.flatnonzero(distinct > 1)}
    selected = sorted(a for a in changed if overlay_allowed(a))
    excluded = sorted(changed - set(selected))
    if not selected:
        raise SystemExit('No overlayable changing zero-page bytes were observed')

    addresses = np.array(selected, dtype=np.uint8)
    trace = ScratchTrace(addresses, np.ascontiguousarray(data[:, addresses]))
    output.parent.mkdir(parents=True, exist_ok=True)
    trace.save(output)
    rom = Path(__file__).with_name('battle_city.nes')
    meta = {
        'version': CAPTURE_VERSION,
        'rom_sha256': hashlib.sha256(rom.read_bytes()).hexdigest(),
        'seed': args.seed,
        'stages': stages,
        'episodes_per_stage': args.episodes,
        'max_frames_per_episode': args.max_frames,
        'segments': segments,
        'rows': int(data.shape[0]),
        'addresses': [int(a) for a in addresses],
        'changed_but_gameplay_denied': excluded,
        'trace_sha256': trace.sha256,
        'note': 'End-of-frame boundary captures; replayed per logical frame, '
                'not instruction-interleaved audio/NMI writes.',
    }
    output.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='ascii')
    print(f'{len(addresses)} scratch addresses over {data.shape[0]} rows -> {output}')
    print('addresses:', ' '.join(f'{a:#04x}' for a in addresses))
    print('denied-but-changing:', ' '.join(f'{a:#04x}' for a in excluded))
    print('trace sha256:', trace.sha256)


if __name__ == '__main__':
    main()
