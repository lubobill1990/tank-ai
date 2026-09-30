"""Repair stream datasets whose first stored frame predates the observe() fix.

The collector stored an uninitialized buffer for each stream's very first
decision. That observation is deterministic given the stage RAM and episode
seed, so it is recomputed with the same native backend and spliced into the
payload. Only frame 0 of each stream is touched; every other byte is copied.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning_lab.demonstrations import FRAME_SHAPE
from native_codec import NativeRunCodec
from rust_env import NativeBatch, episode_seed, stage_ram


def initial_observation(stage, base_seed, stream, ram_cache):
    if stage not in ram_cache:
        ram_cache[stage] = stage_ram(stage, seed=0)
    batch = NativeBatch(ram_cache[stage][None], 12000,
                        seeds=[episode_seed(base_seed, stream, 0)])
    try:
        return batch.observe()[0].copy()
    finally:
        batch.close()


def repair_stream(path, codec, ram_cache):
    path = Path(path)
    metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('first_frame_repaired') == 'recomputed':
        return 'already'
    offsets = np.load(path / 'offsets.npy', allow_pickle=False)
    if len(offsets) < 2:
        return 'empty'
    payload = (path / 'frames.bin').read_bytes()
    first = codec.decode(payload[:int(offsets[1])])
    observation = initial_observation(metadata['stage'], metadata['base_seed'],
                                      metadata['stream'], ram_cache)
    if np.array_equal(first, observation):
        metadata['first_frame_repaired'] = 'recomputed'
        (path / 'metadata.json').write_text(json.dumps(metadata, indent=2),
                                            encoding='utf-8')
        return 'verified'
    chunk = codec.encode(observation)
    rest = payload[int(offsets[1]):]
    (path / 'frames.bin').write_bytes(chunk + rest)
    delta = len(chunk) - int(offsets[1])
    new_offsets = offsets.astype(np.int64)
    new_offsets[1:] += delta
    np.save(path / 'offsets.npy', new_offsets.astype(np.uint64), allow_pickle=False)
    metadata['payload_bytes'] = int(new_offsets[-1])
    metadata['first_frame_repaired'] = 'recomputed'
    (path / 'metadata.json').write_text(json.dumps(metadata, indent=2),
                                        encoding='utf-8')
    return 'repaired'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', required=True)
    args = parser.parse_args()
    codec = NativeRunCodec(FRAME_SHAPE)
    ram_cache = {}
    counts = {}
    for root in args.roots:
        for meta in sorted(Path(root).rglob('metadata.json')):
            result = repair_stream(meta.parent, codec, ram_cache)
            counts[result] = counts.get(result, 0) + 1
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
