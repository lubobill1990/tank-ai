"""Copy only successful (clear) episodes from demonstration streams.

Payload slices are copied byte-for-byte without decoding; array metadata is
subset accordingly. The filtered directory loads with DemonstrationDataset.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np


def filter_dataset(source, destination, *, outcomes=('clear',)):
    source, destination = Path(source), Path(destination)
    metadata = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
    reports = json.loads((source / 'episode_reports.json').read_text(encoding='utf-8'))
    arrays = {name: np.load(source / f'{name}.npy', allow_pickle=False)
              for name in ('offsets', 'teacher_actions', 'executed_actions',
                           'action_masks', 'episode_starts', 'episode_ids', 'outcomes')}
    keep_episodes = [r['episode_id'] for r in reports if r['outcome'] in outcomes]
    if not keep_episodes:
        return None
    keep_set = set(keep_episodes)
    sample_mask = np.isin(arrays['episode_ids'], list(keep_set))
    indices = np.flatnonzero(sample_mask)
    if not len(indices):
        return None
    destination.mkdir(parents=True, exist_ok=False)
    offsets = arrays['offsets']
    payload = np.memmap(source / 'frames.bin', mode='r', dtype=np.uint8)
    new_offsets = [0]
    with (destination / 'frames.bin').open('xb') as stream:
        for index in indices:
            chunk = payload[int(offsets[index]):int(offsets[index + 1])]
            stream.write(chunk.tobytes())
            new_offsets.append(new_offsets[-1] + len(chunk))
    starts = arrays['episode_starts'][indices]
    remap = {old: new for new, old in enumerate(sorted(keep_set))}
    new_reports = []
    for r in reports:
        if r['episode_id'] in keep_set:
            entry = deepcopy(r)
            entry['source_episode_id'] = r['episode_id']
            entry['episode_id'] = remap[r['episode_id']]
            new_reports.append(entry)
    new_arrays = {
        'offsets': np.asarray(new_offsets, np.uint64),
        'teacher_actions': arrays['teacher_actions'][indices],
        'executed_actions': arrays['executed_actions'][indices],
        'action_masks': arrays['action_masks'][indices],
        'episode_starts': starts,
        'episode_ids': np.cumsum(starts, dtype=np.int64) - 1,
        'outcomes': np.asarray([r['outcome'] for r in new_reports], dtype='<U64'),
    }
    for name, array in new_arrays.items():
        np.save(destination / f'{name}.npy', array, allow_pickle=False)
    (destination / 'episode_reports.json').write_text(
        json.dumps(new_reports, indent=2), encoding='utf-8')
    new_metadata = deepcopy(metadata)
    new_metadata.update(
        samples=int(len(indices)), episodes=int(starts.sum()),
        reported_episodes=len(new_reports), payload_bytes=new_offsets[-1],
        raw_frame_sha256='filtered-subset-not-rehashed',
        filtered_from=str(source), filter_outcomes=list(outcomes))
    (destination / 'metadata.json').write_text(
        json.dumps(new_metadata, indent=2), encoding='utf-8')
    return {'samples': int(len(indices)), 'episodes': int(starts.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True,
                        help='directory containing stage-*/stream-* datasets')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--outcomes', default='clear')
    args = parser.parse_args()
    outcomes = tuple(args.outcomes.split(','))
    total = {'samples': 0, 'episodes': 0, 'streams': 0}
    for metadata_path in sorted(args.source_root.rglob('metadata.json')):
        stream_dir = metadata_path.parent
        relative = stream_dir.relative_to(args.source_root)
        destination = args.output_root / relative
        if destination.exists():
            continue
        result = filter_dataset(stream_dir, destination, outcomes=outcomes)
        if result:
            total['samples'] += result['samples']
            total['episodes'] += result['episodes']
            total['streams'] += 1
    print(json.dumps(total))


if __name__ == '__main__':
    main()
