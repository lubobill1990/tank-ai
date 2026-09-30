"""Split genuinely different opening families, never just parameter labels."""
import random


def partition_openings(pool, *, seed, train_count=40, development_count=12, final_count=20):
    counts = (train_count, development_count, final_count)
    if type(seed) is not int or seed < 0 or any(type(n) is not int or n <= 0 for n in counts):
        raise ValueError('Expected a nonnegative split seed and positive family counts')
    families, specs = pool['families'], pool['specs']
    if len(families) != sum(counts):
        raise ValueError('Family count does not match the declared partition; no duplicates or silent omissions')
    ids = [family['family_id'] for family in families]
    hashes = [family['future_semantic_sha256'] for family in families]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate family identity or gameplay trace')
    by_config = {entry['config_index']: entry for entry in specs}
    if len(by_config) != len(specs):
        raise ValueError('Duplicate configuration identity')
    assigned = set()
    by_family = {}
    for family in families:
        members = family['config_indices']
        if not members or len(set(members)) != len(members):
            raise ValueError('Missing or repeated family members')
        for index in members:
            if (index in assigned or index not in by_config
                    or by_config[index]['family_id'] != family['family_id']):
                raise ValueError('Inconsistent family membership')
            assigned.add(index)
        by_family[family['family_id']] = family
    if assigned != set(by_config):
        raise ValueError('Unassigned configurations cannot silently enter another partition')
    order = sorted(ids, key=str)
    random.Random(seed).shuffle(order)
    result = {
        'version': 'within-stage-family-split-v1', 'split_seed': seed,
        'grouping': 'all parameter aliases of a gameplay-future family stay together',
        'selection': 'random family assignment before any learned-policy or teacher outcome',
        'statistical_caution': 'Distinct observed gameplay families are not a claim of IID sampling.',
    }
    offset = 0
    for name, count in zip(('train', 'development', 'final'), counts):
        chosen = order[offset:offset + count]
        result[name] = {
            'family_ids': chosen,
            'all_config_indices': sorted(index for family_id in chosen
                                         for index in by_family[family_id]['config_indices']),
            'representative_config_indices': [
                min(by_family[family_id]['config_indices']) for family_id in chosen
            ],
        }
        offset += count
    return result
