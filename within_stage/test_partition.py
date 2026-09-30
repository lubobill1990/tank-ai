import unittest


def pool_fixture():
    return {
        'families': [
            {'family_id': f'f{i:02d}', 'future_semantic_sha256': f'{i:064x}',
             'config_indices': [2 * i, 2 * i + 1]} for i in range(72)
        ],
        'specs': [
            {'config_index': i, 'family_id': f'f{i // 2:02d}',
             'spec': {'title_wait': i, 'stage_wait': 0, 'post_start_frames': 150}}
            for i in range(144)
        ],
    }


class PartitionTests(unittest.TestCase):
    def test_families_and_all_aliases_stay_in_one_partition(self):
        from within_stage.partition import partition_openings
        result = partition_openings(pool_fixture(), seed=20261001)
        self.assertEqual(len(result['train']['family_ids']), 40)
        self.assertEqual(len(result['development']['family_ids']), 12)
        self.assertEqual(len(result['final']['family_ids']), 20)
        groups = [set(result[k]['family_ids']) for k in ('train', 'development', 'final')]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        for name in ('train', 'development', 'final'):
            indices = set(result[name]['all_config_indices'])
            self.assertTrue(all((i ^ 1) in indices for i in indices))
            self.assertEqual(len(result[name]['representative_config_indices']), len(groups[
                ('train', 'development', 'final').index(name)]))
        self.assertEqual(result, partition_openings(pool_fixture(), seed=20261001))

    def test_duplicate_gameplay_families_or_inconsistent_membership_fail(self):
        from within_stage.partition import partition_openings
        pool = pool_fixture()
        pool['families'][1]['future_semantic_sha256'] = pool['families'][0]['future_semantic_sha256']
        with self.assertRaises(ValueError):
            partition_openings(pool, seed=1)
        pool = pool_fixture()
        pool['specs'][0]['family_id'] = 'wrong'
        with self.assertRaises(ValueError):
            partition_openings(pool, seed=1)

    def test_insufficient_diversity_is_not_filled_with_duplicate_cases(self):
        from within_stage.partition import partition_openings
        pool = pool_fixture()
        pool['families'] = pool['families'][:30]
        with self.assertRaises(ValueError):
            partition_openings(pool, seed=1)
