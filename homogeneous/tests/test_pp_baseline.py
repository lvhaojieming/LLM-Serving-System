"""Replica selection must not alias devices, ports, or the retained base profile."""
import copy
import importlib.util
from pathlib import Path
import unittest


MODULE = Path(__file__).resolve().parents[1]/'experiments/pp_baseline/manage.py'
loader = importlib.util.spec_from_file_location('pp_baseline_manage', MODULE)
manage = importlib.util.module_from_spec(loader)
loader.loader.exec_module(manage)


def config():
    return {'vendors': {'musa': {
        'ray_port': 32200, 'api_port': 18397,
        'nodes': [{'devices': [4, 5]}, {'devices': [0, 1]}],
        'instances': {'pp-02': {'ray_port': 33000, 'api_port': 18398,
                               'devices': [[0, 1], [2, 3]]}}
    }}}


class ReplicaSelectionTests(unittest.TestCase):
    def test_resolved_replica_is_independent(self):
        original = config()
        before = copy.deepcopy(original)
        resolved = manage.select_instance(original, 'musa', 'pp-02')['vendors']['musa']
        self.assertEqual(original, before)
        self.assertEqual(resolved['nodes'][0]['devices'], [0, 1])
        self.assertEqual(resolved['nodes'][1]['devices'], [2, 3])
        self.assertEqual(resolved['instance_id'], 'pp-02')
        self.assertEqual(resolved['ray_port'], 33000)
        self.assertNotIn('instances', resolved)
        resolved['nodes'][0]['devices'].append(7)
        self.assertEqual(original, before)


    def test_reject_conflicts(self):
        for field, value in [
            ('devices', [[4, 5], [2, 3]]), ('devices', [[0, 0], [2, 3]]),
            ('devices', [[0, 1], [True, 3]]), ('ray_port', 32300),
            ('api_port', 32230), ('api_port', 33050),
        ]:
            with self.subTest(field=field, value=value):
                original = config()
                original['vendors']['musa']['instances']['pp-02'][field] = value
                with self.assertRaises(ValueError):
                    manage.select_instance(original, 'musa', 'pp-02')


    def test_reject_path_traversal(self):
        original = config()
        entries = original['vendors']['musa']['instances']
        entries['../base'] = entries.pop('pp-02')
        with self.assertRaisesRegex(ValueError, 'Invalid instance ID'):
            manage.select_instance(original, 'musa', '../base')


if __name__ == '__main__':
    unittest.main()
