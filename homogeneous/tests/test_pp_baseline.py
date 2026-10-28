"""Replica selection must not alias devices, ports, or the retained base profile."""
import copy
import importlib.util
from pathlib import Path
import unittest
import shlex


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

    def test_replica_overrides_only_its_hccl_ports(self):
        original = config()
        original['vendors']['musa']['env'] = {'HCCL_IF_BASE_PORT': '61200', 'HF_HUB_OFFLINE': '1'}
        original['vendors']['musa']['instances']['pp-02']['env'] = {'HCCL_IF_BASE_PORT': '61500'}
        selected = manage.select_instance(original, 'musa', 'pp-02')
        self.assertEqual(selected['vendors']['musa']['env'],
                         {'HCCL_IF_BASE_PORT': '61500', 'HF_HUB_OFFLINE': '1'})
        self.assertEqual(original['vendors']['musa']['env']['HCCL_IF_BASE_PORT'], '61200')

    def test_ascend_visible_ids_follow_container_mounts(self):
        for mounted, expected in [([2, 3], '0,1'), (list(range(16)), '2,3')]:
            with self.subTest(mounted=mounted):
                manager = manage.Manager.__new__(manage.Manager)
                node = dict(host='192.168.162.23', interface='eth0', container='existing',
                            devices=[2, 3], container_devices=mounted)
                manager.config = {'vendors': {'ascend': {'nodes': [node, node],
                                  'env': {}, 'ray_port': 32000}}}
                manager.ssh = lambda node, command: command
                command = shlex.split(manager.docker('ascend', 0, 'true'))
                self.assertIn('ASCEND_RT_VISIBLE_DEVICES='+expected, command)

    def test_each_node_keeps_its_superpod_identity(self):
        manager = manage.Manager.__new__(manage.Manager)
        nodes = [dict(host='192.168.162.'+str(i), interface='eth0', container='existing',
                      devices=[2, 3], container_devices=list(range(16)),
                      env={'HCCL_LOGIC_SUPERPOD_ID':'ascend-node-'+str(i)}) for i in [23, 24]]
        manager.config = {'vendors': {'ascend': {'nodes': nodes, 'ray_port': 32000,
                          'env': {'HCCL_LOGIC_SUPERPOD_ID': 'invalid-common-id'}}}}
        manager.ssh = lambda node, command: command
        for stage, suffix in enumerate([23, 24]):
            with self.subTest(stage=stage):
                command = shlex.split(manager.docker('ascend', stage, 'true'))
                self.assertIn('HCCL_LOGIC_SUPERPOD_ID=ascend-node-'+str(suffix), command)
                self.assertNotIn('HCCL_LOGIC_SUPERPOD_ID=invalid-common-id', command)


if __name__ == '__main__':
    unittest.main()
