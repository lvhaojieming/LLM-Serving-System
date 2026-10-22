"""Serving mode must be explicit and cannot silently retain eager execution."""
import json
from pathlib import Path
import unittest

from heterogeneous_compute.quantization.deploy import server_command


class QuantizationDeploymentTests(unittest.TestCase):
    def setUp(self):
        config_path = Path(__file__).resolve().parents[1]/'quantization/deployment.json'
        self.config = json.loads(config_path.read_text(encoding='utf-8'))

    def test_default_vendor_graph_keeps_real_int4_adapter(self):
        command = server_command(self.config, 'awq')
        self.assertNotIn('--enforce-eager', command)
        self.assertEqual(command[command.index('--quantization')+1], 'moqe_ascend_int4')
        self.assertEqual(command[command.index('--tensor-parallel-size')+1], '2')
        self.assertEqual(command[command.index('--port')+1], '18424')

    def test_eager_requires_explicit_true(self):
        self.config['enforce_eager'] = True
        self.assertIn('--enforce-eager', server_command(self.config, 'awq'))
        self.config['enforce_eager'] = False
        self.assertNotIn('--enforce-eager', server_command(self.config, 'awq'))

    def test_string_false_cannot_silently_enable_eager(self):
        self.config['enforce_eager'] = 'false'
        with self.assertRaisesRegex(ValueError, 'boolean'):
            server_command(self.config, 'awq')

    def test_graph_metrics_are_explicit_and_preserve_graph_mode(self):
        self.config['cudagraph_metrics'] = True
        command = server_command(self.config, 'awq')
        self.assertIn('--cudagraph-metrics', command)
        self.assertNotIn('--enforce-eager', command)
