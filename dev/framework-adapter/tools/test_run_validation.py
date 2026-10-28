import copy
import unittest
from run_validation import command, PATHS


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(schema=1, model_kind='onerec-8b', operator='silu', mode='native',
                           device=0, steps=3, seq_len=16)
        self.config.update({key: '/test/' + key for key in PATHS})

    def test_native(self):
        result = command(self.config)
        self.assertEqual(result[-4:], ['--onerec-mode', 'native', '--seq-len', '16'])

    def test_modes(self):
        for mode, expected in [('off', 'disabled'), ('unqualified', 'unqualified'), ('on', 'target')]:
            with self.subTest(mode=mode):
                cfg = dict(self.config, mode=mode, reference='/test/native')
                result = command(cfg)
                self.assertEqual(result[result.index('--onerec-mode')+1], expected)

    def test_qwen_paired(self):
        result = command(dict(self.config, model_kind='qwen-embedding-0.6b', mode='compare', seq_len=32))
        self.assertEqual(result[-2:], ['--stage', 'qwen'])

    def test_rejects_invalid(self):
        for change in [{'extra': 1}, {'schema': 2}, {'operator': 'sigmoid'}, {'device': True},
                       {'device': -1}, {'steps': 4}, {'steps': True}, {'mode': 'on'},
                       {'seq_len': 64}, {'model': 'relative'}, {'model_kind': 'deepfm'},
                       {'second_device': 1}, {'reference': '/test/native'},
                       {'model_kind': 'qwen-embedding-0.6b', 'mode': 'on', 'seq_len': 32}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                command(dict(self.config, **change))

    def test_missing_keys(self):
        for key in self.config:
            cfg = copy.copy(self.config); del cfg[key]
            with self.subTest(key=key), self.assertRaises(ValueError): command(cfg)


if __name__ == '__main__':
    unittest.main()
