"""Communication consumer checks against an explicitly selected prototype.

Set COMM_RUNTIME_ROOT to the prototype directory. The real public runtime API
is imported, but its current device backend and c10d are mocked: no hardware,
collective transport, recovery or performance acceptance is implied.
"""
import inspect
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import distributed as comm


class RuntimeConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get('COMM_RUNTIME_ROOT')
        if not root:
            raise unittest.SkipTest('Set COMM_RUNTIME_ROOT for prototype consumer checks')
        root = Path(root).resolve()
        if not (root / 'runtime/__init__.py').is_file():
            raise RuntimeError('COMM_RUNTIME_ROOT does not contain runtime/__init__.py')
        sys.path.insert(0, str(root))
        import runtime
        if Path(runtime.__file__).resolve() != root / 'runtime/__init__.py':
            raise RuntimeError('A different runtime module was already imported')
        cls.runtime = runtime

    def setUp(self):
        self.backend = Mock()
        self.original_current = self.runtime.current
        self.current = patch.object(self.runtime, 'current', return_value=self.backend)
        self.current.start()
        self.addCleanup(self.current.stop)

    def group(self):
        with patch.object(comm.dist, 'is_initialized', return_value=False), \
             patch.object(comm.dist, 'init_process_group'), \
             patch.object(comm.dist, 'get_backend', return_value='hccl'):
            return comm.TrainingGroup('hccl', 0, 2,
                lambda: self.runtime.synchronize(1, timeout_ms=30000))

    def bucket(self):
        bucket = Mock()
        bucket.buffer.return_value = torch.tensor([2., 4.])
        return bucket

    def test_consumed_public_signatures(self):
        calls = [('use', ('ascend',), {}), ('device_count', (), {}),
                 ('set_device', (1,), {}), ('current', (), {}),
                 ('synchronize', (1,), {'timeout_ms': 30000})]
        for name, args, kwargs in calls:
            with self.subTest(api=name):
                function = self.original_current if name == 'current' else getattr(self.runtime, name)
                inspect.signature(function).bind(*args, **kwargs)

    def test_hook_forwards_device_and_bounded_sync(self):
        group = self.group()
        with patch.object(comm.dist, 'all_reduce'):
            value = comm.observed_mean_hook(group, self.bucket()).wait()
        self.backend.synchronize.assert_called_once_with(1, timeout_ms=30000)
        torch.testing.assert_close(value, torch.tensor([1., 2.]))
        self.assertEqual(group.calls, 1)

    def test_timeout_blocks_subsequent_collective(self):
        group = self.group()
        failure = TimeoutError('injected bounded device wait')
        self.backend.synchronize.side_effect = failure
        with patch.object(comm.dist, 'all_reduce') as reduce:
            with self.assertRaises(TimeoutError) as caught:
                comm.observed_mean_hook(group, self.bucket())
            self.assertIs(caught.exception, failure)
            with self.assertRaises(RuntimeError):
                comm.observed_mean_hook(group, self.bucket())
            reduce.assert_called_once()
        self.assertEqual(group.state, 'FAILED')
        self.assertEqual(group.calls, 0)

    def test_unified_fatal_error_preserved_without_local_recovery(self):
        group = self.group()
        failure = self.runtime.FlagosError(
            self.runtime.ErrorCategory.L4_FATAL, 'injected original cause',
            location='communication:sync', error_code=507015, mapped=True,
            graded_by='code_map', backend='ascend')
        self.backend.synchronize.side_effect = failure
        with patch.object(comm.dist, 'all_reduce'):
            with self.assertRaises(self.runtime.FlagosError) as caught:
                comm.observed_mean_hook(group, self.bucket())
        self.assertIs(caught.exception, failure)
        self.assertEqual(caught.exception.disposition, 'device_recovery')
        self.assertEqual(group.state, 'FAILED')
        self.backend.recover_device.assert_not_called()
        self.backend.handle_error.assert_not_called()

    def test_group_close_does_not_destroy_device_context(self):
        group = self.group()
        with patch.object(comm.dist, 'is_initialized', return_value=True), \
             patch.object(comm.dist, 'destroy_process_group') as destroy:
            group.close()
            group.close()
        destroy.assert_called_once()
        self.assertEqual(group.state, 'CLOSED')
        self.assertEqual(self.backend.mock_calls, [])


if __name__ == '__main__':
    unittest.main()
