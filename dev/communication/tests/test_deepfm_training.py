"""Host-only checks for the communication integration, not device conformance."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'training'))
import distributed as comm


class DeepFMCommunicationTests(unittest.TestCase):
    def test_shards_disjoint_complete(self):
        spans = [range(*comm.shard_bounds(64,r,2)) for r in range(2)]
        self.assertFalse(set(spans[0]) & set(spans[1]))
        self.assertEqual(sorted([i for s in spans for i in s]),list(range(64)))

    def test_uneven_shards_rejected(self):
        with self.assertRaises(ValueError):comm.shard_bounds(63,0,2)

    def test_small_batch_rejected(self):
        with self.assertRaises(ValueError):comm.shard_bounds(2,0,2)

    def test_invalid_rank_rejected(self):
        with self.assertRaises(ValueError):comm.shard_bounds(64,2,2)

    def test_invalid_world_rejected(self):
        with self.assertRaises(ValueError):comm.shard_bounds(64,0,0)

    def group(self):
        with patch.object(comm.dist,'is_initialized',return_value=False), \
             patch.object(comm.dist,'init_process_group'), \
             patch.object(comm.dist,'get_backend',return_value='gloo'):
            return comm.TrainingGroup('gloo',0,2,Mock())

    def test_backend_no_silent_fallback(self):
        with self.assertRaises(ValueError):comm.TrainingGroup('unknown',0,2,Mock())

    def test_existing_group_not_owned(self):
        with patch.object(comm.dist,'is_initialized',return_value=True):
            with self.assertRaises(RuntimeError):comm.TrainingGroup('gloo',0,2,Mock())

    def test_invalid_timeout_rejected(self):
        with self.assertRaises(ValueError):comm.TrainingGroup('gloo',0,2,Mock(),0)

    def test_wrong_actual_backend_rejected(self):
        with patch.object(comm.dist,'is_initialized',return_value=False), \
             patch.object(comm.dist,'init_process_group'), \
             patch.object(comm.dist,'get_backend',return_value='nccl'):
            with self.assertRaises(RuntimeError):comm.TrainingGroup('gloo',0,2,Mock())

    def test_hook_divides_sum_once(self):
        group=self.group();tensor=torch.tensor([2.,4.]);bucket=Mock();bucket.buffer.return_value=tensor
        with patch.object(comm.dist,'all_reduce',side_effect=lambda t,**kw:t.mul_(2)):
            result=comm.observed_mean_hook(group,bucket).wait()
        torch.testing.assert_close(result,torch.tensor([2.,4.]))
        self.assertEqual(group.calls,1);self.assertEqual(group.payload_bytes,8)
        group.synchronize.assert_called_once()

    def test_failed_collective_marks_failed(self):
        group=self.group();bucket=Mock();bucket.buffer.return_value=torch.ones(2)
        with patch.object(comm.dist,'all_reduce',side_effect=RuntimeError('transport failure')):
            with self.assertRaises(RuntimeError):comm.observed_mean_hook(group,bucket)
        self.assertEqual(group.state,'FAILED');self.assertEqual(group.calls,0)
        with self.assertRaises(RuntimeError):group.require_open()

    def test_sync_failure_marks_failed(self):
        group=self.group();group.synchronize.side_effect=TimeoutError('device wait')
        bucket=Mock();bucket.buffer.return_value=torch.ones(2)
        with patch.object(comm.dist,'all_reduce'):
            with self.assertRaises(TimeoutError):comm.observed_mean_hook(group,bucket)
        self.assertEqual(group.state,'FAILED')

    def test_close_idempotent_and_rejects_use(self):
        group=self.group()
        with patch.object(comm.dist,'is_initialized',return_value=True), \
             patch.object(comm.dist,'destroy_process_group') as destroy:
            group.close();group.close();destroy.assert_called_once()
        with self.assertRaises(RuntimeError):group.require_open()

    def test_reset_stats_excludes_warmup(self):
        group=self.group();group.calls=5;group.payload_bytes=50;group.seconds=2
        group.reset_stats();self.assertEqual(group.calls,0)
        self.assertEqual(group.payload_bytes,0);self.assertEqual(group.seconds,0)


if __name__=='__main__':unittest.main()
