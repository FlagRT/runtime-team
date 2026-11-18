"""Host-only regression for vendor DDP log/actual-reducer metadata divergence."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
import distributed as comm


def bucket(index, count):
    value=Mock();value.index.return_value=index
    value.buffer.return_value=torch.ones(count)
    return value


class ActualBucketTraceTests(unittest.TestCase):
    def test_unmeasured_warmup_not_recorded(self):
        trace=comm.ObservedBucketTrace();value=bucket(0,2)
        trace.record(value,value.buffer());value.index.assert_not_called()
        self.assertEqual(trace.steps,[])

    def test_two_actual_buckets_recorded_per_step(self):
        trace=comm.ObservedBucketTrace()
        for step in range(2):
            trace.begin(step)
            for value in (bucket(1,1),bucket(0,2)):
                trace.record(value,value.buffer())
            trace.end()
        self.assertEqual(trace.sizes(),[8,4])
        self.assertEqual([item['step'] for item in trace.steps],[0,1])

    def test_no_empty_measured_backward(self):
        trace=comm.ObservedBucketTrace();trace.begin(0)
        with self.assertRaises(ValueError):trace.end()

    def test_no_overlapping_or_out_of_order_backward(self):
        trace=comm.ObservedBucketTrace()
        for step in (-1,1,True):
            with self.assertRaises(ValueError):trace.begin(step)
        trace.begin(0)
        with self.assertRaises(ValueError):trace.begin(0)

    def test_duplicate_bucket_in_step_rejected(self):
        trace=comm.ObservedBucketTrace();trace.begin(0);value=bucket(0,2)
        trace.record(value,value.buffer())
        with self.assertRaises(ValueError):trace.record(value,value.buffer())

    def test_noninteger_or_negative_index_rejected(self):
        for index in (True,-1,'0'):
            trace=comm.ObservedBucketTrace();trace.begin(0);value=bucket(index,2)
            with self.assertRaises(ValueError):trace.record(value,value.buffer())

    def test_incomplete_or_changing_plan_rejected(self):
        trace=comm.ObservedBucketTrace()
        with self.assertRaises(ValueError):trace.sizes()
        for step,count in ((0,2),(1,3)):
            trace.begin(step);value=bucket(0,count);trace.record(value,value.buffer());trace.end()
        with self.assertRaises(ValueError):trace.sizes()

    def test_successful_hook_records_real_bucket_after_completion(self):
        with patch.object(comm.dist,'is_initialized',return_value=False), \
             patch.object(comm.dist,'init_process_group'), \
             patch.object(comm.dist,'get_backend',return_value='gloo'):
            group=comm.TrainingGroup('gloo',0,2,Mock())
        group.bucket_trace.begin(0);value=bucket(0,3)
        with patch.object(comm.dist,'all_reduce'):
            comm.observed_mean_hook(group,value).wait()
        group.bucket_trace.end()
        self.assertEqual(group.bucket_trace.sizes(),[12])
        self.assertEqual(group.calls,1)


if __name__=='__main__':unittest.main()
