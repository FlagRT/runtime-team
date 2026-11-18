"""Host-only logic and evidence-gate tests, not accelerator acceptance."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from parallel_minimal import (MinimalPlan, Transport, initial_weights, step_data,
                              reference, tp_step, pp_step)
from run_parallel_matrix import CASES, summarize
from run_p800_selected import choose_idle_devices, validate_health


class ParallelMathTests(unittest.TestCase):
    def test_native_health_requires_all_checks_and_matching_uuid(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'health.json'
            record={'verdict':'PASS','uuid':'GPU-example','checks':{
                'identity':True,'fill_readback':True,'matmul':True,'event_completed':True}}
            path.write_text(json.dumps(record))
            validate_health(path,'example')
            with self.assertRaises(ValueError):validate_health(path,'another-device')
            record['checks']['matmul']=False
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):validate_health(path,'example')

    def test_interrupted_native_health_cannot_start_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'health.json'
            path.write_text(json.dumps({'verdict':'ERROR','uuid':'GPU-example',
                                        'stage':'fill_and_synchronize','checks':{'identity':True}}))
            with self.assertRaises(ValueError):validate_health(path,'example')

    def test_device_exclusion_never_broadens_pair_candidates(self):
        self.assertEqual(choose_idle_devices(set(range(8)), 'pair', [4,5]), (6,7))
        self.assertIsNone(choose_idle_devices({0,2,7}, 'pair'))
        self.assertIsNone(choose_idle_devices({4,5}, 'pair', [4]))

    def test_busy_peer_is_not_selected(self):
        self.assertIsNone(choose_idle_devices({4}, 'pair'))
        self.assertEqual(choose_idle_devices({4,5}, 'single', [5]), (4,))

    def test_plan_rejects_invalid_mesh(self):
        with self.assertRaises(ValueError):
            MinimalPlan('tp').validate(world_size=4)
        with self.assertRaises(ValueError):
            MinimalPlan('data_parallel').validate()

    def test_plan_rejects_invalid_shapes(self):
        for kwargs in ({'hidden_dim': 31}, {'microbatches': 3}, {'batch_size': 0},
                       {'input_dim': True}, {'output_dim': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MinimalPlan('tp', **kwargs).validate()

    def test_hidden_shards_cover_exactly_once(self):
        plan = MinimalPlan('tp').validate()
        sections = [plan.hidden_slice(r) for r in (0, 1)]
        self.assertEqual([v for s in sections for v in range(s.start, s.stop)], list(range(32)))
        with self.assertRaises(ValueError):
            plan.hidden_slice(2)

    def test_tp_sharded_gradients_match_autograd(self):
        for hidden_dim in (32, 64):
            plan = MinimalPlan('tp', hidden_dim=hidden_dim)
            a, b = initial_weights(plan, 1111)
            x, y = step_data(plan, 1112)
            expected, da, db, _ = reference(a, b, x, y)
            for rank in (0, 1):
                with self.subTest(hidden=hidden_dim, rank=rank):
                    transport = Mock()
                    transport.call.side_effect = lambda operation, tensor, **kw: tensor.copy_(expected)
                    section = plan.hidden_slice(rank)
                    out, local_a, local_b = tp_step(plan, rank, a[section], b[:, section], x, y, transport)
                    torch.testing.assert_close(out, expected)
                    torch.testing.assert_close(local_a, da[section])
                    torch.testing.assert_close(local_b, db[:, section])
                    transport.call.assert_called_once()

    def test_pp_microbatch_gradient_scaling_and_order(self):
        for microbatches in (2, 4):
            plan = MinimalPlan('pp', microbatches=microbatches)
            a, b = initial_weights(plan, 1111)
            x, y = step_data(plan, 1112)
            expected, da, db, _ = reference(a, b, x, y)
            hidden = torch.relu(x @ a.T)
            backward = (2 * (expected - y) / y.numel()) @ b
            count = plan.batch_size // microbatches
            for rank in (0, 1):
                with self.subTest(microbatches=microbatches, rank=rank):
                    sent, received = [], []
                    def transfer(operation, tensor, **kwargs):
                        index = len(received) if operation == 'recv' else len(sent)
                        section = slice(index * count, (index + 1) * count)
                        if operation == 'recv':
                            tensor.copy_(backward[section] if rank == 0 else hidden[section])
                            received.append(index)
                        else:
                            torch.testing.assert_close(tensor, hidden[section] if rank == 0 else backward[section])
                            sent.append(index)
                    transport = Mock()
                    transport.call.side_effect = transfer
                    output, gradient = pp_step(plan, rank, a if rank == 0 else b, x, y, transport)
                    torch.testing.assert_close(gradient, da if rank == 0 else db)
                    self.assertEqual(sent, list(range(microbatches)))
                    self.assertEqual(received, list(range(microbatches)))
                    if rank == 1:
                        torch.testing.assert_close(output, expected)

    def test_transport_does_not_count_failed_communication(self):
        group = Mock()
        group.collective.side_effect = RuntimeError('injected')
        transport = Transport(group)
        with self.assertRaises(RuntimeError):
            transport.call('send', torch.ones(4), dst=1)
        self.assertEqual(transport.calls['send'], 0)
        self.assertEqual(transport.input_bytes['send'], 0)

    def test_transport_rejects_unknown_operation(self):
        transport = Transport(Mock())
        with self.assertRaises(ValueError):
            transport.call('gather_all', torch.ones(4))


class ParallelEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        manifest = {'steps': 50, 'repeats': 3, 'scope': 'CPU/Gloo host-only',
                    'device_vendor': 'cpu', 'collective': 'gloo',
                    'prototype_revision': 'fixture-only', 'source_sha256': {'parallel_minimal.py': 'fixture'}, 'runs': []}
        for repeat in (1, 2, 3):
            for case, mode, hidden, microbatches in CASES:
                name = f'{repeat:02d}-{case}'
                manifest['runs'].append({'case': case, 'repeat': repeat, 'name': name,
                                         'exit_code': 0, 'timed_out': False})
                directory = self.root / name
                directory.mkdir()
                for rank in (0, 1):
                    plan = MinimalPlan(mode, hidden_dim=hidden, microbatches=microbatches)
                    record = {'rank': rank, 'world_size': 2, 'steps': 50, 'verdict': 'PASS',
                              'group_state': 'CLOSED', 'device_vendor': 'cpu', 'backend': 'gloo',
                              'prototype_revision': 'fixture-only', 'source_sha256': 'fixture',
                              'plan': asdict(plan), 'tolerance': {'atol': 1e-5, 'rtol': 1e-4},
                              'checks': [{'step': step, 'reference_loss': 1.0, 'forward_max_abs': 0.0,
                                          'tensor_check_max_abs': 0.0, 'parameter_max_abs': 0.0} for step in range(50)],
                              'transport_calls': {'all_reduce': 51 if mode == 'tp' else 1,
                                                  'send': 50 * microbatches if mode == 'pp' else 0,
                                                  'recv': 50 * microbatches if mode == 'pp' else 0},
                              'transport_input_bytes': {'all_reduce': 25604 if mode == 'tp' else 4,
                                                        'send': 102400 if mode == 'pp' else 0,
                                                        'recv': 102400 if mode == 'pp' else 0}}
                    self.write(directory / f'rank{rank}.json', record)
        self.write(self.root / 'execution.json', manifest)

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def mutate_record(self, change):
        target = self.root / '01-tp-h32/rank0.json'
        value = json.loads(target.read_text())
        change(value)
        self.write(target, value)
        with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
            summarize(self.root)

    def test_complete_fixture(self):
        result = summarize(self.root)
        self.assertEqual(result['launches'], 12)
        self.assertEqual(result['rank_records'], 24)
        self.assertEqual(result['verdict'], 'PASS')

    def test_missing_launch_rejected(self):
        path = self.root / 'execution.json'
        value = json.loads(path.read_text())
        value['runs'].pop()
        self.write(path, value)
        with self.assertRaises(ValueError):
            summarize(self.root)

    def test_duplicate_launch_rejected(self):
        path = self.root / 'execution.json'
        value = json.loads(path.read_text())
        value['runs'][-1] = value['runs'][0]
        self.write(path, value)
        with self.assertRaises(ValueError):
            summarize(self.root)

    def test_failure_exit_rejected(self):
        path = self.root / 'execution.json'
        value = json.loads(path.read_text())
        value['runs'][0]['exit_code'] = -9
        self.write(path, value)
        with self.assertRaises(ValueError):
            summarize(self.root)

    def test_incomplete_steps_rejected(self):
        self.mutate_record(lambda r: r['checks'].pop())

    def test_wrong_placement_rejected(self):
        self.mutate_record(lambda r: r['plan'].update(hidden_dim=64))

    def test_wrong_source_rejected(self):
        self.mutate_record(lambda r: r.update(source_sha256='different'))

    def test_wrong_revision_rejected(self):
        self.mutate_record(lambda r: r.update(prototype_revision='different'))

    def test_nonfinite_result_rejected(self):
        self.mutate_record(lambda r: r['checks'][0].update(forward_max_abs=float('nan')))

    def test_large_error_rejected(self):
        self.mutate_record(lambda r: r['checks'][0].update(parameter_max_abs=1e-3))

    def test_wrong_transport_counts_rejected(self):
        self.mutate_record(lambda r: r['transport_calls'].update(all_reduce=50))

    def test_wrong_transport_bytes_rejected(self):
        self.mutate_record(lambda r: r['transport_input_bytes'].update(all_reduce=4))


if __name__ == '__main__':
    unittest.main()
