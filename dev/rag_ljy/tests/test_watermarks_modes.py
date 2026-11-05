"""Offline evidence-classification tests; no device or service access."""
import importlib.util
from pathlib import Path
import unittest
import tempfile

spec = importlib.util.spec_from_file_location('watermarks_modes', Path(__file__).resolve().parents[1]/'scripts/checks/check_watermarks_modes.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def health(used, reserved=0, reason='memory_pressure', free=None):
    return {'device_memory': {'status': 'ok', 'stale': False,
            'stats': {'total_mb': 100, 'used_mb': used, 'free_mb': 100-used if free is None else free}},
            'memory_admission': {'reserved_mib': reserved, 'reason': reason,
            'status': 'ready' if reason == 'memory_available' else 'paused'}}


class EvidenceTests(unittest.TestCase):
    def test_own_timeout_is_ignored_but_other_tests_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processes = {
                10: (0, ['python', 'run_pressure_suite.py']),
                20: (10, ['timeout', '25m', 'python', 'check_watermarks_modes.py', '--run']),
                30: (20, ['python', 'check_watermarks_modes.py', '--run']),
                40: (0, ['timeout', '25m', 'python', 'check_watermarks_modes.py', '--run']),
                50: (40, ['python', 'check_watermarks_modes.py', '--run']),
                60: (0, ['python', 'check_online_pressure.py']),
            }
            for pid, (parent, argv) in processes.items():
                entry = root/str(pid)
                entry.mkdir()
                (entry/'status').write_text(f'PPid:\t{parent}\n')
                (entry/'cmdline').write_bytes(('\0'.join(argv)+'\0').encode())
            self.assertEqual({p['pid'] for p in m.conflicting_processes(root, 30)}, {10, 40, 50, 60})

    def test_high_requires_actual_above_threshold(self):
        self.assertFalse(m.phase_ok(health(85), 'high'))
        self.assertFalse(m.phase_ok(health(70, 20), 'high'))
        self.assertTrue(m.phase_ok(health(85.1), 'high'))

    def test_middle_requires_effective_middle_too(self):
        self.assertTrue(m.phase_ok(health(80), 'middle'))
        self.assertFalse(m.phase_ok(health(80, 6), 'middle'))
        self.assertFalse(m.phase_ok(health(75), 'middle'))
        self.assertFalse(m.phase_ok(health(80, reason='memory_available'), 'middle'))

    def test_recovery_includes_pending_reservations(self):
        self.assertTrue(m.phase_ok(health(73, 2, 'memory_available'), 'recovery'))
        self.assertFalse(m.phase_ok(health(73, 3, 'memory_available'), 'recovery'))

    def test_stale_cannot_pass(self):
        h = health(86)
        h['device_memory']['stale'] = True
        self.assertFalse(m.phase_ok(h, 'high'))

    def test_conservative_measured_usage(self):
        self.assertEqual(m.ratios(health(70, 2, free=15)), (.85, .87))

    def test_capacity_and_budget_rejections_are_not_watermark_evidence(self):
        for detail in ('Request capacity is full; retry later', {'reason': 'request_budget'}, {'reason': 'stale_telemetry'}):
            self.assertFalse(m.response_ok({'status': 503, 'headers': {'retry-after': '1'}, 'body': {'detail': detail}}, 'high'))

    def test_pressure_needs_retry_after(self):
        r = {'status': 503, 'body': {'detail': {'reason': 'memory_pressure'}}}
        self.assertFalse(m.response_ok(r, 'high'))
        r['headers'] = {'retry-after': '1'}
        self.assertTrue(m.response_ok(r, 'high'))
        self.assertFalse(m.response_ok(r, 'recovery'))


if __name__ == '__main__':
    unittest.main()
