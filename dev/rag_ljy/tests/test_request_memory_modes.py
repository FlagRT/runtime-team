"""Request-only experiments must not confuse reservations with physical usage."""
import importlib.util
from pathlib import Path
import sys
import unittest

scripts = Path(__file__).resolve().parents[1]/'scripts/checks'
sys.path.insert(0, str(scripts))
import check_request_memory_modes as experiment
from test_watermarks_modes import health


class RequestOnlyTests(unittest.TestCase):
    def test_reservation_pressure_is_not_actual_threshold(self):
        s = {'t': 1, 'health': health(22, 65)}
        self.assertEqual(experiment.assess([s], [])['status'], 'threshold_not_reached')

    def test_actual_high_alone_is_not_full_pass(self):
        s = {'t': 1, 'health': health(86)}
        self.assertEqual(experiment.assess([s], [])['status'], 'incomplete')

    def test_capacity_rejections_are_separate(self):
        r = {'status': 503, 'reason': 'Request capacity is full; retry later'}
        result = experiment.assess([{'t': 1, 'health': health(22)}], [r])
        self.assertEqual(result['status'], 'threshold_not_reached')
        self.assertEqual(result['rejection_reasons'][r['reason']], 1)

    def test_missing_middle_not_passed(self):
        samples = [{'t': 1, 'health': health(86)},
                   {'t': 3, 'health': health(22, reason='memory_available')}]
        self.assertEqual(experiment.assess(samples, [])['status'], 'incomplete')

    def test_script_has_no_device_allocation_or_cache_manipulation(self):
        import ast
        tree = ast.parse((scripts/'check_request_memory_modes.py').read_text())
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertFalse(set(calls) & {'empty', 'zeros', 'ones', 'empty_cache', 'synchronize'})


if __name__ == '__main__':
    unittest.main()


def test_measured_policy_does_not_require_reservations_below_low_watermark():
    h = health(75, 20, reason='memory_available')
    h['memory_admission']['watermark_basis'] = 'device_measured_usage'
    assert experiment.measured_phase_ok(h, 'recovery')


def test_measured_policy_counts_exactly_85_as_high():
    h = health(85)
    h['memory_admission']['watermark_basis'] = 'device_measured_usage'
    assert experiment.measured_phase_ok(h, 'high')


def test_successful_list_response_is_valid_evidence_input():
    result = experiment.assess([{'t':1,'health':health(40,reason='memory_available')}],
                              [{'status':200,'body':[{'text':'answer'}],'sent':1,'completed':2}])
    assert result['status'] == 'threshold_not_reached'
