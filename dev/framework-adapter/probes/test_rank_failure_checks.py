import copy
import unittest
from test_rank_failure_npu import assess


class VerdictTests(unittest.TestCase):
    def rows(self):
        rows={r:[dict(event='start',restart_count='0'),dict(event='group_ready',backend='hccl')] for r in (0,1)}
        rows[0]+=[dict(event='injected',stage='backward',monotonic=10),
                  dict(event='adapter',backend='flaggems',reason='error_no_retry'),
                  dict(event='finally',dispatcher_restored=True)]
        rows[1]+=[dict(event='gradient_collective_enter',monotonic=9)]
        return rows
    def test_expected_failure(self):
        self.assertEqual(assess(self.rows(),'backward',1,2)['status'],'pass')
    def test_no_injection_rejected(self):
        rows=self.rows(); rows[0]=[e for e in rows[0] if e['event']!='injected']
        with self.assertRaises(AssertionError): assess(rows,'backward',1,2)
    def test_watchdog_not_success(self):
        for code in (0,124,137):
            with self.subTest(code=code),self.assertRaises(AssertionError): assess(self.rows(),'backward',code,2)
    def test_missing_peer_sync(self):
        rows=self.rows(); rows[1]=rows[1][:2]
        with self.assertRaises(AssertionError): assess(rows,'backward',1,2)
    def test_retries_or_updates_rejected(self):
        for event in [dict(event='optimizer_enter'),dict(event='adapter',backend='native',reason='x'),
                      dict(event='start',restart_count='1')]:
            rows=self.rows(); rows[1].append(event)
            with self.subTest(event=event),self.assertRaises(AssertionError): assess(rows,'backward',1,2)
    def test_timeout_and_restore(self):
        with self.assertRaises(AssertionError): assess(self.rows(),'backward',1,91)
        rows=self.rows(); rows[0][-1]['dispatcher_restored']=False
        with self.assertRaises(AssertionError): assess(rows,'backward',1,2)
    def test_control(self):
        rows={r:[dict(event='start',restart_count='0'),dict(event='group_ready',backend='hccl'),
                 dict(event='adapter',backend='flaggems',stage='backward',reason='completed'),
                 *[dict(event='optimizer_done') for _ in range(3)],dict(event='completed'),
                 dict(event='finally',dispatcher_restored=True)] for r in (0,1)}
        self.assertEqual(assess(rows,'control',0,None)['optimizer_steps'],{'0':3,'1':3})

    def test_guarded_peer_wait(self):
        rows=self.rows()
        rows[1][-1]=dict(event='step_gate_enter',phase='backward',step=0,monotonic=9)
        self.assertEqual(assess(rows,'backward',1,2,guarded=True)['status'],'pass')
        rows[1][-1]['phase']='update'
        with self.assertRaises(AssertionError): assess(rows,'backward',1,2,guarded=True)


if __name__=='__main__': unittest.main()
