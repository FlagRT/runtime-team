"""CPU-only policy tests; no torch import or accelerator execution."""
from dataclasses import replace
import unittest
from silu_policy import Environment, InputSpec, SiluPolicy


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.env = Environment("910C", "2.10.0+cpu", "2.10.0", "locked-test-revision")
        self.spec = InputSpec("npu", "torch.float32", (1, 32, 3072), True)
        self.policy = SiluPolicy(self.env, ((1, 32, 3072),), experimental=True)

    def test_admitted_experimental_case(self):
        self.assertIsNone(self.policy.reason(self.env, self.spec))

    def test_default_closed(self):
        self.assertEqual(replace(self.policy, experimental=False).reason(self.env, self.spec),
                         "experimental_disabled")

    def test_environment_fields(self):
        for field in ("chip", "torch", "torch_npu", "gems_revision"):
            with self.subTest(field=field):
                self.assertEqual(self.policy.reason(replace(self.env, **{field: "other"}), self.spec),
                                 "environment_mismatch")

    def test_input_rejections(self):
        for changes, reason in [({"device": "cpu"}, "device"),
                                ({"dtype": "torch.bfloat16"}, "dtype"),
                                ({"shape": (2, 32, 3072)}, "shape"),
                                ({"shape": ()}, "empty_or_scalar"),
                                ({"shape": (0, 32, 3072)}, "empty_or_scalar"),
                                ({"contiguous": False}, "layout"),
                                ({"strided": False}, "layout"),
                                ({"exact_tensor": False}, "tensor_subclass"),
                                ({"inplace": True}, "inplace"),
                                ({"autocast": True}, "autocast")]:
            with self.subTest(changes=changes):
                self.assertEqual(self.policy.reason(self.env, replace(self.spec, **changes)), reason)

    def test_empty_allowlist(self):
        self.assertEqual(replace(self.policy, shapes=()).reason(self.env, self.spec), "shape")


if __name__ == "__main__":
    unittest.main()
