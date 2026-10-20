"""CPU-only policy tests; no torch import or accelerator execution."""
from dataclasses import replace
import unittest
from silu_policy import Capability, Environment, InputSpec, SiluPolicy


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.env = Environment("910C", "2.10.0+cpu", "2.10.0", "locked-test-revision",
                               "test-image", "test-triton", "test-build")
        self.spec = InputSpec("npu", "torch.float32", (1, 32, 3072), True)
        self.capability = Capability(self.env, "silu", self.spec.dtype, self.spec.shape,
                                     ("forward", "backward"), "test-only-evidence", "passed")
        self.policy = SiluPolicy(self.env, ((1, 32, 3072),), experimental=True,
                                 capabilities=(self.capability,))

    def test_admitted_experimental_case(self):
        self.assertIsNone(self.policy.reason(self.env, self.spec))

    def test_default_closed(self):
        self.assertEqual(replace(self.policy, experimental=False).reason(self.env, self.spec),
                         "experimental_disabled")

    def test_environment_fields(self):
        for field in self.env.__dict__:
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

    def test_incomplete_identity_even_when_expected_matches(self):
        for field in self.env.__dict__:
            env = replace(self.env, **{field: ""})
            self.assertEqual(replace(self.policy, expected=env).reason(env, self.spec),
                             "environment_unverified")

    def test_missing_or_stale_evidence(self):
        for changes in ({"status": "unknown"}, {"revision": 0}, {"evidence": ""},
                        {"stages": ("forward",)}, {"operator": "gelu"},
                        {"shape": (2, 3)}, {"dtype": "torch.float16"}):
            with self.subTest(changes=changes):
                policy = replace(self.policy, capabilities=(replace(self.capability, **changes),))
                self.assertEqual(policy.reason(self.env, self.spec), "capability_unverified")
        self.assertEqual(replace(self.policy, capabilities=()).reason(self.env, self.spec),
                         "capability_unverified")

    def test_negative_evidence_overrides_pass(self):
        for status in ("failed", "revoked"):
            policy = replace(self.policy, capabilities=(self.capability,
                             replace(self.capability, status=status)))
            self.assertEqual(policy.reason(self.env, self.spec), "capability_denied")

    def test_unknown_markers_are_not_identity(self):
        for marker in ("unknown", "unavailable", " unverified ", "UNKNOWN"):
            env = replace(self.env, compiler_build=marker)
            self.assertEqual(replace(self.policy, expected=env).reason(env, self.spec),
                             "environment_unverified")

    def test_known_v3_failure_cannot_be_overridden_by_positive_record(self):
        env = replace(self.env, image_id="sha256:e13a15d0d02b86e9cd7b1656df3c6b52838c258f270b6b894a91c2b074da5f15")
        policy = replace(self.policy, expected=env,
                         capabilities=(replace(self.capability, environment=env),))
        self.assertEqual(policy.reason(env, self.spec), "known_compile_failure")


if __name__ == "__main__":
    unittest.main()
