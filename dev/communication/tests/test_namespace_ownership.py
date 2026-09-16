"""Synthetic snapshots only; no server identifiers or device access."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "namespace_ownership", Path(__file__).parents[1] / "probes/namespace_ownership.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NamespaceOwnershipTest(unittest.TestCase):
    devices = "udevid 2 type(davinci)\n has used by container, ns id 0\n"
    namespaces = "ns_id 0 identify abc root_tgid 123 dev_num 2 namespace deadbeef, udev list:\n"
    containers = [{"Name": "/example-owner", "State": {"Pid": 123}}]

    def test_owner_correlated(self):
        row = module.correlate(self.devices, self.namespaces, self.containers)["devices"][0]
        self.assertEqual(row["container"], "example-owner")
        self.assertEqual(row["udevid"], 2)

    def test_missing_container_not_guessed(self):
        row = module.correlate(self.devices, self.namespaces, [])["devices"][0]
        self.assertIsNone(row["container"])

    def test_missing_namespace_unresolved(self):
        row = module.correlate(self.devices, "", self.containers)["devices"][0]
        self.assertEqual(row["state"], "unresolved_namespace_snapshot")

    def test_no_owner_not_declared_available(self):
        row = module.correlate("udevid 2 type(davinci)\n", "", [])["devices"][0]
        self.assertEqual(row["state"], "no_exclusive_owner_in_snapshot")

    def test_shared_not_exclusive(self):
        row = module.correlate("udevid 2 type(davinci)\n shared dev used by ns id: 0 1\n", "", [])["devices"][0]
        self.assertEqual(row["state"], "shared_namespace_registration")

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            module.correlate("", "", [])

    def test_duplicate_namespace_rejected(self):
        with self.assertRaises(ValueError):
            module.correlate(self.devices, self.namespaces * 2, [])


if __name__ == "__main__":
    unittest.main()
