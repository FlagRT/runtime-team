import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.paths = tuple(Path(p) for p in ("/checkout", "/output", "/model"))
        self.info = {"Image": run.IMAGE_ID, "State": {"Running": True},
                     "Config": {"Labels": {"owner": "cgu135", "io.flagrt.flag_gems.commit": run.GEMS}},
                     "Mounts": [{"Destination": dst, "Source": str(src), "RW": rw}
                                for dst, src, rw in zip(("/case-workspace", "/case-output", "/case-model"),
                                                       self.paths, (False, True, False))]}

    def test_exact_owned_environment(self):
        run.validate_container(self.info, *self.paths)

    def test_wrong_image(self):
        self.info["Image"] = "sha256:another"
        with self.assertRaises(RuntimeError):
            run.validate_container(self.info, *self.paths)

    def test_other_owner(self):
        self.info["Config"]["Labels"]["owner"] = "another-user"
        with self.assertRaises(RuntimeError):
            run.validate_container(self.info, *self.paths)

    def test_stopped_container(self):
        self.info["State"]["Running"] = False
        with self.assertRaises(RuntimeError):
            run.validate_container(self.info, *self.paths)

    def test_wrong_gems(self):
        self.info["Config"]["Labels"]["io.flagrt.flag_gems.commit"] = "unverified"
        with self.assertRaises(RuntimeError):
            run.validate_container(self.info, *self.paths)

    def test_mount_source_mismatch(self):
        for index in range(3):
            with self.subTest(index=index):
                info = copy.deepcopy(self.info)
                info["Mounts"][index]["Source"] = "/another"
                with self.assertRaises(RuntimeError):
                    run.validate_container(info, *self.paths)


    def test_mount_permissions(self):
        for index in range(3):
            with self.subTest(index=index):
                info = copy.deepcopy(self.info)
                info["Mounts"][index]["RW"] = not info["Mounts"][index]["RW"]
                with self.assertRaises(RuntimeError):
                    run.validate_container(info, *self.paths)


class LauncherTests(unittest.TestCase):
    """Run the entire launcher with fake external commands, never Docker/SSH."""
    def execute(self, operator_status="blocked", baseline_code=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output, model = root / "output", root / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"synthetic-model")
            repo = Path(run.__file__).resolve().parents[1].parents[3]
            info = {"Image": run.IMAGE_ID, "State": {"Running": True},
                    "Config": {"Labels": {"owner": "cgu135", "io.flagrt.flag_gems.commit": run.GEMS}},
                    "HostConfig": {"Devices": [{"PathOnHost": "/dev/davinci2"}]},
                    "Mounts": [{"Destination": dst, "Source": str(src), "RW": rw}
                               for dst, src, rw in zip(("/case-workspace", "/case-output", "/case-model"),
                                                       (repo, output, model), (False, True, False))]}
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as tf:
                member = tarfile.TarInfo("dev/device-context/prototype/runtime/test.txt")
                member.size = 4
                tf.addfile(member, io.BytesIO(b"test"))
            archive = buffer.getvalue()
            calls = []
            def command(args):
                if args[:2] == ["docker", "inspect"]:
                    return json.dumps([info])
                if args[:2] == ["docker", "diff"]:
                    return ""
                if "rev-parse" in args:
                    return "test-commit"
                if "status" in args:
                    return " M synthetic"
                raise AssertionError(args)
            def external(args, **kwargs):
                calls.append(args)
                if "check" in args:
                    return SimpleNamespace(returncode=baseline_code)
                if "--out" in args:
                    destination = Path(args[args.index("--out") + 1])
                    destination = output / destination.relative_to("/case-output")
                    status = "pass" if destination.stem == "fallback" else operator_status
                    if status != "missing":
                        destination.write_text(json.dumps({"status": status}))
                    return SimpleNamespace(returncode=0 if status == "pass" else 1)
                # Import preflight.
                self.assertIn("-c", args)
                return SimpleNamespace(returncode=0)
            argv = ["run.py", "--public-root", str(root), "--output", str(output),
                    "--model", str(model), "--device", "2"]
            with patch("sys.argv", argv), patch.object(run, "command", side_effect=command), \
                    patch.object(run.subprocess, "check_output", return_value=archive), \
                    patch.object(run.subprocess, "run", side_effect=external), \
                    patch.object(run, "RUNTIME_ARCHIVE_SHA", hashlib.sha256(archive).hexdigest()), \
                    patch.object(run, "MODEL_SHA", hashlib.sha256(b"synthetic-model").hexdigest()):
                code = run.main()
            summaries = list(output.glob("*/summary.json"))
            self.assertEqual(len(summaries), 1)
            return code, json.loads(summaries[0].read_text()), calls

    def test_blocked_stops_before_model(self):
        code, summary, calls = self.execute()
        self.assertEqual(code, 3)
        self.assertEqual(summary["overall"], "BLOCKED")
        self.assertEqual(summary["not_run"], ["model-three-steps"])
        probes = [args for args in calls if "--out" in args]
        self.assertEqual(len(probes), 2)
        self.assertTrue(all("FLAGRT_VERIFIED_IMAGE_ID=" + run.IMAGE_ID in args for args in probes))

    def test_pass_requires_all_three_cases(self):
        code, summary, calls = self.execute("pass")
        self.assertEqual(code, 0)
        self.assertEqual([c["status"] for c in summary["results"]], ["PASS"] * 3)
        self.assertEqual(len([args for args in calls if "--out" in args]), 3)

    def test_missing_result_not_pass(self):
        code, summary, _ = self.execute("missing")
        self.assertEqual(code, 2)
        self.assertEqual(summary["overall"], "ERROR")
        self.assertEqual(summary["not_run"], ["model-three-steps"])

    def test_baseline_failure_starts_no_probe(self):
        code, summary, calls = self.execute(baseline_code=1)
        self.assertEqual(code, 2)
        self.assertEqual(summary["results"], [])
        self.assertFalse(any(args[:2] == ["docker", "exec"] for args in calls))

if __name__ == "__main__":
    unittest.main()
