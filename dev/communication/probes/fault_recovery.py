"""Real worker-exit -> runtime classification -> fresh-group recovery evidence.

Classification may be a conservative fallback; it is never promoted to a
vendor-confirmed error code. This does not reset devices or replay a failed op.
"""
import argparse
import json
from pathlib import Path
import sys

from run_acceptance import run

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run-root", type=Path, required=True)
p.add_argument("--runtime-root", type=Path, required=True)
args = p.parse_args()
here = Path(__file__).resolve().parent
fault = run(["torchrun", "--standalone", "--nproc_per_node=2", str(here / "fault_worker.py")],
            args.run_root, iterations=1, timeout=45)
fault_dir = Path(fault["output_dir"])
marker = json.loads((fault_dir / "fault_marker_rank0.json").read_text())
assert marker["collective_verified"] and marker["run_id"] == fault["run_id"]
assert fault["verdict"] == "FAIL" and fault["returncode"] != 0
log = (fault_dir / "launcher.log").read_text()
assert "42" in log  # retain complete launcher log as the primary evidence
sys.path.insert(0, str(args.runtime_root))
import torch_fl  # noqa: F401
import runtime
runtime.use("flagos")
error = runtime.translate_error(RuntimeError(log), location="communication/worker_exit")
classified = {"category": error.category.value, "disposition": error.disposition,
              "mapped": error.mapped, "graded_by": error.graded_by,
              "is_grade_confident": error.is_grade_confident}
recovery = run(["torchrun", "--standalone", "--nproc_per_node=2",
                str(here / "communication_correctness.py")],
               args.run_root, iterations=100, timeout=180)
result = {"fault_run_id": fault["run_id"], "fault_returncode": fault["returncode"],
          "fault_timed_out": fault["timed_out"], "classification": classified,
          "recovery_action": "new_process_group_not_same_group_replay",
          "recovery_run_id": recovery["run_id"], "recovery_verdict": recovery["verdict"],
          "note": "Prototype evidence; downstream monitoring confirmation still required."}
(args.run_root / "fault_recovery.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
raise SystemExit(0 if recovery["verdict"] == "PASS" else 1)
