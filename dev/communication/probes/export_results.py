"""Export allowlisted experiment values; never copy host logs or absolute paths.

Review generated JSON before publishing. Input is a private evidence directory.
SHA-256 fingerprints let the owner correlate summaries with retained originals.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def select(row, keys):
    return {key: row[key] for key in keys.split() if key in row}


def fingerprint(identity):
    return {"inner_name": identity.get("inner_name"),
            "libraries": [{"name": Path(lib["path"]).name, "sha256": lib["sha256"]}
                          for lib in identity["libraries"]]}


def export(root):
    records = []
    for path in sorted(root.rglob("*.json")):
        row = json.loads(path.read_text())
        name = path.name
        result = None
        if name.startswith("matrix_rank"):
            result = select(row, "rank iterations sizes mode phase counting_unit elapsed_s passed total")
            result["identity"] = fingerprint(row["identity"])
            groups = {}
            for r in row["checks"]:
                key = (r["operation"], r["dtype"], r["elements"])
                group = groups.setdefault(key, dict(operation=key[0], dtype=key[1], elements=key[2],
                                                    total=0, passed=0, max_abs_diff=0.0, nonfinite_diff=False))
                group["total"] += 1
                group["passed"] += int(r["ok"])
                diff = r["max_abs_diff"]
                if math.isfinite(diff):
                    group["max_abs_diff"] = max(group["max_abs_diff"], diff)
                else:
                    group["nonfinite_diff"] = True
            result["groups"] = list(groups.values())
        elif name.startswith("communication_correctness_rank"):
            result = select(row, "rank schema_version run_id counting_unit phase world_size iterations dtypes calls_per_operation total_calls passed_calls failed_calls immediate_is_completed_true elapsed_s failures verdict")
        elif name.startswith("train_leg_result_rank"):
            result = select(row, "rank world_size perf loss_curve verdict passed total reference_sha256 phase transformers_version attribution")
            result["checks"] = {key: {"ok": value["ok"]} for key, value in row["checks"].items()}
            if "communication_identity" in row:
                result["identity"] = fingerprint(row["communication_identity"])
        elif name == "acceptance.json":
            result = select(row, "numeric_ok cleanup_ok run_id returncode timed_out forced_cleanup validation_scope verdict")
        elif name == "fault_recovery.json":
            result = select(row, "fault_run_id fault_returncode fault_timed_out classification recovery_action recovery_run_id recovery_verdict")
        if result is not None:
            records.append({"source": str(path.relative_to(root)),
                            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "result": result})
    if not records:
        raise ValueError("no recognized experiment results")
    for name, keys in {
        "event.log": "created destroyed recorded waited iterations scope",
        "acl_release.log": "acl_init get_device_count device_count acl_finalize",
    }.items():
        path = root / name
        if path.exists():
            records.append({"source": name,
                            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "result": select(json.loads(path.read_text()), keys)})
    return {"schema_version": 1, "scope": "sanitized_experiment_results_not_raw_logs", "records": records}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = export(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
