"""Single-entry regression in an already coordinated, owned v3 container.

Never starts/stops containers, changes packages, or retries failed target ops.
Full logs and JSON stay in --output; only excerpts belong in the repository.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

IMAGE_ID = "sha256:e13a15d0d02b86e9cd7b1656df3c6b52838c258f270b6b894a91c2b074da5f15"
GEMS = "f7ae8e6b934a33ec1ccaf2c9aae71edf205f8fb4"
RUNTIME = "0c59b9549d45d55b959294869f605a30dc3789c9"
RUNTIME_ARCHIVE_SHA = "ffb5231cb58a5cad804f60a967927a7f31c2373a8bbd5c0858a1d1a29705b6e7"
MODEL_SHA = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
PYTHON = "/usr/local/python3.11.15/bin/python3"
PROBES = "dev/framework-adapter/probes"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(args):
    return subprocess.check_output(args, text=True, timeout=120)


def validate_container(info, repo, output, model):
    if info["Image"] != IMAGE_ID or not info["State"]["Running"]:
        raise RuntimeError("Container must run the exact locked v3 image")
    if (info["Config"].get("Labels") or {}).get("owner") != "cgu135":
        raise RuntimeError("Refusing another person's container; coordinate a test slot first")
    if (info["Config"].get("Labels") or {}).get("io.flagrt.flag_gems.commit") != GEMS:
        raise RuntimeError("Unexpected baked FlagGems commit label")
    mounts = {m["Destination"]: m for m in info["Mounts"]}
    for dst, source, readonly in (("/case-workspace", repo, True),
                                  ("/case-output", output, False),
                                  ("/case-model", model, True)):
        item = mounts.get(dst, {})
        if Path(item.get("Source", "")).resolve() != source or item.get("RW") is not (not readonly):
            raise RuntimeError(f"Unexpected mount or write permission: {dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--public-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="same host directory mounted at /case-output")
    p.add_argument("--model", type=Path, required=True, help="same verified directory mounted at /case-model")
    p.add_argument("--device", type=int, required=True, help="coordinated physical device visible to this container")
    p.add_argument("--container", default="cgu135-framework-adapter-train-v3")
    p.add_argument("--admission-records", type=Path,
                   help="Reviewed bundle JSON under repo, mounted read-only; absent means target BLOCKED")
    args = p.parse_args()
    if args.device < 0:
        p.error("device must be nonnegative")
    case = Path(__file__).resolve().parents[1]
    repo = case.parents[3]
    output, model = args.output.resolve(), args.model.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir()
    result = {"overall": "ERROR", "results": [], "runtime_commit": RUNTIME,
              "scope": "experimental SiLU; single device; 3 model steps; not complete three-level fallback"}
    try:
        info = json.loads(command(["docker", "inspect", args.container]))[0]
        validate_container(info, repo, output, model)
        mapped = [d['PathOnHost'] for d in info['HostConfig']['Devices']
                  if d['PathOnHost'].startswith('/dev/davinci')
                  and d['PathOnHost'] != '/dev/davinci_manager']
        if mapped != [f'/dev/davinci{args.device}']:
            raise RuntimeError('Container must map exactly the requested physical device')
        result['physical_device'] = args.device
        result['container_device'] = 0
        changes = command(["docker", "diff", args.container])
        if any("site-packages/" in line and "__pycache__" not in line
               for line in changes.splitlines()):
            raise RuntimeError("Container packages differ from image; use a clean owned container")
        result["image_id"] = info["Image"]
        env = dict(os.environ, WORKSPACE_ROOT=str(repo), FLAGRT_WORKSPACE_ROOT=str(repo),
                   FLAGRT_CASE_OUTPUT=str(output), FLAGRT_MODEL_DIR=str(model),
                   FLAGRT_MODELS_DIR=str(model.parent))
        import sys
        with (run / "baseline-check.log").open("w") as log:
            check = subprocess.run([sys.executable, str(args.public_root.resolve() / "dev/lib/baseline.py"),
                                    "check", str(case / "demo")], env=env,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=120)
        if check.returncode:
            raise RuntimeError("Baseline check failed; no NPU test started")
        if sha(model / "model.safetensors") != MODEL_SHA:
            raise RuntimeError("Qwen model differs from verified 20261014 checkpoint")
        # Runtime is an explicit pinned source snapshot, not the stale personal
        # branch's device-context subtree and not an unrecorded pip installation.
        archive = subprocess.check_output(["git", "-C", str(repo), "archive", RUNTIME,
                                           "dev/device-context/prototype/runtime"], timeout=60)
        if hashlib.sha256(archive).hexdigest() != RUNTIME_ARCHIVE_SHA:
            raise RuntimeError("Pinned runtime archive identity differs")
        runtime_dir = run / "runtime-source"
        runtime_dir.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
            for member in tf.getmembers():
                if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise RuntimeError("Unexpected runtime archive member")
                target = (runtime_dir / member.name).resolve()
                if runtime_dir not in target.parents:
                    raise RuntimeError("Unsafe runtime archive path")
            tf.extractall(runtime_dir)
        source_paths = sorted((repo / PROBES).glob("*.py")) + sorted(
            (repo / "dev/framework-adapter/src/flagrt_adapter").glob("*.py"))
        result["source_sha256"] = {str(f.relative_to(repo)): sha(f) for f in source_paths}
        result["runtime_archive_sha256"] = hashlib.sha256(archive).hexdigest()
        result["model_sha256"] = MODEL_SHA
        result["code_commit"] = command(["git", "-C", str(repo), "rev-parse", "HEAD"]).strip()
        result["code_dirty"] = bool(command(["git", "-C", str(repo), "status", "--porcelain"]).strip())
        # Preserve image PYTHONPATH. The probe adds only the explicit runtime path
        # to sys.path; no mounted FlagGems or shmem stub is introduced.
        container_run = "/case-output/" + run.name
        # A single mapped device is enumerated as logical 0 inside the container.
        common = ["docker", "exec", "-e", "ASCEND_RT_VISIBLE_DEVICES=0",
                  "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HF_HUB_OFFLINE=1",
                  "-e", "TRANSFORMERS_OFFLINE=1", "-e", f"FLAGRT_VERIFIED_GEMS_COMMIT={GEMS}",
                  "-e", f"FLAGRT_VERIFIED_IMAGE_ID={info['Image']}",
                  args.container, "timeout", "--signal=TERM", "--kill-after=10s", "900s", PYTHON]
        preflight = (
            "import torch, triton, importlib.metadata as m, flag_gems; "
            "from pathlib import Path; "
            "assert torch.__version__ == '2.10.0+cpu'; "
            "assert m.version('torch-npu') == '2.10.0'; "
            "assert triton.__version__ == '3.5.1'; "
            "assert m.version('flagtree') == '0.6.0+ascend.git15ec1a6c'; "
            "assert 'site-packages' in Path(flag_gems.__file__).parts; "
            "print(torch.__version__, triton.__version__, flag_gems.__file__)"
        )
        with (run / "imports.log").open("w") as log:
            r = subprocess.run(common + ["-c", preflight], stdout=log,
                               stderr=subprocess.STDOUT, timeout=930)
        if r.returncode:
            raise RuntimeError("v3 package identity/import check failed")
        cases = [("fallback", ["--fallback-only"]),
                 ("operators", ["--operator-only"]), ("model-three-steps", ["--steps", "3"])]
        admission_args = []
        if args.admission_records:
            relative = args.admission_records.resolve().relative_to(repo)
            admission_args = ["--admission-records", "/case-workspace/" + str(relative)]
        for name, extra in cases:
            # Independent cold caches prevent prior workarounds masking defects.
            cache = container_run + "/cache-" + name
            cmd = common[:2] + ["-e", "TRITON_CACHE_DIR=" + cache] + common[2:] + [
                "/case-workspace/" + PROBES + "/test_silu_training_npu.py",
                "--scoped-adapter", "--chip-name", "Ascend910_9382",
                "--runtime-root", container_run + "/runtime-source/dev/device-context/prototype",
                "--model", "/case-model", "--out", container_run + "/" + name + ".json"] + extra + admission_args
            with (run / (name + ".log")).open("w") as log:
                r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=930)
            path = run / (name + ".json")
            data = json.loads(path.read_text()) if path.exists() else {}
            passed = r.returncode == 0 and data.get("status") == "pass"
            blocked = r.returncode != 0 and data.get("status") == "blocked"
            result["results"].append({"case": name, "exit_code": r.returncode,
                                      "status": "PASS" if passed else "BLOCKED" if blocked else "FAIL"})
            if not passed:
                result["overall"] = "BLOCKED" if blocked else "FAIL" if path.exists() else "ERROR"
                result["not_run"] = [case_name for case_name, _ in cases[len(result["results"]):]]
                break
        else:
            result["overall"] = "PASS"
        if any(sha(repo / path) != digest for path, digest in result["source_sha256"].items()):
            raise RuntimeError("Host source changed during execution; invalidate result")
    except (Exception, KeyboardInterrupt) as exc:
        result["overall"] = "ERROR"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (run / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"results: {run}\noverall: {result['overall']}")
    return {"PASS": 0, "FAIL": 1, "ERROR": 2, "BLOCKED": 3}[result["overall"]]


if __name__ == "__main__":
    raise SystemExit(main())
