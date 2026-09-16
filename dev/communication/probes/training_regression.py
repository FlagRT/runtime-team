"""Run the team's unchanged training algorithm with private paths and rank binding.

Reference: dev/device-context/prototype/runtime/proto/proto_train_leg.py.
No shared result files are overwritten. Run under torchrun and external timeout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    reference = args.runtime_root / "runtime/proto/proto_train_leg.py"
    source = reference.read_text()
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    replacements = {
        'sys.path.insert(0, "/mnt/raid/hliu553/runtime-team/dev/device-context")':
            f"sys.path.insert(0, {str(args.runtime_root.resolve())!r})",
        'MODEL = "/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B"':
            f"MODEL = {str(args.model.resolve())!r}",
        '    dist.init_process_group("flagos", timeout=timedelta(seconds=180))':
            '    torch.flagos.set_device(local_rank)\n    dist.init_process_group("flagos", timeout=timedelta(seconds=180))',
        'f"/mnt/raid/hliu553/runtime-team/scratch/train_leg_result_rank{rank}.json"':
            'str(Path(os.environ["COMM_TRAIN_OUTPUT"]) / f"train_leg_result_rank{rank}.json")',
        '    res: dict = {"rank": rank, "world_size": world, "model": MODEL, "checks": {}}':
            '    res: dict = {"rank": rank, "world_size": world, "model": MODEL, "checks": {}, "communication_identity": identity()}',
    }
    for original, replacement in replacements.items():
        if source.count(original) != 1:
            raise RuntimeError("Reference changed; review path/binding adapter before running")
        source = source.replace(original, replacement)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["COMM_TRAIN_OUTPUT"] = str(args.out_dir.resolve())
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch_fl  # noqa: F401 -- registration before reference imports torch
    from communication_matrix import identity
    scope = {"__name__": "communication_training_reference", "Path": Path, "identity": identity}
    exec(compile(source, str(reference), "exec"), scope)
    scope["main"]()
    rank = int(os.environ["RANK"])
    output = args.out_dir / f"train_leg_result_rank{rank}.json"
    result = json.loads(output.read_text())
    result.update(reference_sha256=source_hash, phase="group_destroyed",
                  adapter_changes=["private input/output paths", "bind rank before group init", "record loaded backend"],
                  attribution="device-context training algorithm; communication regression run")
    import transformers
    result["transformers_version"] = transformers.__version__
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "TRAIN_LEG_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
