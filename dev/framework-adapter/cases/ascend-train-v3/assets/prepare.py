"""Generate the consumer baseline using the unmodified public baseline tool.

No daemon start, Git checkout, dependency install or image substitution.
Run with Python + PyYAML and Docker Compose v2 available on PATH.
"""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-root", type=Path, required=True,
                        help="checkout/export of runtime-team public baseline 74197bf")
    args = parser.parse_args()
    case = Path(__file__).resolve().parents[1]
    repo = case.parents[3]
    public = args.public_root.resolve()
    image_dir = public / "dev/images/ascend-train-comm/v3"
    for path in (image_dir / "image.baseline.yaml", public / "dev/lib/baseline.py"):
        if not path.is_file():
            parser.error(f"Missing public source: {path}")
    for relative in ("dev/lib/baseline.py", "dev/lib/up.sh",
                     "dev/images/ascend-train-comm/v3/image.baseline.yaml",
                     "dev/images/ascend-train-comm/v3/demo/docker-compose.yml"):
        expected = subprocess.check_output([
            "git", "-C", str(repo), "show",
            "74197bf7d4b5dc8cdd3b2b89b15ac3561936dda6:" + relative])
        if (public / relative).read_bytes() != expected:
            parser.error(f"Public source differs from reviewed version: {relative}")
    link = case / "public-baseline"
    if link.is_symlink():
        if link.resolve() != image_dir:
            parser.error("Existing public-baseline points elsewhere; inspect before changing")
    elif link.exists():
        parser.error("public-baseline is not a symlink; refusing replacement")
    else:
        link.symlink_to(image_dir, target_is_directory=True)
    env = dict(os.environ, FLAGRT_WORKSPACE_ROOT=str(repo), WORKSPACE_ROOT=str(repo))
    # Export honestly records dirty/unpushed local development. It is not an
    # acceptance certificate; --require-clean must be used before publication.
    import sys
    return subprocess.call([sys.executable, str(public / "dev/lib/baseline.py"),
                            "export", str(case / "demo")], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
