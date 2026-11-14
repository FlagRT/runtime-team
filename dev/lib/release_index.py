#!/usr/bin/env python3
"""机读锚点（锚点 2）生成器：把各现行环境的 image.baseline.yaml 聚合为
dev/images/RELEASE_INDEX.yaml（schema: flagrt.release-index/v1）。

七环境声明表（ENVIRONMENTS，本文件内唯一的数据声明）只登记「环境键 → 哪份
baseline / 哪个运行环境目录 / 什么形态 / 哪份锁定声明」；逐环境的版本事实
（image.ref / id / digest）一律抄录自对应 image.baseline.yaml，绝不在本表重复。

导航入口见 dev/images/README.md §1；互锁校验（四查）见 dev/lib/check_release.py；
环境基线机制规范见 dev/ENV-SPEC.md。

用法：
  release_index.py export        # 生成 dev/images/RELEASE_INDEX.yaml（生成物，禁手改）

退出码：0 成功；1 事实或文件缺失；2 用法错误。
"""
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "dev/images/RELEASE_INDEX.yaml"

SCHEMA = "flagrt.release-index/v1"
GENERATED_BY = "dev/lib/release_index.py export"

# 七环境声明表：现行环境 → 事实来源与形态。
#   baseline    = image.baseline.yaml 路径（版本事实的唯一来源，逐字抄录）
#   env_dir     = 现行运行环境目录（env/vN 为 up-sh 形态；demo/ 为该镜像验收冻结记录
#                 兼现行形态；None = 登记式、无仓内运行环境目录）
#   env_form    = up-sh | demo | registered-only
#   stack_lock  = 该环境所在的现行锁定声明
ENVIRONMENTS = {
    "ascend910c-train": {
        "baseline": "dev/images/ascend-train-comm/v3/image.baseline.yaml",
        "env_dir": "dev/images/ascend-train-comm/v3/env/v1",
        "env_form": "up-sh",
        "stack_lock": "dev/stack.lock.910c.v4.yaml",
    },
    "ascend910c-infer": {
        "baseline": "dev/images/ascend-infer-vllm/v1/image.baseline.yaml",
        "env_dir": "dev/images/ascend-infer-vllm/v1/env/v1",
        "env_form": "up-sh",
        "stack_lock": "dev/stack.lock.910c.v4.yaml",
    },
    "kunlunxin-train": {
        "baseline": "dev/images/kunlun-operator-runtime/v2/image.baseline.yaml",
        "env_dir": "dev/images/kunlun-operator-runtime/v2/env/v1",
        "env_form": "up-sh",
        "stack_lock": "dev/stack.lock.kunlunxin.v1.yaml",
    },
    "kunlunxin-infer": {
        "baseline": "dev/images/kunlun-infer-vllm/v1/image.baseline.yaml",
        "env_dir": None,
        "env_form": "registered-only",
        "stack_lock": "dev/stack.lock.kunlunxin.v1.yaml",
    },
    "cambricon-train": {
        "baseline": "dev/images/cambricon-mlu-runtime/v1/image.baseline.yaml",
        "env_dir": "dev/images/cambricon-mlu-runtime/v1/demo",
        "env_form": "demo",
        "stack_lock": "dev/stack.lock.cambricon.v1.yaml",
    },
    "cambricon-infer": {
        "baseline": "dev/images/cambricon-infer-vllm/v1/image.baseline.yaml",
        "env_dir": "dev/images/cambricon-infer-vllm/v1/demo",
        "env_form": "demo",
        "stack_lock": "dev/stack.lock.cambricon.v1.yaml",
    },
    "thead-ppu": {
        "baseline": "dev/images/thead-ppu-runtime/v2/image.baseline.yaml",
        "env_dir": "dev/images/thead-ppu-runtime/v2/env/v1",
        "env_form": "up-sh",
        "stack_lock": "dev/stack.lock.thead-ppu.v1.yaml",
    },
}


def load_baseline(rel_path: str) -> dict:
    path = ROOT / rel_path
    if not path.is_file():
        raise SystemExit(f"release_index: baseline 不存在：{rel_path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def export() -> int:
    environments = {}
    for key, decl in ENVIRONMENTS.items():
        data = load_baseline(decl["baseline"])
        image = (data or {}).get("image") or {}
        if not image.get("ref") or not image.get("id"):
            raise SystemExit(f"release_index: {key} 的 baseline 缺 image.ref/id：{decl['baseline']}")
        entry_image = {
            "ref": image["ref"],
            "id": image["id"],
            "baseline": decl["baseline"],
        }
        if image.get("digest"):
            entry_image["digest"] = image["digest"]

        if decl["env_dir"] is None:
            env_current = None
        else:
            env_dir = ROOT / decl["env_dir"]
            env_baseline = Path(decl["env_dir"]) / "env.baseline.yaml"
            if not env_dir.is_dir():
                raise SystemExit(f"release_index: {key} 的运行环境目录不存在：{decl['env_dir']}")
            if not (ROOT / env_baseline).is_file():
                raise SystemExit(f"release_index: {key} 缺 env.baseline.yaml：{env_baseline}")
            env_current = {"dir": decl["env_dir"], "baseline": str(env_baseline)}

        environments[key] = {
            "image": entry_image,
            "env_current": env_current,
            "env_form": decl["env_form"],
            "stack_lock": decl["stack_lock"],
        }

    header = (
        "# ⚠️ 生成物，禁手改：由 dev/lib/release_index.py export 聚合各 image.baseline.yaml 生成；\n"
        "# 互锁校验：python3 dev/lib/check_release.py；导航入口：dev/images/README.md §1。\n"
    )
    body = {
        "schema": SCHEMA,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environments": environments,
    }
    text = header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True,
                                   default_flow_style=False, width=4096)
    INDEX_PATH.write_text(text, encoding="utf-8")
    print(f"release_index: 已生成 {INDEX_PATH.relative_to(ROOT)}（{len(environments)} 环境）")
    return 0


def main(argv) -> int:
    if argv[1:] == ["export"]:
        return export()
    print("用法：release_index.py export", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
