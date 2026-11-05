"""`DC_MODEL` 的取值与解析 —— **两条腿共用的唯一实现**。

════ 为什么单独成一个文件（2026-10-27）════
  原来只有 `proto_infer_leg.py` 里有一份 `resolve_model()`，`proto_train_leg.py` 没有
  ⇒ 同一台机器、同一份 `DC_MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B`：

     推理腿 ✅ 解析到 snapshots/<hash> 后正常加载
     训练腿 ❌ `ValueError: Unrecognized model in /hf_cache/hub/models--Qwen--…`

  这是同一个环境变量、两条腿理解不一致（**声明即承诺**那一族：变量名相同，
  但「缓存根也算合法传法」这件事只在一半实现里落地）。10-27 MLU590 复跑实测暴露时，
  为不把共享层改动混进回归记录，当时的做法是在容器里先手工解析 snapshot 绕过。

  处置**不是**把 `resolve_model` 复制一份到训练腿（那样两份实现迟早漂移），
  而是抽到本模块、两条腿 import 同一个函数；
  `scripts/model_path_parity_check.py` 做静态守卫 —— 任何一条腿再自己写一份
  `def resolve_model(` 即判 FAIL。

════ 接受的 `DC_MODEL` 传法 ════
  1. **snapshot 目录**（自身含 `config.json`）          → 原样使用
  2. **hub 缓存目录** `models--<org>--<name>/`         → 解析到其中一个 snapshot
       · 优先 `refs/main`（HF 自己的口径：该文件内容就是 commit hash）
       · 无 `refs/main`，或它指向的 hash 不在 `snapshots/` 下 ⇒ 取排序最后一个，
         **并打印告警**（多个 snapshot 时「挑哪个」必须可见，不允许静默任选）
  3. **任意普通本地目录**（`from_pretrained` 认的那种）  → 原样使用

  ⛔ **不支持**直接给 `/hf_cache/hub` 这类「所有模型的父目录」
    —— 它既无 `config.json` 也无 `snapshots/`，会走到 `from_pretrained` 报
       `ValueError: Unrecognized model in /hf_cache/hub`；本模块刻意不猜「你想用哪个模型」。
     （父目录下只有一个 `models--*` 时也不猜：猜对了没人知道，猜错了更难查。）

════ 不传 `DC_MODEL` 时 ════
  按 `INSTANCE_LAYOUTS` 依次探测（顺序＝现有实例数序），命中即用并**打印来源**。
  ⚠️ **不允许出现「某一家路径当全局默认」**：2026-10-26 PPU 全量重跑抓到过
     —— 原默认值是 910C 专有的 `/mnt/raid/hliu553/models/…`，在 PPU / MLU590 上直接抛
       `HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`，
     看起来像「模型 id 格式错误」，与真实原因无关。
"""

from __future__ import annotations

import os
import sys

#: 各实例已实测的模型布局（顺序＝探测顺序；命中即用并打印来源）
INSTANCE_LAYOUTS = (
    "/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B",   # 910C（宿主与容器同路径）
    "/workspace/models/Qwen3-Embedding-0.6B",          # P800 / PPU（宿主 /data2|/bmcp_lvm_fs ↔ /workspace）
    "/work/models/Qwen3-Embedding-0.6B",               # MLU590（宿主 /srv/hliu553 ↔ /work）
    # P800 / MLU590 的 HF 缓存形态（给缓存根会自动解析到 snapshots/<hash>，见 resolve_model）
    "/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B",
)


def resolve_model(path: str) -> str:
    """把 `DC_MODEL` 的任一种合法传法，解析成 `from_pretrained` 可直接用的目录。

    只做路径形态判断，**不校验内容完整性**（权重是否损坏由两腿各自的「步 0 模型预检」负责，
    见 `910C/probes/run_910c_rerun_20261027.sh` 的 `[0.5]` 段与手册 §9 坑 16）。
    """
    if not path or not os.path.isdir(path):
        return path

    snaps = os.path.join(path, "snapshots")
    if not os.path.isdir(snaps):
        return path                                   # 形态 1 / 3：自身就是模型目录

    hashes = sorted(d for d in os.listdir(snaps)
                    if os.path.isdir(os.path.join(snaps, d)))
    if not hashes:
        return path                                   # 空 snapshots/：交给 from_pretrained 报错

    ref = os.path.join(path, "refs", "main")
    commit = ""
    if os.path.isfile(ref):
        try:
            with open(ref, encoding="utf-8") as fh:
                commit = fh.read().strip()
        except OSError as exc:                        # 读不到就当没有 refs，不静默吞掉
            print(f"[model][WARN] 读 {ref} 失败（{exc.__class__.__name__}: {exc}）；"
                  f"按排序回退", flush=True)
    if commit:
        if os.path.isdir(os.path.join(snaps, commit)):
            return os.path.join(snaps, commit)        # HF 自己的口径：refs/main ⇒ 确定的 commit
        print(f"[model][WARN] {ref} 指向 {commit!r}，但 snapshots/ 下没有该目录；"
              f"回退到排序最后 {hashes[-1]}", flush=True)
    elif len(hashes) > 1:
        print(f"[model][WARN] {path} 下有 {len(hashes)} 个 snapshot 且无 refs/main ⇒ "
              f"取排序最后 {hashes[-1]}；要固定版本请显式给 DC_MODEL=<snapshot 目录>", flush=True)
    return os.path.join(snaps, hashes[-1])


def pick_model() -> tuple[str, str]:
    """取 `DC_MODEL` 原始值；未给则按实例布局探测。返回 (原始值, 来源说明)。"""
    explicit = os.environ.get("DC_MODEL")
    if explicit:
        return explicit, "DC_MODEL"
    for cand in INSTANCE_LAYOUTS:
        if os.path.isdir(cand):
            return cand, "按实例布局探测"
    return "", "**未找到**（须显式给 DC_MODEL）"


def require_model(raw: str, source: str) -> str:
    """解析 + 早失败。取不到就打印**怎么给**并 `exit 4`，不让厂商/transformers 的报错冒充原因。

    原实现在这里会抛 `HFValidationError: Repo id must be in the form …`，
    长得像「模型 id 格式错误」，与真实原因（路径不存在）无关。
    """
    model = resolve_model(raw)
    if raw and os.path.isdir(model):
        return model
    print(f"❌ 取不到模型路径：DC_MODEL={os.environ.get('DC_MODEL', '<未设>')} "
          f"（来源={source}，解析后={model or '<空>'}）", flush=True)
    print("   已探测的实例布局：" + " · ".join(INSTANCE_LAYOUTS), flush=True)
    print("   ⇒ 显式给 `DC_MODEL=<模型目录>`："
          "普通目录 / snapshot 目录 / HF 的 `models--<org>--<name>` 缓存根**三者都可以**；"
          "⛔ 但不要给 `/hf_cache/hub` 这种所有模型的父目录", flush=True)
    sys.exit(4)
