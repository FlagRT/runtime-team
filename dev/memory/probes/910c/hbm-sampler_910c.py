#!/usr/bin/env python3
"""infer910c_hbm_sampler.py —— 910C 外挂 HBM / AICore 采样器（宿主侧，无 torch 依赖）。

用途
  锁定推理镜像画像的**设备级真值来源**：EngineCore 是 spawn 子进程，driver 进程
  读不到 worker 分配器计数（见 archive/V1-显存画像报告-20260817 §3.3），故 HBM 峰值
  必须靠外部按固定间隔轮询 `npu-smi info` 采样 + vLLM 日志交叉验证。

  本脚本**在宿主机运行**（不进容器、不 import torch），与容器内 infer910c_mem_profile.py
  同时起：容器内跑加载/推理，宿主这边采 HBM 曲线，事后按时间戳对齐。

解析
  `npu-smi info` 每颗芯片占**两行物理文本**：
    行1 = | NPU  Name | Health | Power(W)  Temp(C)  Hugepages-Usage |
    行2 = | Chip      | Bus-Id | AICore(%) Memory-Usage(MB)  HBM-Usage(MB) |
  HBM-Usage 是行2 最后一组 `used / total`（MB）；AICore(%) 是行2 数据格首个数字。
  不同 npu-smi 版本列宽/字段略有出入 → 解析只依赖「行2 最后两个数字 = HBM used/total」
  「行2 第一个数字 = AICore%」，不按列位置硬切。

输出
  追加写 CSV（多次运行落同一文件），列：
    timestamp,tag,chip,hbm_used_mb,hbm_total_mb,hbm_util_pct,aicore_pct
  `--tag` 用于把不同 run（如 gpu-mem-util=0.4 / 0.9）区分在同一 CSV 里。

用法（示例）
  # 采 chip 0，每 0.5s 一次，跟随到 Ctrl+C：
  python3 dev/memory/probes/infer910c_hbm_sampler.py --chips 0 --interval 0.5 \
      --out /mnt/raid/xliu969/mem910c/hbm.csv --tag gmu0.9-eager

  # 采 chip 0,1，跑 180s 自动停：
  python3 dev/memory/probes/infer910c_hbm_sampler.py --chips 0,1 --duration 180 \
      --out /mnt/raid/xliu969/mem910c/hbm.csv --tag load-phase

UNTESTED —— pending 910C 锁定镜像验证。npu-smi 输出格式以实机为准，解析回退分支
（见 parse_npu_smi）可能需按实机样例微调。
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime

CSV_FIELDS = [
    "timestamp",
    "tag",
    "chip",
    "hbm_used_mb",
    "hbm_total_mb",
    "hbm_util_pct",
    "aicore_pct",
]

_NUM = re.compile(r"\d+(?:\.\d+)?")
# 行1：| <id>  <Name 非纯数字> ... |  —— 芯片块首行
_CHIP_HEAD = re.compile(r"^\|\s*(\d+)\s+\S+")
# 行2（新版式 npu-smi 25.x / A3，实机 npu1-27 2026-09-10 取样为准）：
#   | <Chip-id 0/1>  <Phy-ID 0..15> | <Bus-Id 0000:93:00.0> | <AICore%> <Mem u/t> <HBM u/t> |
#   首格是「Chip-id + Phy-ID」两个数，Phy-ID 才是 davinci 号（--chips 用它做键）。
_CHIP_BODY_PHY = re.compile(
    r"^\|\s*(\d+)\s+(\d+)\s*\|\s*"
    r"([0-9A-Fa-f]{2,4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d)\s*\|(.*)")
# 行2（旧版式：首格只有一个 id，直接接 Bus-Id）——回退用，按该 id 做键。
# [\s|]+ 兼容「id 与 Bus-Id 之间有/无中缝竖线」。
_CHIP_BODY = re.compile(
    r"^\|\s*(\d+)[\s|]+([0-9A-Fa-f]{2,4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d)")

_STOP = False


def _on_sigint(_signum, _frame):
    global _STOP
    _STOP = True


def run_npu_smi() -> str:
    """调 `npu-smi info`，返回 stdout 文本；失败抛 RuntimeError。"""
    try:
        p = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError as e:
        raise RuntimeError("npu-smi 不在 PATH（本脚本须在宿主机运行）") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("npu-smi info 超时 15s") from e
    if p.returncode != 0:
        raise RuntimeError(f"npu-smi info 退出码 {p.returncode}: {p.stderr.strip()[:200]}")
    return p.stdout


def parse_npu_smi(text: str) -> dict[int, dict[str, float]]:
    """解析 npu-smi info 文本 → {chip_id: {hbm_used_mb, hbm_total_mb, aicore_pct}}。

    键：新版式（A3 / npu-smi 25.x）用 **Phy-ID**（davinci 号 0..15）；旧版式回退用首格 id。
    只依赖：芯片块次行（含 Bus-Id）末尾 `used / total` 为 HBM(MB)，首个数字为 AICore(%)。
    对列宽/字段顺序变化鲁棒：不按固定列位置切分。
    """
    out: dict[int, dict[str, float]] = {}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _CHIP_BODY_PHY.match(line)
        if m:
            chip = int(m.group(2))          # Phy-ID = davinci 号
            after_busid = m.group(4)
        else:
            m = _CHIP_BODY.match(line)
            if not m:
                continue
            chip = int(m.group(1))          # 旧版式：首格单 id
            after_busid = line[m.end():]
        # 从 Bus-Id 之后到行尾找所有数字（不同版本 `|` 数不定，不按固定列位置切分）。
        nums = _NUM.findall(after_busid)
        if len(nums) < 3:
            # 回退：AICore 可能在行1（Power/Temp 那行的某些定制版），或本行被截断
            # → 尝试合并行1 的数字一起看，最后两位仍取作 HBM used/total。
            head = lines[i - 1] if i > 0 else ""
            nums = _NUM.findall(head) + nums
        if len(nums) < 2:
            continue
        hbm_used = float(nums[-2])
        hbm_total = float(nums[-1])
        aicore = float(nums[0]) if nums else 0.0
        # 合理性检查：HBM total 至少几千 MB（910C 单芯 65536 MiB）；否则视为解析错位
        if hbm_total < 1024:
            continue
        out[chip] = {
            "hbm_used_mb": hbm_used,
            "hbm_total_mb": hbm_total,
            "aicore_pct": aicore,
        }
    return out


def select_chips(spec: str, available: list[int]) -> list[int]:
    if spec.strip().lower() in ("all", "*", ""):
        return sorted(available)
    want = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            want.append(int(tok))
    return want


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=1.0, help="采样间隔秒（默认 1.0）")
    ap.add_argument("--out", required=True, help="CSV 输出路径（追加写）")
    ap.add_argument("--chips", default="0", help="芯片号，逗号分隔或 all（默认 0）")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="采样时长秒；0 = 跑到 SIGINT（默认 0）")
    ap.add_argument("--tag", default="", help="写入 CSV tag 列，用于区分多次 run")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    # 首采：确认 npu-smi 可用 + 解析出芯片
    try:
        first = parse_npu_smi(run_npu_smi())
    except RuntimeError as e:
        print(f"[fatal] {e}", file=sys.stderr)
        return 1
    if not first:
        print("[fatal] 未从 npu-smi info 解析出任何芯片，检查输出格式并调整 parse_npu_smi",
              file=sys.stderr)
        return 1
    chips = select_chips(args.chips, list(first.keys()))
    missing = [c for c in chips if c not in first]
    if missing:
        print(f"[warn] 请求的芯片 {missing} 首采未出现（可能空闲未列），仍会尝试采样",
              file=sys.stderr)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    new_file = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    f = open(out_path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if new_file:
        w.writeheader()
        f.flush()

    print(f"[start] out={out_path} chips={chips} interval={args.interval}s "
          f"duration={'∞' if args.duration == 0 else str(args.duration) + 's'} tag={args.tag!r}")

    t0 = time.time()
    n = 0
    try:
        while not _STOP:
            loop_start = time.time()
            ts = datetime.now().isoformat(timespec="milliseconds")
            try:
                parsed = parse_npu_smi(run_npu_smi())
            except RuntimeError as e:
                print(f"[warn] {ts} 采样失败: {e}", file=sys.stderr)
                parsed = {}
            for chip in chips:
                d = parsed.get(chip)
                if d is None:
                    continue
                util = (d["hbm_used_mb"] / d["hbm_total_mb"] * 100.0) if d["hbm_total_mb"] else 0.0
                w.writerow({
                    "timestamp": ts,
                    "tag": args.tag,
                    "chip": chip,
                    "hbm_used_mb": f"{d['hbm_used_mb']:.0f}",
                    "hbm_total_mb": f"{d['hbm_total_mb']:.0f}",
                    "hbm_util_pct": f"{util:.2f}",
                    "aicore_pct": f"{d['aicore_pct']:.1f}",
                })
            f.flush()
            n += 1
            if args.duration and (time.time() - t0) >= args.duration:
                break
            sleep = args.interval - (time.time() - loop_start)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        f.close()
        print(f"[done] 写入 {n} 轮采样 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
