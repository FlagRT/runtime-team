#!/bin/bash
# thead-ppu-runtime v2 构建入口（单入口，ENV-SPEC 四硬规则之一）
#
# 用法（在**有 docker 的机器**上，任意目录）：
#   bash dev/images/thead-ppu-runtime/v2/build.sh
#   REF=...        bash build.sh      # 覆盖镜像 ref
#   MIN_FREE_GB=3  bash build.sh      # 覆盖磁盘水位守卫阈值
#   WHEEL_DIR=...  bash build.sh      # 覆盖 wheel 所在目录
#
# ⚠️ 磁盘约束（2026-10-27 实测）：共享机根盘 `/var/lib/docker` 只剩 ~3.1G，
#    派生层要落 ~1.1G（解包后的 triton）。故本脚本：
#      ① 构建**前**做水位守卫，不足直接 ABORT（绝不把共享机写爆）；
#      ② 构建上下文在**数据盘**组装（`/bmcp_lvm_fs/...`），只把 358MB 的 wheel 复制过去，
#         **不**在 root 盘做 context（那会让峰值翻倍）；
#      ③ 构建后打印余量，便于判断是否要回收旧容器/镜像。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WHEEL_DIR="${WHEEL_DIR:-/bmcp_lvm_fs/A/data/flagtree-src/dist}"
WHEEL_NAME="flagtree-0.7.0+ppu.gite223f42c-cp312-cp312-linux_x86_64.whl"
REF="${REF:-flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64}"
MIN_FREE_GB="${MIN_FREE_GB:-2.5}"
CTX="${CTX:-/bmcp_lvm_fs/hliu553/image-build/thead-ppu-runtime-v2}"

free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }

echo "===== [0] 磁盘水位守卫 ====="
f0=$(free_gb)
echo "  当前 / 可用：${f0}G（要求 ≥ ${MIN_FREE_GB}G）"
if [ "$f0" -lt "${MIN_FREE_GB}" ]; then
  echo "  ⛔ ABORT：root 盘余量不足，构建会挤占共享机。先回收（旧容器/旧镜像）再重试。" >&2
  exit 3
fi

echo "===== [1] 资产核对（一个 wheel 同时提供 triton + flagtree）====="
[ -f "$WHEEL_DIR/$WHEEL_NAME" ] || { echo "  ⛔ 缺 wheel：$WHEEL_DIR/$WHEEL_NAME" >&2; exit 4; }
printf "  ✅ %s\n     %s  %s\n" "$WHEEL_NAME" "$(du -h "$WHEEL_DIR/$WHEEL_NAME" | cut -f1)" \
       "sha256=$(sha256sum "$WHEEL_DIR/$WHEEL_NAME" | cut -d' ' -f1)"
# 自证：wheel 顶层必须含 triton/ 与 flagtree/（否则装完仍然缺 triton）
python3 - "$WHEEL_DIR/$WHEEL_NAME" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
tops = {n.split("/")[0] for n in z.namelist()}
need = {"triton", "flagtree"}
missing = need - tops
if missing:
    sys.exit(f"  ⛔ wheel 缺顶层 {missing}（实测顶层={sorted(tops)[:6]}）")
print(f"  ✅ wheel 顶层含 {sorted(need)}（条目 {len(z.namelist())}）")
PY

echo "===== [2] 组装构建上下文（数据盘，只复制 358MB wheel）====="
rm -rf "$CTX"; mkdir -p "$CTX/src/flagtree"
# ⚠️ 用**复制**而非硬链接：源 `/bmcp_lvm_fs/A/data/...` 属他人，`fs.protected_hardlinks=1`
#    会以 `Operation not permitted` 拒绝跨属主硬链接（2026-10-27 实测）。
cp -a "$WHEEL_DIR/$WHEEL_NAME" "$CTX/src/flagtree/"
cp "$HERE/Dockerfile.repro" "$CTX/Dockerfile"
echo "  上下文=$CTX"

echo "===== [3] 构建 ====="
docker build -f "$CTX/Dockerfile" -t "$REF" "$CTX"

echo "===== [4] 结果 ====="
docker images --format '{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}' | grep -F "$REF" || true
f1=$(free_gb)
echo "  / 可用：${f0}G → ${f1}G（本层新增约 $((f0 - f1))G）"
echo "  wheel sha256：$(sha256sum "$WHEEL_DIR/$WHEEL_NAME" | cut -d' ' -f1)"
echo "  下一步：按 REBUILD.md「验证判据」用**全新容器**验证 triton 免手工补齐"
