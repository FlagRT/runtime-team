#!/usr/bin/env bash
# =============================================================================
# 新芯片接入 · 第 0 步环境普查（芯片无关）
# -----------------------------------------------------------------------------
# 用途：把《新芯片接入手册》§1「环境风险前置 7 项」做成一条命令，在目标机上
#       一次跑完并落盘一份环境报告，避免接入当天才逐项问。
#
# 用法（在目标机容器内或宿主机上执行；带卡环境建议在容器内跑 torch 部分）：
#     bash preflight_env.sh                       # 自动探测
#     bash preflight_env.sh --vendor cambricon    # 指定厂商（只影响命名空间候选顺序）
#     OUT=/srv/hliu553/preflight bash preflight_env.sh
#
# 输出：$OUT/preflight_env_<date>.log  —— 直接可作为「环境报告」的证据附件
#
# 设计约定（来自 P800 的实测教训，见接入手册）
#   · 只读：本脚本不装任何东西、不改任何配置、不下载任何镜像
#   · 缺项如实打印「未取得 / 不可用」，不猜测、不补零
#   · 磁盘一节必须看 docker 数据目录的**真实挂载点**（P800 曾误判根分区）
#   · 容器内通常没有 npu-smi / xpu-smi（宿主工具）⇒ 降级用 torch 侧查询
# =============================================================================
set -u

OUT=${OUT:-/tmp/dc_preflight}
VENDOR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --vendor) VENDOR="${2:-}"; shift 2 ;;
    *) echo "未知参数：$1"; exit 2 ;;
  esac
done

mkdir -p "$OUT" 2>/dev/null || OUT=/tmp
LOG="$OUT/preflight_env_$(date +%Y%m%d).log"
exec > >(tee "$LOG") 2>&1

hr() { printf '%s\n' "------------------------------------------------------------------------"; }
sec() { echo; hr; echo "## $*"; hr; }
have() { command -v "$1" >/dev/null 2>&1; }

echo "=========== 新芯片接入 · 环境普查 $(date '+%F %T') ==========="
echo "host=$(hostname 2>/dev/null)  user=$(whoami 2>/dev/null)  vendor_arg=${VENDOR:-（未指定）}"
echo "输出：$LOG"

# ---------------------------------------------------------------- 1 基础环境
sec "1. 基础环境（OS / 内核 / python / 容器）"
echo "-- uname:"; uname -a 2>&1 | head -2
if [ -r /etc/os-release ]; then echo "-- os-release:"; grep -E '^(NAME|VERSION)=' /etc/os-release; fi
echo "-- CPU / 内存:"
echo "   核数: $(nproc 2>/dev/null || echo 未取得)"
if [ -r /proc/meminfo ]; then
  awk '/^MemTotal/{printf "   内存: %.1f GiB\n", $2/1048576}' /proc/meminfo
fi
echo "-- python（多解释器并列）："
for p in python3 python python3.8 python3.9 python3.10 python3.11 python3.12; do
  if have "$p"; then printf '   %-12s %s  (%s)\n' "$p" "$($p -V 2>&1)" "$(command -v $p)"; fi
done
echo "-- conda / venv 环境:"
have conda && { echo "   conda: $(conda --version 2>&1)"; conda env list 2>/dev/null | sed 's/^/     /'; } || echo "   未安装 conda"
find / -maxdepth 4 -name pyvenv.cfg 2>/dev/null | head -5 | sed 's/^/   venv: /'
echo "-- 容器可见性:"
if have docker; then
  echo "   docker: $(docker --version 2>&1)"
  docker ps --format '   running: {{.Names}} | {{.Image}}' 2>&1 | head -8
  echo "   Up 容器数: $(docker ps -q 2>/dev/null | wc -l)"
  echo "   本用户是否在 docker 组: $(id -nG 2>/dev/null | tr ' ' '\n' | grep -qx docker && echo 是 || echo '否（需 sudo 或加组）')"
else
  echo "   docker 不可用"
fi
echo "-- sudo: $(sudo -n true 2>/dev/null && echo 免密可用 || echo '需密码 / 不可用')"

# ---------------------------------------------------------------- 2 芯片与驱动
sec "2. 芯片与驱动（厂商工具 + torch 侧双路取证）"
echo "-- 厂商工具（容器内通常没有，属正常）："
for t in cnmon cnmoninfo mlu-smi npu-smi xpu-smi nvidia-smi; do
  if have "$t"; then echo "   ✅ $t -> $(command -v $t)"; "$t" 2>/dev/null | head -12 | sed 's/^/       /'; fi
done
echo "-- 设备节点 / 驱动模块:"
# 各厂商节点命名不同（寒武纪：/dev/cambricon_dev{N} / cambricon_ctl / cambricon_gdr / cambricon_ipcm{N}）
ls -l /dev/*mlu* /dev/*cambricon* /dev/*davinci* /dev/*npu* /dev/xpu* 2>/dev/null | head -12 || echo "   （未发现常见设备节点）"
ls /dev/cambricon 2>/dev/null | head -8 | sed 's/^/   /dev/cambricon\/: /'
lsmod 2>/dev/null | grep -iE 'mlu|cambricon|cnrt|davinci|drv|xpu|nvidia' | head -8 || echo "   （lsmod 无匹配 / 不可用）"
echo "-- 驱动 / SDK 版本（按厂商候选路径）:"
for d in /usr/local/neuware /usr/local/cambricon /usr/local/Ascend /opt/neuware; do
  [ -d "$d" ] && { echo "   ✔ $d"; ls "$d" 2>/dev/null | head -8 | sed 's/^/       /'; cat "$d/version.txt" 2>/dev/null | head -3 | sed 's/^/       version: /'; }
done

echo "-- torch 侧（同一段代码依次尝试各命名空间，如实报告哪个可用）:"
PY=$(command -v python3 || command -v python)
if [ -n "$PY" ]; then
  "$PY" - <<'PY' 2>&1 | sed 's/^/   /'
import warnings; warnings.filterwarnings("ignore")
import importlib
print("torch:", end=" ")
try:
    import torch
    print(torch.__version__, "| file:", getattr(torch, "__file__", "?"))
except Exception as e:
    print("导入失败:", type(e).__name__, e); raise SystemExit

# 厂商插件候选（按《新芯片接入手册》§2 的四条路径）
for mod in ("torch_mlu", "torch_npu", "torch_xla", "torch_fl"):
    try:
        m = importlib.import_module(mod); print(f"插件 {mod}: 可导入 -> {getattr(m,'__file__','?')}")
    except Exception as e:
        print(f"插件 {mod}: 不可导入（{type(e).__name__}）")

# 命名空间候选
print("命名空间探测（device_count / is_available）:")
for ns in ("mlu", "cuda", "npu", "xpu"):
    api = getattr(torch, ns, None)
    if api is None:
        print(f"  torch.{ns:<5} 不存在"); continue
    try:
        n = api.device_count()
        try: av = api.is_available()
        except Exception: av = "?"
        names = []
        for i in range(min(n, 8)):
            try: names.append(api.get_device_name(i))
            except Exception: names.append("?")
        print(f"  torch.{ns:<5} available={av} device_count={n}")
        for i, nm in enumerate(names): print(f"        [{i}] {nm}")
    except Exception as e:
        print(f"  torch.{ns:<5} 调用异常: {type(e).__name__}: {str(e)[:90]}")
print("注意：device_type 只是命名空间，不能作为厂商标识（见接口约定修订建议第 1 条）")
PY
else
  echo "   未找到 python 解释器"
fi

# ---------------------------------------------------------------- 3 集合通信
sec "3. 集合通信可用性（训练腿前置）"
"${PY:-python3}" - <<'PY' 2>&1 | sed 's/^/   /'
import warnings; warnings.filterwarnings("ignore")
# 只做「可导入性」检查；真正可用性需 2 卡实跑（用 conformance / 训练腿验证）
for mod in ("flagcx", "torch.distributed", "torch_mlu", "torch_npu"):
    try:
        __import__(mod); print(f"{mod}: 可导入")
    except Exception as e:
        print(f"{mod}: 不可导入（{type(e).__name__}）")
PY
echo "-- 通信库文件（按厂商候选）:"
find / -maxdepth 5 \( -name 'libcncl*' -o -name 'libhccl*' -o -name 'libnccl*' -o -name 'libflagcx*' -o -name 'libkccl*' \) 2>/dev/null | head -8 | sed 's/^/   /' || true

# ---------------------------------------------------------------- 4 磁盘
sec "4. 磁盘与 docker 数据目录（★ P800 曾误判根分区，必须看真实挂载点）"
echo "-- 文件系统总览:"; df -h 2>/dev/null | head -12 | sed 's/^/   /'
echo "-- docker 数据目录的真实挂载点:"
if have docker; then
  ROOT=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)
  echo "   DockerRootDir = ${ROOT:-未取得（通常是当前用户不在 docker 组）}"
  # 兜底：无 docker 权限时，仍可用 readlink 判出 data-root 落在哪块盘
  # （本机实测：/var/lib/docker 是 → /srv/var/lib/docker 的符号链接 ⇒ 镜像本就在数据盘）
  LINK=$(readlink -f /var/lib/docker 2>/dev/null)
  [ -n "${LINK:-}" ] && echo "   readlink -f /var/lib/docker = $LINK$([ "$LINK" != "/var/lib/docker" ] && echo '  ← 是符号链接，data-root 已落在该路径所在磁盘')"
  [ -n "${ROOT:-}" ] && { echo "   findmnt -T $ROOT:"; findmnt -T "$ROOT" 2>/dev/null | sed 's/^/     /' || echo "     （findmnt 不可用）"; }
fi
echo "-- 逐目录可用空间（P800 教训：du -x 跨文件系统即停，不可用于判断容量）:"
for d in / /var/lib/docker /srv /data /data1 /data2 /mnt /home; do
  [ -d "$d" ] && printf '   %-18s %s\n' "$d" "$(df -h "$d" 2>/dev/null | awk 'NR==2{print $4" 可用 / "$2" 总 ("$5" 已用)"}')"
done
echo "-- 现有镜像（体积口径：docker images 显示磁盘占用，inspect Size 是镜像层）:"
have docker && docker images --format '   {{.Repository}}:{{.Tag}} | images={{.Size}}' 2>/dev/null | head -15
echo "-- 共享目录判定（本机是否有 NFS/共享挂载）:"
findmnt -t nfs,nfs4,cifs,glusterfs,ceph,fuse.glusterfs 2>/dev/null | sed 's/^/   /' || echo "   （未发现网络文件系统挂载）"

# ---------------------------------------------------------------- 5 网络
sec "5. 网络可达性（镜像 / whl 源；决定能否在线补依赖）"
for u in https://resource.flagos.net/repository/flagos-pypi-hosted/simple https://pypi.org/simple https://mirrors.aliyun.com/pypi/simple; do
  code=$(curl -s -o /dev/null -m 12 -w '%{http_code}' "$u" 2>/dev/null)
  printf '   %-72s HTTP %s\n' "$u" "${code:-超时/失败}"
done
echo "-- 代理环境变量: env | grep -i proxy"
env 2>/dev/null | grep -iE 'proxy' | sed 's/^/   /' || echo "   （无）"
echo "-- 网卡与 RoCE/IB 设备（训练腿拓扑参考）:"
ip -o -4 addr show 2>/dev/null | awk '{print "   "$2" "$4}' | head -8
ls /sys/class/infiniband/ 2>/dev/null | sed 's/^/   IB/RoCE: /' || echo "   （无 /sys/class/infiniband）"

# ---------------------------------------------------------------- 6 拓扑
sec "6. 拓扑（多卡实验的用卡选择依据）"
echo "-- NUMA 与设备亲和（有 lspci 时按 PCI 归属推断；无则如实标注）:"
have lspci && lspci -nn 2>/dev/null | grep -iE 'co-processor|processing accelerator|display|3d controller|network' | head -12 | sed 's/^/   /' || echo "   （无 lspci）"
echo "-- NUMA 节点:"
have numactl && numactl -H 2>/dev/null | grep -E 'available|node [0-9]+ cpus' | head -6 | sed 's/^/   /' || ls /sys/devices/system/node/node*/ 2>/dev/null | sed -n 's#.*/node\([0-9]*\)/#   NUMA \1\n#p' | head -4
echo "-- 共享机提醒：用卡前务必确认目标卡空闲（厂商工具或 torch 查询），并记录用卡号"

# ---------------------------------------------------------------- 7 汇总
sec "7. 结论与待补（如实标注，不猜测）"
cat <<'EOF'
   请逐项核对并填入「已取得 / 待确认」：
     [ ] 1 连接：SSH 端口、认证方式（密钥/密码）、是否需跳板
     [ ] 2 权限：docker 组、（如需改 docker data-root 则需）sudo
     [ ] 3 镜像：目标镜像是否已在位；不在位则 pull 来源与体积
     [ ] 4 共享用户：是否共享机、并发上限（P800/910C 均实测有上限）
     [ ] 5 磁盘：数据目录与 docker 数据目录各自的可用空间
     [ ] 6 网络：whl / 镜像源可达性（决定能否在线补依赖）
     [ ] 7 拓扑：NUMA 归属与组内/跨组链路，2 卡实验的取卡建议
   缺失项照实写「未取得」——不猜、不补零（本方向证据规范）。
EOF
echo
echo "=========== 普查结束 $(date '+%F %T')  报告：$LOG ==========="
