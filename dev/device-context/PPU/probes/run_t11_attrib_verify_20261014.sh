#!/bin/bash
# PPU · T11 归属复核（**完全不含本层**的纯净复现 + 缺项对照）—— 在宿主（ai-server）上执行
# 用法：VARIANTS="pure_baseline pure_wref_30_dropped" ROUNDS=2 bash -s < 本脚本
# 路径纪律：HO 由 O 按挂载映射推出（/bmcp_lvm_fs/hliu553 ↔ /workspace），不得手写
set -u

C=${C:-hliu553-dc-dev}
O=/workspace/probes_ppu/dc_t11_attrib_20261014
HO=/bmcp_lvm_fs/hliu553/probes_ppu/dc_t11_attrib_20261014
DEV=${DEV:-1}
PY=${PY:-/usr/local/bin/python3}
ROUNDS=${ROUNDS:-2}
VARIANTS=${VARIANTS:-"pure_baseline pure_stream_30_dropped pure_wref_30_dropped"}
M=$O/t11_attrib_verify_20261014.py
HM=$HO/t11_attrib_verify_20261014.py

mkdir -p "$HO/log"
if [ ! -f "$HM" ]; then
    echo "!! 复核脚本未就位：$HM（请先 scp 上去）"; exit 2
fi

echo "===== [0] 时间戳 ====="
date "+[%F %T]"

echo "===== [1] 前置：容器在跑 / 残留进程 0 / 用卡读数 ====="
docker ps --format '{{.Names}} {{.Status}}' | grep -E "^$C " || echo "  !! 容器 $C 不在运行"
docker exec "$C" bash -lc 'ps -eo pid,cmd | grep -E "[v]llm|[s]erve_standard" | wc -l' | sed 's/^/  业务残留进程数=/'
docker exec "$C" bash -lc "CUDA_VISIBLE_DEVICES=$DEV $PY -c \"import torch;f,t=torch.cuda.mem_get_info(0);print('  dev free=%.1f GiB / total %.1f'%(f/2**30,t/2**30))\""

echo "===== [2] 自证「跑的是本层当前版本」（同一容器、稍后用于对照读数） ====="
docker exec "$C" bash -lc "ls -l --time-style=+%F_%H:%M /workspace/runtime-team/dev/device-context/prototype/runtime/backends/base.py" | sed 's/^/  /'

echo "===== [3] 矩阵：每轮内跑遍所有变体 ⇒ 轮次交替（分不清变量效应与时间漂移的那条纪律）====="
# 按 run-id 分目录 + rc_matrix.log 只追加（同族教训：扁平目录 + 清空会把上一批原始日志覆盖掉）
RUNID=${RUNID:-$(date +%Y%m%d_%H%M%S)}
LD=$HO/log/$RUNID
mkdir -p "$LD"
echo "===== run=$RUNID  variants=[$VARIANTS]  rounds=$ROUNDS  开始 $(date '+%F %T') =====" >> "$HO/rc_matrix.log"
for r in $(seq 1 "$ROUNDS"); do
    for V in $VARIANTS; do
        # ⚠️ 刻意**不传** DC_ROOT / DC_BACKEND / PYTHONPATH：本组要求进程内没有本层
        docker exec -e CUDA_VISIBLE_DEVICES=$DEV -e PYTHONDONTWRITEBYTECODE=1 \
            -e PYTHONFAULTHANDLER=1 \
            "$C" bash -lc "cd $O && $PY -u $M $V" > "$LD/${V}_r${r}.log" 2>&1
        rc=$?
        echo "$rc" > "$LD/${V}_r${r}.rc"
        printf "run=%s round=%s variant=%-32s rc=%s\n" "$RUNID" "$r" "$V" "$rc" | tee -a "$HO/rc_matrix.log"
    done
done

echo
echo "===== [4] 汇总（rc 分布 + 自证标记）====="
HOSTPY=$(command -v python3 || command -v python || echo "")
if [ -n "$HOSTPY" ]; then
$HOSTPY - <<PYEOF 2>/dev/null || true
import collections, pathlib
hd = pathlib.Path("$HO")
rows = collections.defaultdict(list)
marks = collections.defaultdict(set)
for p in sorted(hd.glob("log/**/*.log", recursive=True)):
    variant, rnd = p.stem.rsplit("_r", 1)
    rc = p.with_suffix(".rc").read_text().strip() if p.with_suffix(".rc").exists() else "?"
    txt = p.read_text(errors="ignore")
    for ln in txt.splitlines():
        if ln.startswith("BRANCH="):
            marks[variant].add("BRANCH")
        if ln.startswith("EXIT_REACHED"):
            marks[variant].add("EXIT_REACHED")
        if ln.startswith("runtime_loaded="):
            marks[variant].add(ln.strip().replace("runtime_loaded=", "rt="))
        if "Fatal Python error: Segmentation fault" in txt:
            marks[variant].add("SEGV")
    rows[variant].append((rnd, rc))
print("%-34s %-12s %-10s %s" % ("变体", "rc 序列", "非0/总数", "自证"))
for variant in sorted(rows):
    rs = sorted(rows[variant], key=lambda x: int(x[0]))
    seq = " ".join(rc for _, rc in rs)
    bad = sum(1 for _, rc in rs if rc != "0")
    print("%-34s %-12s %-10s %s" % (variant, seq, "%d/%d" % (bad, len(rs)), "/".join(sorted(marks[variant]))))
PYEOF
fi
