#!/bin/bash
# PPU · T11「解释器退出期段错误」最小对照矩阵 —— 在宿主（ai-server）上执行
# 用法：VARIANTS="native_plain ours_wrapper" ROUNDS=3 bash -s < 本脚本
# 路径纪律：HO 由 O 按挂载映射推出（/bmcp_lvm_fs/hliu553 ↔ /workspace），不得手写
set -u

C=${C:-hliu553-dc-dev}
P=/workspace/runtime-team/dev/device-context/prototype
O=/workspace/probes_ppu/dc_t11_20261014
HO=/bmcp_lvm_fs/hliu553/probes_ppu/dc_t11_20261014
DEV=${DEV:-1}
PY=${PY:-/usr/local/bin/python3}
ROUNDS=${ROUNDS:-3}
VARIANTS=${VARIANTS:-"native_plain ours_wrapper"}
M=$O/t11_exit_segv_matrix.py
HM=$HO/t11_exit_segv_matrix.py

mkdir -p "$HO/log"
if [ ! -f "$HM" ]; then
    echo "!! 矩阵脚本未就位：$HM（请先 scp 上去）"; exit 2
fi

echo "===== [0] 时间戳 ====="
date "+[%F %T]"

echo "===== [1] 前置：服务残留计数（应为 0）/ 用卡读数 ====="
docker exec "$C" bash -lc 'ps -eo pid,cmd | grep -E "[v]llm|[s]erve_standard" | wc -l'
docker exec "$C" bash -lc "CUDA_VISIBLE_DEVICES=$DEV $PY -c \"import torch;f,t=torch.cuda.mem_get_info(0);print('dev free=%.1f GiB / total %.1f'%(f/2**30,t/2**30))\""

echo "===== [2] 自证「跑的是当前版本」 ====="
docker exec "$C" bash -lc "grep -c tname $P/runtime/backends/base.py; ls -l --time-style=+%F_%H:%M $P/runtime/backends/ppu/backend.py $P/runtime/__init__.py"

echo "===== [3] 矩阵（每轮内跑遍所有变体 ⇒ 轮次交替，避免时间漂移混入变量效应）====="
# ⚠️ **2026-10-14 修（自审发现）**：原实现用 `: > rc_matrix.log` + 扁平 `log/`，
#    每跑一次新矩阵就把**上一批的原始日志覆盖掉** ⇒ 报告表格里 15 个变体，
#    归档目录里只剩最后一批的 3 个，「读数无法从证据回读」。
#    现改为**按 run-id 分目录 + rc_matrix.log 只追加**，历史批次互不覆盖。
RUNID=${RUNID:-$(date +%Y%m%d_%H%M%S)}
LD=$HO/log/$RUNID
mkdir -p "$LD"
echo "===== run=$RUNID  variants=[$VARIANTS]  rounds=$ROUNDS  开始 $(date '+%F %T') =====" >> "$HO/rc_matrix.log"
for r in $(seq 1 "$ROUNDS"); do
    for V in $VARIANTS; do
        docker exec -e DC_ROOT=$P -e DC_BACKEND=ppu -e CUDA_VISIBLE_DEVICES=$DEV \
            -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONFAULTHANDLER=1 \
            "$C" bash -lc "cd $P && $PY -u $M $V" > "$LD/${V}_r${r}.log" 2>&1
        rc=$?
        echo "$rc" > "$LD/${V}_r${r}.rc"
        printf "run=%s round=%s variant=%-24s rc=%s\n" "$RUNID" "$r" "$V" "$rc" | tee -a "$HO/rc_matrix.log"
    done
done

echo
echo "===== [4] 汇总（rc 分布 + 分支自证标记）====="
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
        if ln.startswith("backend="):
            marks[variant].add(ln.strip())
    rows[variant].append((rnd, rc, txt))
print("变体                     轮次  非0退出/总数   rc 序列（按轮）            自证")
for variant in sorted(rows):
    rs = sorted(rows[variant], key=lambda x: int(x[0]))
    bad = sum(1 for _, rc, _ in rs if rc != "0")
    print("%-24s %4d  %d/%d          %-24s %s" % (
        variant, len(rs), bad, len(rs),
        " ".join(rc for _, rc, _ in rs), "/".join(sorted(marks[variant]))))
PYEOF
else
    echo "（宿主无 python3，汇总请在本地做）"
fi

echo
echo "===== [5] 退出期段错误的 Python 侧现场（若有）====="
grep -alrE "Fatal Python error" "$HO"/log 2>/dev/null | head -5 | while read -r f; do
    echo "--- $f ---"
    grep -aA4 "Fatal Python error" "$f" | head -8
done

echo
echo "===== [6] 证据落盘 ====="
ls -la "$HO" "$HO/log" | head -30
date "+[%F %T]"
