#!/bin/bash
# T11 修法对照：把「本层登记表里的弱引用」这一因素逐个去掉，看 139 是否消失
# 载体：① 最小复现 ours_default_dropped ② T4 错误闭环 ③ T6 调度效果探针
# 纪律：轮次交替（每轮内跑遍所有组合），避免把「时间漂移」读成「变量效应」
set -u
HO=/bmcp_lvm_fs/hliu553/probes_ppu/dc_t11_20261014
O=/workspace/probes_ppu/dc_t11_20261014
P=/workspace/runtime-team/dev/device-context/prototype
C=hliu553-dc-dev
PY=/usr/local/bin/python3
ROUNDS=${ROUNDS:-2}
PATCHES=${PATCHES:-"control note_off strong"}

# sitecustomize（注入入口）= t11_patch.py 的**真实副本**
# ⚠️ 两次踩坑（都表现成"补丁无效"而不是"补丁没装上"，属最贵的假象）：
#    ① 用 `ln -sf $HO/t11_patch.py $HO/sitecustomize.py`：软链指向**宿主**路径 ⇒ 容器内是断链；
#    ② 想用 `cp -f` 覆盖软链：软链目标就是 t11_patch.py 自身 ⇒ cp 写回原文件，软链依旧。
#    ⇒ 必须 `rm -f` 再 `cp`，并在**容器内**做存在性断言（宿主看得见 ≠ 容器看得见）。
rm -f "$HO/sitecustomize.py"
cp -f "$HO/t11_patch.py" "$HO/sitecustomize.py"
grep -q "T11_PATCH_LOADED" "$HO/sitecustomize.py" && echo "[pre] 宿主侧 sitecustomize 内容校验通过"
docker exec "$C" bash -lc "ls -l $O/sitecustomize.py && grep -q T11_PATCH_LOADED $O/sitecustomize.py && echo '[pre] 容器侧 sitecustomize 就位'" \
    || { echo "!! 容器侧 sitecustomize 未就位 ⇒ 停止（不得在注入未生效的情况下读结果）"; exit 3; }

run_one() {  # $1=载体标识 $2=命令
    local tag=$1; shift
    local cmd=$1
    docker exec -e DC_BACKEND=ppu -e DC_ROOT=$P -e CUDA_VISIBLE_DEVICES=1 \
        -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$O -e T11_PATCH=$PATCH -e DC_OUT_DIR=$O -e DC_TAG=_pat \
        -e LD_PRELOAD=$O/t11_segv_handler.so \
        "$C" bash -lc "cd $P && $PY -u $cmd" > "$HO/patch__${tag}__${PATCH}_r${r}.log" 2>&1
    local rc=$?
    echo "$rc" > "$HO/patch__${tag}__${PATCH}_r${r}.rc"
    # 注入必须自证：否则"补丁没生效"会被读成"补丁无效"
    local mk
    mk=$(grep -a -m1 -o "T11_PATCH_LOADED\|T11_PATCH_FAILED\|（未加载）" "$HO/patch__${tag}__${PATCH}_r${r}.log" || true)
    grep -aq "T11_PATCH_LOADED\|T11_PATCH_FAILED" "$HO/patch__${tag}__${PATCH}_r${r}.log" || mk="（未加载）"
    printf "  round=%s patch=%-9s carrier=%-14s rc=%-4s injection=%s\n" "$r" "$PATCH" "$tag" "$rc" "$mk" | tee -a "$HO/patch_matrix.log"
}

# ⚠️ 2026-10-14 修（自审发现）：原为 `: > patch_matrix.log`（清空）⇒ 复跑销毁上一批记录。
#    改为**只追加**，每批带 run 头；阅读时取最新 run 段。
echo "===== run=${RUNID:-$(date +%Y%m%d_%H%M%S)}  patches=[$PATCHES] rounds=$ROUNDS  开始 $(date '+%F %T') =====" >> "$HO/patch_matrix.log"
for r in $(seq 1 "$ROUNDS"); do
    for PATCH in $PATCHES; do
        run_one "min_dropped"   "$O/t11_exit_segv_matrix.py ours_default_dropped"
        run_one "t4_errorloop"  "runtime/proto/proto_error_recovery_loop.py --backend ppu --out $O/patch_t4_${PATCH}_r${r}.json"
        run_one "t6_sched"      "probes/probe_stream_priority_sched_effect.py --dev 0 --out $O/patch_t6_${PATCH}_r${r}.json"
    done
done

echo
echo "===== 汇总（rc + 注入自证）====="
HOSTPY=$(command -v python3 || echo "")
if [ -n "$HOSTPY" ]; then
$HOSTPY - <<PYEOF 2>/dev/null || true
import collections, pathlib
hd = pathlib.Path("$HO")
rows = collections.defaultdict(list)
inj = collections.defaultdict(set)
cap = collections.defaultdict(set)
for p in sorted(hd.glob("patch__*_r*.rc")):
    tag, patch = p.stem.split("_r")[0].replace("patch__", "").split("__")
    rows[(tag, patch)].append(p.read_text().strip())
for p in sorted(hd.glob("patch__*_r*.log")):
    tag, patch = p.stem.split("_r")[0].replace("patch__", "").split("__")
    txt = p.read_text(errors="ignore")
    inj[(tag, patch)].add("loaded" if "T11_PATCH_LOADED" in txt else ("FAILED" if "T11_PATCH_FAILED" in txt else "NOT_LOADED"))
    if "T11_CRASH_CAPTURED" in txt:
        cap[(tag, patch)].add("captured")
print("%-14s %-9s %-13s %-12s %s" % ("carrier", "patch", "非0退出/总数", "注入自证", "原生现场"))
for k in sorted(rows):
    rs = rows[k]
    print("%-14s %-9s %d/%d            %-12s %s" % (
        k[0], k[1], sum(1 for x in rs if x != "0"), len(rs),
        ",".join(sorted(inj[k])), ",".join(sorted(cap[k])) or "-"))
print()
print("rc 序列：")
for k in sorted(rows):
    print("  %-14s %-9s %s" % (k[0], k[1], " ".join(rows[k])))
PYEOF
fi
