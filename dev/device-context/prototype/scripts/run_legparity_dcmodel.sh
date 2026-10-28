#!/bin/bash
# =============================================================================
# `DC_MODEL` 两腿口径 · 跨实例复验核心（共享层改动的复验驱动）
#
# 为什么要它（2026-10-27）：
#   `proto_infer_leg` 有 `resolve_model()`（HF 缓存根 → snapshots/<hash>），
#   `proto_train_leg` **没有** ⇒ 同一份 `DC_MODEL=/hf_cache/hub/models--Qwen--…`：
#     推理腿 ✅ / 训练腿 ❌ `ValueError: Unrecognized model in …`
#   （10-27 MLU590 复跑实测暴露；当时在容器内先手工解析 snapshot 绕过，未夹带改共享层。）
#   修法＝抽成 `runtime/proto/model_path.py` 两腿共用（不是复制两份）+ 静态同源守卫。
#
# 本脚本按**判据**而不是按「跑通没报错」验收，四组判据：
#   P1 离线口径自检 `scripts/model_path_parity_check.py` 全过（23 项，含 AST 同源守卫）
#   P2 **同一份缓存根** ⇒ 推理腿与训练腿解析到**同一个** snapshot（逐字比对两腿打印行）
#   P3 **诱饵判据**：合成缓存里放一个字典序更大、且不是模型的 `zzz…` 目录，
#      正确实现必须听 `refs/main` 选真 snapshot；若退化成「排序取最后」会选中诱饵并**报错**
#   P4 回归：同一份 DC_MODEL 换成「普通模型目录」传法，两腿仍解析到同一路径、仍各自 PASS
#      ＋两次训练腿 50 步的 loss 读数对比（模型若被换掉，loss 不可能一样）
#
# ⚠️ 结论边界：本脚本只验「模型路径口径」这一改动面。不改 runtime API/后端 ⇒
#    离线契约 114/103/92、smoke、契约不变式、职责审计 78 项**不受影响**，本脚本不重复跑
#    （如需整轮回归请用各实例的 `run_*_rerun_*.sh`，那是另一个口径，别混引）。
#
# 用法：由各实例的薄封装调用，例如
#   bash dev/device-context/MLU590/probes/run_mlu590_legparity_20261027.sh
# 也可手工给全环境变量调用本核心（见下方「必给」）。
# =============================================================================
set -u

# ── 必给 ──────────────────────────────────────────────────────────────────────
: "${C:?容器名}"                 # 容器名
: "${PY:?容器内解释器}"           # 容器内 python（可带 conda 前缀）
: "${PROTO:?容器内 prototype 路径}"
: "${WORK:?容器内工作区（写 out 与合成缓存）}"
: "${HOST_WORK:?宿主视角的同一个工作区（落日志用；与 WORK 必须是同一目录的两种视角）}"
: "${MODEL:?容器内真实模型目录}"
: "${VIS_VAR:?选卡环境变量名}"    # ASCEND_RT_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES / MLU_VISIBLE_DEVICES
: "${DC_BACKEND:?后端名}"
# ── 可选 ──────────────────────────────────────────────────────────────────────
DEVS=${DEVS:-0,1}                 # 训练腿两卡（**容器内 0 基**）
DC_DIST_BT=${DC_DIST_BT:-}
MAX_STEPS=${MAX_STEPS:-50}        # 与接入手册训练腿口径一致（别为了快改小）
REAL_HASH=${REAL_HASH:-97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3}
DECOY_HASH=${DECOY_HASH:-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz}   # 字典序 > 真 hash

# ⚠️ 证据按 RUNID 分目录、**只追加**（这条坑踩过：驱动脚本用固定目录 + `rm -rf`/`: > log`
#    会静默销毁上一轮证据，曾把 15 行表格变成 3 行）⇒ 复跑不覆盖历史批次。
RUNID=${RUNID:-$(date +%Y%m%d_%H%M%S)}
O="$HOST_WORK/legparity_out/$RUNID"            # **宿主视角**：日志落在宿主（本脚本在宿主上跑）
OI="$WORK/legparity_out/$RUNID"                # **容器视角**：同一目录，交给 docker exec / python 用
HUB="$WORK/legparity_hub/models--Qwen--Qwen3-Embedding-0.6B"
PLAIN_DIR="$WORK/legparity_plain"
PLAIN="$PLAIN_DIR/Qwen3-Embedding-0.6B"        # 「普通目录」传法（软链到真模型，路径不同）
mkdir -p "$O" || { echo "❌ 宿主侧日志目录建不出来：$O（HOST_WORK 给错？）" >&2; exit 2; }
PASS=0; FAIL=0
chk() {  # chk <名称> <0=通过>
  if [ "$2" -eq 0 ]; then echo "  [PASS] $1"; PASS=$((PASS+1)); else echo "  [FAIL] $1"; FAIL=$((FAIL+1)); fi
}

echo "############################################################"
echo "# DC_MODEL 两腿口径复验  $(date '+%F %T')"
echo "# 容器=$C ｜ 解释器=$PY ｜ 后端=$DC_BACKEND ｜ 选卡变量 $VIS_VAR=$DEVS"
echo "# 模型（本实例真实目录）=$MODEL"
echo "# 宿主工作区=$HOST_WORK ｜ 容器工作区=$WORK"
echo "############################################################"

# ⚠️ **同一世界断言**（这条坑已累计踩过 9 次以上）：宿主路径与容器路径混用会让
#    `> $O/x.log` 在宿主建不出目录、`docker exec` 里的路径又指向别处，表现为「静默空转」。
#    这里用标记文件双向回读，确认 HOST_WORK 与 WORK 确实是**同一目录的两种视角**。
echo "legparity-$$-$(date +%s)" > "$HOST_WORK/.legparity_marker" || exit 2
_m_host=$(cat "$HOST_WORK/.legparity_marker")
_m_cont=$(docker exec "$C" bash -lc "cat '$WORK/.legparity_marker'" 2>/dev/null)
if [ "$_m_host" != "$_m_cont" ]; then
  echo "❌ 宿主/容器路径不是同一目录：HOST_WORK=$HOST_WORK 读到 '$_m_host'，" \
       "容器 $WORK 读到 '$_m_cont' ⇒ 先修映射再跑（否则整轮静默空转）" >&2
  exit 2
fi
echo "标记文件双向一致: $_m_host（宿主 $HOST_WORK ≡ 容器 $WORK）"

dex() {  # dex <日志名> <额外 -e 参数...> -- <容器内命令>
  local log="$1"; shift
  docker exec "$@" > "$O/$log.log" 2>&1
  local rc=$?
  echo "  rc=$rc（out/$log.log）"
  return $rc
}

echo ""
echo "=== [0] 版本自证（跑的是哪份代码）==="
docker exec "$C" bash -lc "cd $PROTO && md5sum runtime/proto/model_path.py runtime/proto/proto_infer_leg.py \
  runtime/proto/proto_train_leg.py scripts/model_path_parity_check.py" 2>&1 | sed 's/^/  /'
docker exec "$C" bash -lc "cd $PROTO && grep -c 'refs' runtime/proto/model_path.py" 2>/dev/null \
  | sed 's/^/  model_path 内 refs\/main 处理行数: /'
for leg in proto_infer_leg proto_train_leg; do
  docker exec "$C" bash -lc "cd $PROTO && grep -c 'from model_path import' runtime/proto/$leg.py" 2>/dev/null \
    | sed "s/^/  $leg 导入共用模块: /"
done

echo ""
echo "=== [1] 离线口径自检（P1：含 AST 同源守卫，无设备）==="
dex s1_offline "$C" bash -lc \
  "cd $PROTO && PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/model_path_parity_check.py"
grep -aE "通过 / |FAIL\]" "$O/s1_offline.log" | tail -3 | sed 's/^/    /'
grep -aq "0 失败" "$O/s1_offline.log"; chk "P1 离线口径自检 0 失败" $?

echo ""
echo "=== [1.5] 合成 HF 缓存（P3 诱饵：refs/main 指向真 hash，另放一个字典序更大的假目录）==="
docker exec "$C" bash -lc "
  set -e
  rm -rf '$HUB'
  mkdir -p '$HUB/snapshots' '$HUB/refs'
  ln -s '$MODEL' '$HUB/snapshots/$REAL_HASH'
  mkdir -p '$HUB/snapshots/$DECOY_HASH'          # 诱饵：**不是**模型目录，也无 config.json
  echo '$REAL_HASH' > '$HUB/refs/main'
  ls -l '$HUB/snapshots' | sed 's/^/    /'
  echo \"  refs/main = \$(cat '$HUB/refs/main')\"
  test -f '$HUB/snapshots/$REAL_HASH/config.json' && echo '  真 snapshot 有 config.json: OK'
  test -f '$HUB/snapshots/$DECOY_HASH/config.json' || echo '  诱饵无 config.json: OK（选中它必失败）'
  # 「普通目录」传法用一个**路径不同**的软链，这样 P4 才能分辨「两腿是否各自原样透传」
  mkdir -p '$PLAIN_DIR'
  ln -sfn '$MODEL' '$PLAIN'
  echo \"  普通目录传法 = $PLAIN -> \$(readlink '$PLAIN')\"
" 2>&1 | tail -9
docker exec "$C" bash -lc "test -f '$HUB/snapshots/$REAL_HASH/config.json' -a -f '$PLAIN/config.json'"
chk "P3-0 合成缓存 + 普通目录软链就绪" $?

echo ""
echo "=== [2] 推理腿 × HF 缓存根（P2 上半）==="
dex s2_infer_hub -e "$VIS_VAR=0" -e DC_BACKEND="$DC_BACKEND" -e DC_ROOT="$PROTO" -e DC_MODEL="$HUB" \
  -e DC_OUT_DIR="$OI" "$C" bash -lc \
  "cd $PROTO && PYTHONDONTWRITEBYTECODE=1 $PY -u runtime/proto/proto_infer_leg.py --backend $DC_BACKEND --out $OI/infer_hub.json"
grep -aE "\[model\]|INFER_LEG|区分度" "$O/s2_infer_hub.log" | tail -3 | sed 's/^/    /'
grep -aq "INFER_LEG_PASS" "$O/s2_infer_hub.log"; chk "P2-a 推理腿（缓存根）INFER_LEG_PASS" $?

echo ""
echo "=== [3] 训练腿 ${MAX_STEPS} 步 × HF 缓存根（P2 下半；**故意不做任何手工 snapshot 解析**）==="
dex s3_train_hub -e "$VIS_VAR=$DEVS" -e DC_BACKEND="$DC_BACKEND" -e DC_ROOT="$PROTO" \
  -e ${DC_DIST_BT:+DC_DIST_BT=$DC_DIST_BT} -e DC_MODEL="$HUB" -e DC_OUT_DIR="$OI" \
  -e MAX_STEPS="$MAX_STEPS" -e BATCH=4 -e SEQ=128 "$C" bash -lc \
  "cd $PROTO && PYTHONDONTWRITEBYTECODE=1 $PY -m torch.distributed.run --standalone --nproc_per_node=2 \
     runtime/proto/proto_train_leg.py --backend $DC_BACKEND"
grep -aE "\[model\]|TRAIN_LEG|loss=" "$O/s3_train_hub.log" | tail -4 | sed 's/^/    /'
grep -aq "TRAIN_LEG_PASS" "$O/s3_train_hub.log"; chk "P2-b 训练腿（缓存根）TRAIN_LEG_PASS" $?

echo ""
echo "=== [4] 推理腿 × 普通模型目录（P4 回归）==="
dex s4_infer_plain -e "$VIS_VAR=0" -e DC_BACKEND="$DC_BACKEND" -e DC_ROOT="$PROTO" -e DC_MODEL="$PLAIN" \
  -e DC_OUT_DIR="$OI" "$C" bash -lc \
  "cd $PROTO && PYTHONDONTWRITEBYTECODE=1 $PY -u runtime/proto/proto_infer_leg.py --backend $DC_BACKEND --out $OI/infer_plain.json"
grep -aE "\[model\]|INFER_LEG|区分度" "$O/s4_infer_plain.log" | tail -3 | sed 's/^/    /'
grep -aq "INFER_LEG_PASS" "$O/s4_infer_plain.log"; chk "P4-a 推理腿（普通目录）INFER_LEG_PASS" $?

echo ""
echo "=== [5] 训练腿 ${MAX_STEPS} 步 × 普通模型目录（P4 回归）==="
dex s5_train_plain -e "$VIS_VAR=$DEVS" -e DC_BACKEND="$DC_BACKEND" -e DC_ROOT="$PROTO" \
  -e ${DC_DIST_BT:+DC_DIST_BT=$DC_DIST_BT} -e DC_MODEL="$PLAIN" -e DC_OUT_DIR="$OI" \
  -e MAX_STEPS="$MAX_STEPS" -e BATCH=4 -e SEQ=128 "$C" bash -lc \
  "cd $PROTO && PYTHONDONTWRITEBYTECODE=1 $PY -m torch.distributed.run --standalone --nproc_per_node=2 \
     runtime/proto/proto_train_leg.py --backend $DC_BACKEND"
grep -aE "\[model\]|TRAIN_LEG|loss=" "$O/s5_train_plain.log" | tail -4 | sed 's/^/    /'
grep -aq "TRAIN_LEG_PASS" "$O/s5_train_plain.log"; chk "P4-b 训练腿（普通目录）TRAIN_LEG_PASS" $?

echo ""
echo "=== [6] 解析路径对账（P2/P3 的核心判据）==="
# ⚠️ 抽取只认**路径字符集**（`[A-Za-z0-9._/-]`），不要用 `.*`：
#    训练腿是 torchrun 双进程，两 rank 的 `[model] …` 行**可能被交错写进同一物理行**
#    （2026-10-27 轮 B 实测：rank1 那行尾部紧跟 rank0 的 `[model]`，用 `.*` 会把它一起吞掉
#     ⇒ 拿到的"路径"不等于期望值 ⇒ **判据自己误判 FAIL**。这是判据写法的脆弱，不是被测对象的问题）。
paths() { grep -aoE '→ 解析 [A-Za-z0-9._/-]+' "$O/$1.log" 2>/dev/null | sed 's/^→ 解析 //' | sort -u | tr '\n' '|'; }
IH=$(paths s2_infer_hub); TH=$(paths s3_train_hub)
IP=$(paths s4_infer_plain); TP=$(paths s5_train_plain)
echo "  推理腿·缓存根  → $IH"
echo "  训练腿·缓存根  → $TH"
echo "  推理腿·普通目录 → $IP"
echo "  训练腿·普通目录 → $TP"
EXPECT_HUB="$HUB/snapshots/$REAL_HASH"
[ "$IH" = "$TH" ] && [ "$IH" = "$EXPECT_HUB|" ]; chk "P2 缓存根：两腿解析同一 snapshot 且=refs/main 指向的真 hash" $?
case "$IH$TH" in
  *"$DECOY_HASH"*) chk "P3 未选中诱饵 snapshot" 1 ;;
  *)               chk "P3 未选中诱饵 snapshot" 0 ;;
esac
[ "$IP" = "$TP" ] && [ "$IP" = "$PLAIN|" ]; chk "P4 普通目录：两腿解析同一路径" $?

echo ""
echo "=== [7] 两次训练腿 loss 对比（模型被换掉则不可能相同）==="
for f in s3_train_hub s5_train_plain; do
  printf "  %-14s " "$f"
  grep -aoE "loss=[0-9.]+" "$O/$f.log" | tail -1
done
l1=$(grep -aoE "loss=[0-9.]+" "$O/s3_train_hub.log" | tail -1 | cut -d= -f2)
l2=$(grep -aoE "loss=[0-9.]+" "$O/s5_train_plain.log" | tail -1 | cut -d= -f2)
if [ -n "${l1:-}" ] && [ -n "${l2:-}" ]; then
  awk -v a="$l1" -v b="$l2" 'BEGIN{d=a-b; if(d<0)d=-d; printf "  |Δloss| = %.6f  %s\n", d, (d<1e-6?"（逐字相同 ⇒ 两条传法喂的是同一份权重）":"⚠️ 有差异，需解释后才可引用")}'
fi

echo ""
echo "############ 汇总 ############"
echo "  DC_MODEL 口径复验: $PASS 通过 / $FAIL 失败"
echo "  证据目录：宿主 $O ｜ 容器 $OI"
echo "  ⚠️ 结论边界：只覆盖「模型路径口径」改动面；离线 114/103/92、smoke、契约不变式、审计 78 不在本轮范围。"
[ "$FAIL" -eq 0 ] && echo "VERDICT: LEGPARITY_PASS" || echo "VERDICT: LEGPARITY_FAIL"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
