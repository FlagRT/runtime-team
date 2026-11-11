#!/bin/bash
# 重新生成 segv/*.txt：用更宽的过滤，保证「干净」的对照组也有实质内容
set -uo pipefail
WS=/bmcp_lvm_fs/hliu553
EV=$WS/probes_ppu/evidence_20261010

declare -A LABEL=(
  [e11_verbatim]="逐字副本（只改内容以外的东西都不改）"
  [e13_ptl_nostream]="副本：仅把 st = runtime.create_stream() 改成 st = None"
  [e13_ptl_noruntime]="副本：整块移除原型运行时调用"
  [e12_ptl_notimeout]="副本：去掉 init_process_group 的 timeout"
  [e12_ptl_nopreflight]="副本：去掉 _preflight_env_check()"
  [e14_torchstream]="同位置/同生命周期：原生 torch.cuda.Stream()"
  [e14_runtimestream]="同位置/同生命周期：我方 runtime.create_stream()"
  [e5]="完整训练循环（无原型）"
  [e2_runtime_only]="仅原型（无模型）"
  [e3_model_only]="仅模型（无原型）"
  [e4_p2p_gather]="all_reduce + all_gather + P2P（无模型无原型）"
  [e10_stream]="原型三档深度（nodev/setdev/stream）"
  [e15_base]="del st + gc.collect()（回收 0 对象）后是否仍崩"
  [e16_after]="建组与建流的先后顺序"
)

for T in "${!LABEL[@]}"; do
  L="$WS/probes_ppu/segv_isolate/$T/run.log"
  [ -f "$L" ] || continue
  {
    echo "# 对照：${LABEL[$T]}"
    echo "# 源日志：$L （下列为滤掉 NCCL INFO 后的判定/崩溃行）"
    if grep -q "Segmentation fault" "$L"; then
      echo "# 结论：**复现段错误**（判据行若存在则说明判据仍通过）"
    else
      echo "# 结论：**干净退出**（无 Segmentation fault，且到达结束行）"
    fi
    echo
    grep -vE "NCCL INFO" "$L" \
      | grep -E "TRAIN_LEG_(PASS|FAIL)|INFER_LEG_(PASS|FAIL)|\[e[0-9]+[^]]*\]|Segmentation fault|Fatal Python|exitcode|gc.collect|Loss =|loss [0-9]" \
      | head -18
  } > "$EV/segv/$T.txt"
  printf "%-24s %s\n" "$T" "$(grep -c . "$EV/segv/$T.txt") 行"
done
echo "=== 全部对照的复现/干净一览 ==="
for F in "$EV"/segv/*.txt; do
  printf "%-26s %s\n" "$(basename "$F" .txt)" "$(grep -m1 '^# 结论' "$F" | sed 's/# 结论：//')"
done
