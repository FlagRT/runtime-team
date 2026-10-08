#!/bin/bash
# MLU590 · 补齐轮 m1 第 4 组：服务化腿（2026-10-08）
# **在宿主（Mlu-1）上执行** —— 它要调 `docker exec`，不能在容器内跑
#   （容器内没有 docker；且本脚本的重定向要用**宿主**路径，给容器程序的 `--out` 用**容器**路径
#    —— 混用会得到一堆 rc=1 且日志为空的怪现象，P800 上踩过两次）。
#
# 覆盖依据 = `MLU590/docs/MLU590_FIX_WORKPACK_20261008.md` §3 第 4 组（序 20 / 21）。
#   · 序 20 `SERVE_FORM=embed` 的 `serve_standard.sh` —— 历史值 `SERVE_STANDARD_PASS`
#     （就绪 150 s、维度 1024、范数 1.000001）
#   · 序 21 `proto_infer_serve.py`（本层消费方，`create_stream()` 的最后一个消费方）
#     —— **本实例从未跑过**（`MLU590/probes/` 只有 `accept_serve_*`，无 `infer_serve` 证据）
#
# ⚠️ 四条前提（前三条是三条腿都适用的，第四条是本轮首跑踩出来的）：
#   ① 服务化**必须用 vLLM 应用镜像容器**（运行时镜像不含 vLLM）；
#   ② 结果必须读 `$DC_OUT_DIR/serve_standard.log` —— 脚本内部第一件事就是
#      `exec > serve_standard.log 2>&1`，外层重定向得到的是 0 字节空文件；
#   ③ 就绪轮询用端口 **8100**（`serve_standard.sh` 默认 `PORT=8100`，不是 8000）——
#      写错不会让服务起不来，只会让你空等到超时（“等得久 ≠ 服务没起来”）；
#   ④ ⭐ **MODEL 是"给容器程序的路径"，必须在容器内解析**：
#      2026-10-08 首跑用宿主上的 `ls /hf_cache/...` 取值 ⇒ 宿主**没有** `/hf_cache`
#      （宿主是 `/srv/data/hf_cache`）⇒ 变量为空 ⇒ 退到脚本默认值、整轮失败。
#      这是运行手册 §3.5「宿主/容器路径不得混用」的第 N 次复现 ⇒ 现改为 `docker exec` 内解析，
#      **并加非空断言**（前置条件显式化）。
set -u

C=${1:?用法: bash run_mlu590_m1_serve_20261008.sh <vllm容器名>}
P=/work/dc_mlu_regen_20261008/prototype
O=/work/dc_mlu_regen_20261008/out_serve_m1
HO=/srv/hliu553/dc_mlu_regen_20261008/out_serve_m1
DEV=2                       # 当次普查：卡 2 完全空闲；卡 0/1 有他人进程
PORT=8100
SPORT=$PORT

mkdir -p "$HO/form" "$HO/consumer"

echo "===== [0] 容器与版本 ====="
date "+[%H:%M:%S]"
docker ps --format '{{.Names}}|{{.Status}}' | grep "$C" || true
docker exec "$C" bash -lc 'python3 -c "import vllm, torch, torch_mlu; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__, \"torch_mlu\", torch_mlu.__version__); print(\"mlu_cnt\", torch.mlu.device_count())"'

# ⭐ MODEL 必须在**容器内**解析（给容器程序的路径）
M=$(docker exec "$C" bash -lc 'ls -d /hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/ 2>/dev/null | head -1')
M=${M%$'\r'}                                # 去掉可能的 CR
if [ -z "$M" ]; then
  echo "❌ 容器内取不到模型路径 ⇒ 中止（不要让它退到某个默认值上）"
  docker exec "$C" bash -lc 'ls -l /hf_cache/hub/ 2>&1 | head'
  exit 2
fi
echo "MODEL(容器内) = $M"
docker exec "$C" bash -lc "test -f '$M/config.json' && echo '  config.json ✓' || echo '  ✗ 缺 config.json'"

echo
echo "===== [A] 序 20：serve_standard.sh（SERVE_FORM=embed · DEV=$DEV · EAGER=1 · STOP_AFTER=1）====="
echo "     ⭐ 本段**刻意不传 MODEL** —— 为验证同日修好的 cambricon 分支默认值真的可用"
echo "        （原默认是宿主路径 /srv/hliu553/models/... —— 容器内永不存在）"
date "+[%H:%M:%S] start"
docker exec -e DC_BACKEND=cambricon -e SERVE_FORM=embed -e DEV=$DEV -e EAGER=1 \
    -e STOP_AFTER=1 -e READY_BUDGET=900 -e PORT=$PORT \
    -e SERVED_NAME=qwen3-embedding-0.6b \
    -e DC_OUT_DIR="$O/form" \
    "$C" bash -lc "cd $P && bash scripts/serve_standard.sh"
echo "form rc=$?"
date "+[%H:%M:%S] end"
echo "-- verdict 与关键行（取自 serve_standard.log，非外层重定向）--"
grep -aE "verdict|SERVE_STANDARD|维度|范数|就绪|ready|✅|❌" "$HO/form/serve_standard.log" | tail -10
echo "---- vllm 实际启动参数（确认形态与模型真的生效）----"
grep -aE "non-default args" "$HO/form/vllm_serve.log" 2>/dev/null | tail -2

echo
echo "===== [B] 序 21：proto_infer_serve.py（本层消费方）====="
echo "     本段**显式传 MODEL=$M** 与 [A] 形成互补（一条走默认、一条走显式）"
# 先以 STOP_AFTER=0 起服务（detached），轮询 8100 就绪后再跑消费方
date "+[%H:%M:%S] start serve"
docker exec -d -e DC_BACKEND=cambricon -e SERVE_FORM=embed -e DEV=$DEV -e EAGER=1 \
    -e STOP_AFTER=0 -e READY_BUDGET=900 -e PORT=$PORT \
    -e MODEL="$M" -e SERVED_NAME=qwen3-embedding-0.6b \
    -e DC_OUT_DIR="$O/consumer" \
    "$C" bash -lc "cd $P && bash scripts/serve_standard.sh"

ready=0
for i in $(seq 1 45); do
  sleep 10
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${SPORT}/v1/models" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then ready=1; echo "[i] 第 $((i*10)) s 就绪（http $code）"; break; fi
done
echo "ready=$ready"

if [ "$ready" = "1" ]; then
  date "+[%H:%M:%S] run consumer"
  # ⚠️ 刻意**不传** `--backend`：验证它同日修好后真的读 DC_BACKEND
  docker exec -e DC_BACKEND=cambricon -e SERVE_PORT=$SPORT \
      "$C" bash -lc "cd $P && python3 runtime/proto/proto_infer_serve.py --rounds 5 --out $O/consumer/proto_infer_serve_result.json"
  echo "consumer rc=$?"
  tail -6 "$HO/consumer/proto_infer_serve_result.json" 2>/dev/null
else
  echo "!! 服务未在预算内就绪 —— 先看 $HO/consumer/vllm_serve.log 与 serve_standard.log"
  tail -20 "$HO/consumer/vllm_serve.log" 2>/dev/null
fi

echo
echo "===== [C] 停服务（只停自己起的）====="
# ⚠️ 2026-10-08 首跑缺陷：原写 `ps | head -5; echo "(上方为空=已清)"` ——
#    **"已清"是无条件打印的**，而首跑的 ps 里明明还有一条 vllm（当时处于 `<defunct>` 僵尸态、
#    端口已关）⇒ 那句话在说谎。判据必须由**结果**决定，不能靠人去看上方。
docker exec "$C" bash -lc 'pkill -f "[s]erve_standard.sh" 2>/dev/null; sleep 1; pkill -f "[v]llm serve" 2>/dev/null'
for i in $(seq 1 12); do
  sleep 5
  left=$(docker exec "$C" bash -lc 'ps -eo pid,stat,cmd | grep -E "[v]llm|[s]erve_standard" | grep -v "defunct" | wc -l')
  [ "$left" = "0" ] && break
done
left=$(docker exec "$C" bash -lc 'ps -eo pid,stat,cmd | grep -E "[v]llm|[s]erve_standard" | grep -v "defunct" | wc -l')
if [ "$left" = "0" ]; then
  echo "  ✅ 服务进程已清（等待 $((i*5))s；<defunct> 僵尸由父进程回收，不算残留）"
else
  echo "  ❌ 仍有 $left 条存活进程 —— 需要人工处理："
  docker exec "$C" bash -lc 'ps -eo pid,stat,cmd | grep -E "[v]llm|[s]erve_standard" | grep -v defunct'
fi
echo "-- 端口检查（真实判据：8100 应不可达）--"
curl -s -o /dev/null -w "  http=%{http_code}（000/非200 = 已停）\n" --max-time 5 "http://127.0.0.1:${SPORT}/v1/models" 2>/dev/null || echo "  http=连接失败（= 已停）"

echo "===== ALL DONE ====="
date "+[%H:%M:%S]"
