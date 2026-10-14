#!/bin/bash
# PPU · T3 服务化腿（2026-10-14 v2）—— 在宿主（ai-server）上执行
# 仿 MLU590 run_mlu590_m1_serve_20261008.sh 配方。
#
# v2 修正（v1 踩坑）：
#   ① ⚠️ **宿主/容器路径映射写错**（第 8 次同族）：v1 把日志目录写成
#      HO=/bmcp_lvm_fs/hliu553/dc_t3_20261014，而容器内 DC_OUT_DIR=/workspace/probes_ppu/dc_t3_20261014
#      对应宿主是 /bmcp_lvm_fs/hliu553/**probes_ppu**/dc_t3_20261014 ⇒ 判据全 grep 在空目录上，
#      屏幕上只有 "rc=0" 而 verdict 一片空白。**规则：HO 必须由 O 按挂载映射推出，不得手写。**
#   ② 容器内 pip 的 index 指向不可达的 aliyun 内网地址（`pip index versions triton` 全超时）
#      ⇒ 不能用 pip 装依赖，改为从共享 wheel 目录直接复制（见 T3 报告）。
set -u

C=hliu553-dc-dev
P=/workspace/runtime-team/dev/device-context/prototype
O=/workspace/probes_ppu/dc_t3_20261014                 # 容器视角
HO=/bmcp_lvm_fs/hliu553/probes_ppu/dc_t3_20261014       # 宿主视角（= /workspace 的宿主侧）
DEV=${DEV:-1}   # ⚠️ 2026-10-14 实测：容器 dev0=ppu0 被他人占 30.7 GiB ⇒ vLLM 默认 0.9 利用率要 86.4 GiB 会直接 ValueError；dev1=ppu1 完全空闲
PORT=8100

rm -rf "$HO"; mkdir -p "$HO/form" "$HO/consumer"

echo "===== [0] 容器与版本 ====="
date "+[%H:%M:%S]"
docker ps --format '{{.Names}}|{{.Status}}' | grep "$C" || true
docker exec "$C" bash -lc 'python3 -c "import vllm, torch, triton; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__, \"triton\", triton.__version__); print(\"cuda_cnt\", torch.cuda.device_count())"'

M=$(docker exec "$C" bash -lc 'ls -d /workspace/models/Qwen3-Embedding-0.6B 2>/dev/null')
M=${M%$'\r'}
if [ -z "$M" ]; then
  echo "❌ 容器内取不到模型路径 ⇒ 中止"; exit 2
fi
docker exec "$C" bash -lc "test -f '$M/config.json' && echo 'MODEL ✓ config.json 在' || echo '✗ 缺 config.json'"

echo
echo "===== [A] serve_standard.sh（SERVE_FORM=embed · DEV=$DEV · EAGER=1 · STOP_AFTER=1）====="
date "+[%H:%M:%S] start"
docker exec -e DC_BACKEND=ppu -e SERVE_FORM=embed -e DEV=$DEV -e EAGER=1 \
    -e STOP_AFTER=1 -e READY_BUDGET=900 -e PORT=$PORT -e SMOKE_TIMEOUT=600 \
    -e MODEL="$M" -e SERVED_NAME=qwen3-embedding-0.6b \
    -e DC_OUT_DIR="$O/form" \
    "$C" bash -lc "cd $P && bash scripts/serve_standard.sh"
echo "form rc=$?"
date "+[%H:%M:%S] end"
echo "-- verdict 与关键行（宿主路径 $HO/form/serve_standard.log）--"
grep -aE "verdict|SERVE_STANDARD|维度|范数|就绪|耗时|✅|❌" "$HO/form/serve_standard.log" 2>/dev/null | tail -12
echo "---- vllm 实际启动参数 ----"
grep -aE "non-default args" "$HO/form/vllm_serve.log" 2>/dev/null | tail -2
if ! grep -qa "SERVE_STANDARD_PASS" "$HO/form/serve_standard.log" 2>/dev/null; then
  echo "---- [A] 未通过，取 vllm 日志的异常尾 ----"
  grep -aE "ERROR|Error|ModuleNotFound|RuntimeError" "$HO/form/vllm_serve.log" 2>/dev/null | grep -avE "NCCL INFO" | tail -8
fi

echo
echo "===== [B] proto_infer_serve.py（本层消费方）====="
date "+[%H:%M:%S] start serve (detached)"
docker exec -d -e DC_BACKEND=ppu -e SERVE_FORM=embed -e DEV=$DEV -e EAGER=1 \
    -e STOP_AFTER=0 -e READY_BUDGET=900 -e PORT=$PORT -e SMOKE_TIMEOUT=600 \
    -e MODEL="$M" -e SERVED_NAME=qwen3-embedding-0.6b \
    -e DC_OUT_DIR="$O/consumer" \
    "$C" bash -lc "cd $P && bash scripts/serve_standard.sh"

ready=0
for i in $(seq 1 90); do
  sleep 10
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then ready=1; echo "[i] 第 $((i*10)) s 就绪（http $code）"; break; fi
done
echo "ready=$ready"

if [ "$ready" = "1" ]; then
  date "+[%H:%M:%S] run consumer"
  docker exec -e DC_BACKEND=ppu -e SERVE_PORT=$PORT \
      "$C" bash -lc "cd $P && python3 runtime/proto/proto_infer_serve.py --rounds 5 --out $O/consumer/proto_infer_serve_result.json"
  echo "consumer rc=$?"
  grep -aE "verdict|passed|\"total\"" "$HO/consumer/proto_infer_serve_result.json" 2>/dev/null | tail -5
else
  echo "!! 服务未在预算内就绪 —— 看 $HO/consumer/vllm_serve.log"
  tail -25 "$HO/consumer/vllm_serve.log" 2>/dev/null
fi

echo
echo "===== [C] 停服务（只停自己起的；判据由结果决定）====="
docker exec "$C" bash -lc 'pkill -f "[s]erve_standard.sh" 2>/dev/null; sleep 1; pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "VLLM::Engin[e]Core" 2>/dev/null'
left=99
for i in $(seq 1 12); do
  sleep 5
  left=$(docker exec "$C" bash -lc 'ps -eo pid,stat,cmd | grep -E "[v]llm|[s]erve_standard" | grep -v "defunct" | wc -l')
  [ "$left" = "0" ] && break
done
echo "残留进程数 = $left（0=已清）"
echo "T3_DONE"
