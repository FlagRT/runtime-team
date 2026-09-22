#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# serve_standard.sh — 统一推理服务启动脚本（**组内服务启动标准**）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 【这份脚本解决什么】
#   此前 910C 与 P800 各自维护一份启动脚本（910C 62 行 / P800 91 行），
#   环境变量、参数口径、停机清理各写各的 —— 下游各方向再各抄一份，就会分叉。
#   本脚本把"启动推理服务"收敛成**一条命令 + 一组环境变量**，跨芯片只改 DC_BACKEND。
#
# 【与两套旧脚本的关系（差异点均已在内部分支处理，不是删旧脚本）】
#   910C：`910C/distributed_inference/inference/start_vllm_serve_910c.sh`
#         · 坑 A5 禁用 vllm-plugin-FL（unset VLLM_PLUGINS）
#         · 坑 A3 DO_NOT_TRACK=1
#         · 坑 A2 清理残留 EngineCore 子进程
#         · 选卡 ASCEND_RT_VISIBLE_DEVICES
#         · 另有 D10/D11 集成（错误翻译包装器 + 设备状态监控）——见下方"集成层说明"
#   P800：`P800/probes/F2_vllm_serve.sh`
#         · PYTHONPATH=/env/FlagGems/src（硬前置）
#         · VLLM_FL_PLATFORM=kunlunxin / VLLM_FL_PREFER=vendor / USE_FLAGGEMS=0 /
#           GEMS_VENDOR=kunlunxin / KLX_USE_AUTOTUNE=0
#         · 选卡 CUDA_VISIBLE_DEVICES
#         · 停机必须连带清理 EngineCore 残留（实测占卡 73850 MiB / 96 GiB）
#   MLU590（寒武纪，2026-09-22 新增，⚠️ 尚未真机验证）：
#         · 设备 API 走 torch.mlu（PrivateUse1），选卡 MLU_VISIBLE_DEVICES
#         · 镜像定档 flagos-runtime-cambricon-neuware4.4.3:2.2.0（宿主驱动 v6.2.29 同 6.2.x 线）
#         · 推理形态**待实测**：寒武纪有厂商移植版 vLLM（Cambricon/vllm-mlu），
#           但是否需要在社区 vLLM 之外额外装插件、--runner pooling 是否被支持，均未验证
#           ⇒ 首次跑请保留本脚本日志，按实际报错回填本节
#
# 【统一的服务参数口径】（两实例同口径，便于横向比对）
#     --runner pooling --convert embed --max-model-len 4096 --port 8100
#   ⚠️ 该版本 vLLM **没有 --task 参数**；embedding 服务必须用 --runner pooling --convert embed
#      （写 --task embed 会报 `vllm: error: unrecognized arguments: --task embed`）。
#
# 【功能冒烟（两形态都有，且纳入 verdict）】
#     embedding 形态（kunlun / cambricon）→ POST /v1/embeddings，校验维度与范数
#     生成形态     （ascend）→ POST /v1/completions，校验产出非空且 completion_tokens>0
#   ⚠️ "就绪" ≠ "可用"：只看 /v1/models 返回 200 会掩盖"服务起来了但算不出"的情况。
#      2026-09-20 补齐（此前生成形态只验就绪，与 embedding 形态强度不对等）。
#
# 【绑定地址】HOST 默认 127.0.0.1（不对外暴露）。跨容器 / 跨机访问需显式 HOST=0.0.0.0，
#   并自行确认网络与访问控制策略。
#
# 【集成层说明（D10/D11）——如实标注当前范围】
#   910C 版脚本额外挂了两项集成：错误码翻译包装器（inject_error_translation.serve()）
#   与设备状态监控（device_state_monitor.py），资产在 910C 目录下。
#   本统一脚本**保留开关**（ERROR_TRANSLATION / MONITOR），但当前**仅在 910C 上可用**；
#   P800 侧需先把这两个资产上提到 prototype 侧才能启用 —— 见标准文档"待上提项"。
#   基础启动路径（起服务 / 就绪 / 健康检查 / 停机）在两个芯片上均已验证。
#
# 【用法】
#   # P800（容器内）
#   DC_BACKEND=kunlun DEV=6 MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B \
#     bash prototype/scripts/serve_standard.sh
#   # 910C（容器内）
#   DC_BACKEND=ascend MODEL=/mnt/raid/hliu553/models/Qwen3-4B SERVED_NAME=qwen3-4b \
#     bash prototype/scripts/serve_standard.sh
#   # MLU590（寒武纪，容器内；首次建议 STOP_AFTER=1 起完即停）
#   DC_BACKEND=cambricon DEV=0 STOP_AFTER=1 MODEL=/srv/hliu553/models/Qwen3-Embedding-0.6B \
#     bash prototype/scripts/serve_standard.sh
#
# 【环境变量】
#   DC_BACKEND      后端名（ascend | kunlun | cambricon），默认 ascend
#   SERVE_FORM      服务形态 `embed` | `generate`（留空=按后端默认：
#                   ascend→generate 沿用 910C 既有口径，kunlun/cambricon→embed。
#                   2026-09-22 补：验收模型统一为 Qwen3-Embedding-0.6B 后，
#                   三实例需能起**同形态**服务才能横向比对，故加此覆盖开关）
#   MODEL           模型路径（**须给到 snapshots/<hash>**，给缓存根目录会报 Unrecognized model）
#   SERVED_NAME     服务暴露的模型名（默认按后端给）
#   PORT            端口（8100）
#   DEV             选卡（ascend→ASCEND_RT_VISIBLE_DEVICES / kunlun→CUDA_VISIBLE_DEVICES
#                   / cambricon→MLU_VISIBLE_DEVICES）
#   TP              tensor parallel（1）
#   MAX_MODEL_LEN   （4096）
#   GPU_MEM_UTIL    （留空则不传；P800 实测共享机建议 0.25）
#   EAGER           1=加 --enforce-eager（默认 1）；0=启用图捕获
#                   （P800 实测：去掉 eager 反而更慢 p50 132ms，故默认保留）
#   DC_OUT_DIR      日志与 pid 目录（默认 /tmp/dc_serve）
#   DC_CONDA_ENV    vllm 不在 PATH 时尝试激活的 conda 环境名（默认 python310_torch29_cuda，P800 用）
#   STOP_AFTER      1=就绪并健康检查后立即停机（验证用）；0=保持运行（下游起服务用，默认 0）
#   READY_BUDGET    就绪等待上限秒数（180）
#   HOST            绑定地址（127.0.0.1；跨容器/跨机访问需设 0.0.0.0）
#   ERROR_TRANSLATION / MONITOR  集成层开关（默认 0；**当前仅 910C 可用**）
#
# 【verdict 口径】STOP_AFTER=1 时输出 `SERVE_STANDARD_PASS` 需 **ready=1 且 smoke=1**；
#   任一不满足即 `SERVE_STANDARD_FAIL (ready=? smoke=?)`。
# ═══════════════════════════════════════════════════════════════════════════════
set -u

BACKEND=${DC_BACKEND:-ascend}
PORT=${PORT:-8100}
TP=${TP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-}
EAGER=${EAGER:-1}
DEV=${DEV:-}
DC_OUT_DIR=${DC_OUT_DIR:-/tmp/dc_serve}
STOP_AFTER=${STOP_AFTER:-0}
READY_BUDGET=${READY_BUDGET:-180}
ERROR_TRANSLATION=${ERROR_TRANSLATION:-0}
MONITOR=${MONITOR:-0}
HOST=${HOST:-127.0.0.1}

mkdir -p "$DC_OUT_DIR"
LOG="$DC_OUT_DIR/serve_standard.log"
SRV_LOG="$DC_OUT_DIR/vllm_serve.log"
PIDFILE="$DC_OUT_DIR/serve_standard.pid"
exec > "$LOG" 2>&1

echo "=========== 统一推理服务启动 开始 $(date) ==========="
echo "backend=$BACKEND dev=${DEV:-<全部可见>} port=$PORT tp=$TP max_model_len=$MAX_MODEL_LEN eager=$EAGER"
echo "stop_after=$STOP_AFTER（1=验证后停机；0=保持运行）"

# ─────────────────────────────────────────────────────────────────────────────
# 1) 后端环境（两芯片的差异全部收敛在这里；其余流程完全共用）
# ─────────────────────────────────────────────────────────────────────────────
case "$BACKEND" in
  ascend)
    MODEL=${MODEL:-/mnt/raid/hliu553/models/Qwen3-4B}
    SERVED_NAME=${SERVED_NAME:-qwen3-4b}
    # 坑 A5：A 线禁用 vllm-plugin-FL（该插件无 ascend 后端，启用后 platform.device_type 变空
    #         → RuntimeError: Device string must not be empty）
    unset VLLM_PLUGINS
    # 坑 A3
    export DO_NOT_TRACK=1
    [ -n "$DEV" ] && export ASCEND_RT_VISIBLE_DEVICES="$DEV"
    SKIP_EXTRA=1        # 910C 既有脚本不传 --runner pooling（生成类服务）
    DEV_API=npu         # 供 card_snapshot 的 torch 侧降级查询使用
    ;;
  kunlun)
    MODEL=${MODEL:-/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B}
    SERVED_NAME=${SERVED_NAME:-qwen3-embedding-0.6b}
    # 硬前置：site-packages 里的 flag_gems 子模块不完整，vllm_fl import 会失败
    #         → 缺它报 "Failed to infer device type"（看着像设备问题，其实是包问题）
    export PYTHONPATH=/env/FlagGems/src${PYTHONPATH:+:$PYTHONPATH}
    # 算子路径取 vendor：避开 FlagGems 路径及其 KL3 依赖（与本方向锁定口径一致）
    export VLLM_FL_PLATFORM=kunlunxin VLLM_FL_PREFER=vendor USE_FLAGGEMS=0
    export GEMS_VENDOR=kunlunxin KLX_USE_AUTOTUNE=0
    [ -n "$DEV" ] && export CUDA_VISIBLE_DEVICES="$DEV"
    SKIP_EXTRA=0
    DEV_API=cuda
    ;;
  cambricon)
    # ⚠️ 本分支 2026-09-22 按规范新写，**尚未真机验证**（寒武纪机器当时缺 docker 组权限）。
    #    首次在 MLU 容器内跑时，请把实际报错回填到本分支与《寒武纪接入方案》。
    MODEL=${MODEL:-/srv/hliu553/models/Qwen3-Embedding-0.6B}
    SERVED_NAME=${SERVED_NAME:-qwen3-embedding-0.6b}
    # 选卡：torch_mlu 认 MLU_VISIBLE_DEVICES（⚠️ 待容器内实测确认）
    [ -n "$DEV" ] && export MLU_VISIBLE_DEVICES="$DEV"
    # 暂不设任何厂商专用算子/插件环境变量 —— 未实测前不臆造（前两家的变量互不通用，
    # 手册 §9 坑 5 明确「同一插件跨芯片可用性可以完全相反，须逐个实测」）
    SKIP_EXTRA=0        # 本方向的验收模型是 embedding 模型 ⇒ 走 embedding 服务形态
    DEV_API=mlu         # 供 card_snapshot 的 torch 侧降级查询使用
    ;;
  *)
    echo "❌ 未知 DC_BACKEND=$BACKEND（支持 ascend | kunlun | cambricon）"; exit 2 ;;
esac

# ─────────────────────────────────────────────────────────────────────────────
# 1b) 服务入口就绪（vllm 可执行文件）
#     跨芯片差异：910C 镜像自带（/usr/local/python3.11.15/bin/vllm）；
#     P800 的 vLLM 装在 conda 环境 `python310_torch29_cuda` 里，
#     **不激活就直接 nohup: failed to run command 'vllm': No such file or directory**
#     —— 2026-09-20 910C 补跑时顺带发现的缺口（下游照抄同样会踩），此处自动激活。
# ─────────────────────────────────────────────────────────────────────────────
if ! command -v vllm >/dev/null 2>&1; then
  CONDA_ENV=${DC_CONDA_ENV:-python310_torch29_cuda}
  echo "--- vllm 不在 PATH，尝试激活厂商 python 环境（$CONDA_ENV）---"
  for CAND in /root/miniconda/etc/profile.d/conda.sh /opt/conda/etc/profile.d/conda.sh; do
    [ -f "$CAND" ] || continue
    # shellcheck disable=SC1090
    . "$CAND" 2>/dev/null || continue
    if conda activate "$CONDA_ENV" 2>/dev/null; then
      echo "  已激活：$CONDA_ENV"; break
    fi
  done
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo "❌ 找不到 vllm 可执行文件（服务入口不可用）"
  echo "   910C：镜像自带 vllm（/usr/local/python3.11.15/bin/vllm）——确认容器镜像是否为 vllm-ascend"
  echo "   P800：vLLM 在 conda 环境内，需 DC_CONDA_ENV=<env 名>（默认 python310_torch29_cuda）"
  echo "   手动方式：source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda"
  exit 3
fi
echo "  vllm = $(command -v vllm) ｜ python3 = $(command -v python3)（$(python3 -V 2>&1)）"

# ─────────────────────────────────────────────────────────────────────────────
# 2) 停机清理（坑 A2 / P800 实测教训：EngineCore 子进程残留持续占卡）
#    变量拼接避免 pkill 命中自身（坑）
# ─────────────────────────────────────────────────────────────────────────────
cleanup() {
  P1=VLLM::Engin; P2="vllm serve"
  pkill -9 -f "${P1}eCore" 2>/dev/null || true
  pkill -9 -f "$P2" 2>/dev/null || true
  [ "$MONITOR" = "1" ] && { pkill -9 -f "device_state_monitor" 2>/dev/null || true; }
  sleep 3
}

# 用卡现状（便于前后对比；各家工具名不同）
#   ⚠️ npu-smi / xpu-smi / cnmon 都是**宿主工具**，容器内通常不存在
#      （910C 的 vllm-ascend 镜像实测 `npu-smi: command not found`），
#      故补一层 torch 侧查询（按后端取 torch.npu / torch.cuda / torch.mlu），
#      保证容器内日志也有卡状态。
card_snapshot() {
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info 2>/dev/null | grep -E "^\| [0-9]+" | head -8
  elif command -v xpu-smi >/dev/null 2>&1; then
    xpu-smi 2>/dev/null | awk '/MiB \//{print}' | head -8
  elif command -v cnmon >/dev/null 2>&1; then
    # 寒武纪：cnmon 的显存行形如 "Used(MiB)：xxx / Total(MiB)：xxx"
    # ⚠️ 输出格式未在容器内验证过，故两种常见形态都抓一遍，抓不到就走下面的 torch 侧降级
    cnmon 2>/dev/null | grep -iE "MiB" | head -8
  else
    python3 - "$DEV_API" <<'PY' 2>/dev/null || echo "  (无 smi 工具，torch 侧查询亦不可用)"
import sys, torch
name = sys.argv[1]
# 厂商扩展需显式导入才会注册对应的 torch 命名空间
# （torch_npu→torch.npu / torch_mlu→torch.mlu；cuda 无需导入）
if name == "npu" and not hasattr(torch, "npu"):
    try:
        __import__("torch_npu")
    except Exception:
        pass
if name == "mlu" and not hasattr(torch, "mlu"):
    try:
        __import__("torch_mlu")
    except Exception:
        pass
api = getattr(torch, name, None)
if api is None or not hasattr(api, "mem_get_info"):
    raise SystemExit(1)
n = api.device_count()
for i in range(n):
    try:
        free, total = api.mem_get_info(i)
        print(f"  [torch.{sys.argv[1]}:{i}] free={free/2**30:.2f}GiB / total={total/2**30:.2f}GiB")
    except Exception as e:
        print(f"  [torch.{sys.argv[1]}:{i}] 查询失败：{type(e).__name__}")
    if i >= 7:
        print("  ... （仅列前 8 个）"); break
PY
  fi
}

echo "--- 用卡前 ---"; card_snapshot
cleanup

# ─────────────────────────────────────────────────────────────────────────────
# 3) 启动（服务参数统一口径）
# ─────────────────────────────────────────────────────────────────────────────
# 形态覆盖（SERVE_FORM）：留空则保持各后端分支的既有默认 ⇒ 本开关**零行为变更**
case "${SERVE_FORM:-}" in
  embed)    SKIP_EXTRA=0; echo "  [SERVE_FORM] 强制 embedding 形态（--runner pooling --convert embed）" ;;
  generate) SKIP_EXTRA=1; echo "  [SERVE_FORM] 强制生成形态（不传 --runner pooling）" ;;
  "")       : ;;
  *) echo "❌ 未知 SERVE_FORM=$SERVE_FORM（支持 embed | generate）"; exit 2 ;;
esac

ARGS=( "$MODEL" --served-model-name "$SERVED_NAME" --host "$HOST" --port "$PORT"
       --max-model-len "$MAX_MODEL_LEN" --tensor-parallel-size "$TP" )
# embedding 服务形态（三实例同口径）：--runner pooling --convert embed
[ "$SKIP_EXTRA" = "0" ] && ARGS+=( --runner pooling --convert embed )
[ -n "$GPU_MEM_UTIL" ] && ARGS+=( --gpu-memory-utilization "$GPU_MEM_UTIL" )
[ "$EAGER" = "1" ] && ARGS+=( --enforce-eager )
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=( ${EXTRA_ARGS} )

echo
echo "############ 启动 vllm serve ############"
if [ "$ERROR_TRANSLATION" = "1" ]; then
  # D10 集成（当前仅 910C 可用）：serve 以包装器启动，未捕获异常先过 translate_error
  nohup python3 -c "import inject_error_translation; inject_error_translation.serve()" serve "${ARGS[@]}" \
    > "$SRV_LOG" 2>&1 &
else
  nohup vllm serve "${ARGS[@]}" > "$SRV_LOG" 2>&1 &
fi
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "  pid=$SRV_PID  日志: $SRV_LOG"

# ─────────────────────────────────────────────────────────────────────────────
# 4) 就绪等待（判据：/v1/models 返回 200）
# ─────────────────────────────────────────────────────────────────────────────
READY=0
for i in $(seq 1 $((READY_BUDGET/5))); do
  sleep 5
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://$HOST:$PORT/v1/models" 2>/dev/null)
  if [ "$code" = "200" ]; then READY=1; echo "  ✅ 服务就绪 t=$((i*5))s"; break; fi
  kill -0 $SRV_PID 2>/dev/null || { echo "  ❌ 进程已退出（t=$((i*5))s）"; break; }
done
[ "$READY" = "0" ] && { echo "  ❌ 服务未在 ${READY_BUDGET}s 内就绪"; tail -20 "$SRV_LOG"; }

# ─────────────────────────────────────────────────────────────────────────────
# 4b) 功能冒烟（**两形态都有**）
#     "就绪"只证明端口通了、引擎起来了；冒烟才证明**服务真的能算**。
#     此前只有 embedding 形态有冒烟，生成形态（910C Qwen3-4B）只验了就绪
#     —— 2026-09-20 补齐，并纳入 verdict（PASS 要求 READY=1 且 SMOKE=1）。
# ─────────────────────────────────────────────────────────────────────────────
SMOKE=0
if [ "$READY" = "1" ]; then
  if [ "$SKIP_EXTRA" = "0" ]; then
    echo "--- 冒烟（embedding 形态）：/v1/embeddings ---"
    OUT=$(curl -s -m 60 -X POST "http://$HOST:$PORT/v1/embeddings" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$SERVED_NAME\",\"input\":[\"服务启动标准冒烟\"]}" 2>/dev/null) \
      && echo "$OUT" | python3 -c "import sys,json,math; d=json.load(sys.stdin); v=d['data'][0]['embedding']; print(f'  维度={len(v)} 范数={math.sqrt(sum(x*x for x in v)):.6f}')" 2>/dev/null \
      && SMOKE=1
  else
    echo "--- 冒烟（生成形态）：/v1/completions ---"
    OUT=$(curl -s -m 60 -X POST "http://$HOST:$PORT/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$SERVED_NAME\",\"prompt\":\"1+1=\",\"max_tokens\":8,\"temperature\":0}" 2>/dev/null) \
      && echo "$OUT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
t = (d['choices'][0].get('text') or '').strip()
n = d.get('usage', {}).get('completion_tokens', 0)
if not t or not n:
    raise SystemExit(1)
print(f'  生成 {n} tokens，首段={t[:40]!r}')
" 2>/dev/null \
      && SMOKE=1
  fi
  [ "$SMOKE" = "1" ] || { echo "  ❌ 冒烟失败（服务已就绪但功能不通，见服务日志与上方响应）"; echo "  raw: $(echo "$OUT" | head -c 300)"; }
else
  echo "--- 冒烟跳过（服务未就绪）---"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5) 收尾：验证模式则停机；否则保持运行（并给出下游需要的三项信息）
# ─────────────────────────────────────────────────────────────────────────────
if [ "$STOP_AFTER" = "1" ]; then
  echo
  echo "############ 验证模式：停机 ############"
  cleanup
  echo "--- 用卡后复查（确认已释放）---"; card_snapshot
  # verdict 要求"就绪 + 冒烟"双通过（只看端口会掩盖"起来了但算不出"的情况）
  if [ "$READY" = "1" ] && [ "$SMOKE" = "1" ]; then
    echo "[verdict] SERVE_STANDARD_PASS (ready=1 smoke=1)"
  else
    echo "[verdict] SERVE_STANDARD_FAIL (ready=$READY smoke=$SMOKE)"
  fi
else
  echo
  echo "############ 服务保持运行 ############"
  echo "  base_url   = http://$HOST:$PORT/v1"
  echo "  model      = $SERVED_NAME"
  echo "  停止服务   : STOP 时请用 cleanup 逻辑（kill -9 主进程 **与 EngineCore 子进程**），"
  echo "               或重跑本脚本并设 STOP_AFTER=1 以触发清理"
  echo "  pid        = $SRV_PID（$PIDFILE）"
fi
echo "=========== 结束 $(date) ==========="
