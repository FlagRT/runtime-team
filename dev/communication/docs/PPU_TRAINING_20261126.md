# 平头哥双卡训练通信与十一月阶段验收

复用公共设备原型 `ac9a9494` 和组内锁定 PPU 镜像。设备原型负责设备选择和同步；本方向新增身份门禁、集合通信数值检查及训练梯度和参数对照。不把设备自检、框架算子替换验证或监控采集算作双卡训练验收。PPU 使用 PyTorch `nccl` 接口下的 PCCL，不是 FlagCX。

| 验证 | 实测结果 |
| --- | --- |
| 单机双卡通信 | 两端矩阵乘、事件检查通过；FP32 AllReduce / AllGather 分别覆盖 8、2048 元素，每档 5 次，合计 40/40 项逐端检查，两端通信组正常关闭。 |
| DeepFM | 逐张量发送、合并后发送各 3 轮，每轮预热 5 步、训练 50 步，6/6 通过。每端 17/17 项梯度参考检查通过，最大误差 7.45e-9，两端最终参数差 0。每轮每端梯度通信由 850 次减至 50 次；不据此推断吞吐收益。 |
| DCN-Mix / SASRec | 两个最小模型各运行 50 步。每步与同批输入的独立 CPU 计算结果比较，梯度最大误差分别为 7.45e-8、1.49e-7，两端最终参数一致、启动器正常退出。 |

不同于同组的单算子或接口验证，这里检查数据分片、梯度合并、参数更新和通信组销毁的完整链路。合成数据只验功能正确性；未验证真实数据收敛、原生默认 DDP、跨机或规模性能。[脱敏结果](../results/20261126-ppu-training/summary.json)只保留环境版本、覆盖计数和数值判据。

## 复现

在组内锁定 PPU 镜像中只暴露已分配的两张 PPU，预先填好两个设备 UUID、空输出目录、与试验对应的设备原型提交；不改公共镜像或主机网络。以下均在仓库根目录执行，`OUT` 指向本次独立的可写证据目录：

```bash
set -eu
base=dev/communication
proto=dev/device-context/prototype
: "${OUT:?}" "${PROTOTYPE_REVISION:?}" "${PPU_UUID_0:?}" "${PPU_UUID_1:?}"
mkdir -p "$OUT/collectives"
export OMP_NUM_THREADS=1
free_port() {
  python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}
timeout 105s python3 -m torch.distributed.run --rdzv-backend=static --master-addr=127.0.0.1 \
  --master-port="$(free_port)" --nnodes=1 --nproc-per-node=2 \
  "$base/probes/ppu_two_rank.py" --runtime-root "$proto" \
  --expected-uuid "$PPU_UUID_0" --expected-uuid "$PPU_UUID_1" --output "$OUT/collectives"
timeout 600s python3 "$base/training/run_repeated.py" --backend ppu --collective nccl \
  --runtime-root "$proto" --prototype-revision "$PROTOTYPE_REVISION" \
  --model-root dev/model-defs/deepfm/v1 --output "$OUT/deepfm" --launch-timeout 120
python3 "$base/training/summarize_repeated.py" "$OUT/deepfm"
mkdir -p "$OUT/priority"
for model in dcn-mix sasrec; do
  mkdir -p "$OUT/priority/$model"
  rc=0
  timeout 120s python3 -m torch.distributed.run --rdzv-backend=static --master-addr=127.0.0.1 \
    --master-port="$(free_port)" --nnodes=1 --nproc-per-node=2 \
    "$base/training/run_priority_models.py" --backend ppu --runtime-root "$proto" \
    --revision "$PROTOTYPE_REVISION" --model "$model" --steps 50 \
    --output "$OUT/priority/$model" || rc=$?
  printf '%s\n' "$rc" > "$OUT/priority/$model/exit_code"
done
python3 "$base/training/check_priority_models.py" "$OUT/priority" --backend ppu --models dcn-mix sasrec
```

本次宿主业务网卡的本机 TCP 自连接超时；在同一主机内以 `NCCL_SOCKET_IFNAME=lo` 和回环 rendezvous 重跑后通过，卡间数据仍走 P2P/IPC。复现该分支时在运行命令前 `export NCCL_SOCKET_IFNAME=lo`；不得把这个设置用于跨机通信。

## 验收边界与后续

十一月四项计划可按约定的阶段范围收口：[昇腾双卡张量并行及两阶段流水最小原型](PARALLEL_MINIMAL_20261111.md)、昇腾及[昆仑芯 DeepFM 受控自动同步](P800_EXECUTION_STATE_ROOTCAUSE_20261111.md)，以及本次补齐的三芯片通信接入。昆仑芯本次卡对获准复位后 6/6 次单卡计算检查通过，但 FlagCX 与原生 XCCL 对照均没有通过首次 AllReduce（原生 8 元素 FP32 对照约 30 秒超时）。这不等于训练复验通过，也不足以认定 FlagCX 根因或提交修复。后续在健康卡对先重跑基础通信，再做训练；昇腾待合规双卡窗口回归第二批样例。

年内[九模型 × 三芯片矩阵](../training/priority_matrix_2026.json)已有 5 个模型取得至少一条双卡功能证据。十二月先验证 BGE-M3、LightGCN，再推进 CLIP、Qwen3-Reranker-8B；数据、权重与显存准备并行做。九个模型各至少一条双卡功能路径、三款芯片各至少一个代表性模型是阶段验收目标；矩阵保留未测和阻塞项。
