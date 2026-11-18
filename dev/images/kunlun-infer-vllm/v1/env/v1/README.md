# env/v1 —— 昆仑芯推理腿第一版运行环境（up-sh）

- **是什么**：kunlun-infer-vllm v1 镜像（ADR-0026 定档，tele4）的首个运行环境目录——
  单卡授予 + `/dev/xpuctrl` + FlagMemory 只读挂载（PYTHONPATH 生效，RUNBOOK §1 口径）+
  `network none`。定位 = **KV connector 判据与样例运行环境**（验收样例
  [`dev/memory/cases/connector-kunlunxin/`](././././memory/cases/connector-kunlunxin/)），
  不是服务化环境（`vllm serve` 服务化口径见 xpu RUNBOOK §1 与 serve_standard 线）。
- **形态出处**：FM 仓 `flagmemory/tiering/offload/xpu/RUNBOOK.md` 与
  `tests/tiering/test_xpu_tier_manager_realcard.py` 文件头的 2026-11-09 已验证形态；
  两处派生差异（memlock ulimit、仓根整体挂载）的出处见 docker-compose.yml 头注。
- **版本事实**：本目录 `env.baseline.yaml`（`dev/lib/baseline.py export` 生成）；
  镜像事实承自 [`././image.baseline.yaml`](././image.baseline.yaml)（digest 主锚）。
  挂载锚：`mounted_repos.FlagMemory.commit` = v0.3.0 发版提交（58e60a9）。
- **用法**：`xpu-smi` 挑空闲卡（避开卡 1/卡 2 与有残留占用者）→
  `XPU_CARD=<n> dev/lib/up.sh dev/images/kunlun-infer-vllm/v1/env/v1 --strict` →
  容器 `flagrt-kunlun-infer-v1-env-v1-p800a` 内 0 号卡 = 宿主 `<n>` 号卡。收尾 down 命令
  见 compose 头注。
- **已知环境事实**（2026-11-15 P800-A 实测，探查留痕 `/tmp/s4-kunlun-infer-20261115/`）：
  - 勿把仓根（或任何含 vllm-plugin-FL 的目录）挂到 /workspace：镜像 /workspace 自带
    vllm-plugin-FL 等源树（其 VENDOR_DEVICE_MAP 含 kunlunxin），被仓内陈旧副本遮蔽后 import 链断在
    `Vendor kunlunxin not found`（二分实测：整仓挂 /workspace 必挂、只挂 FlagMemory 必过）；
  - FlagMemory 以 PYTHONPATH 生效（RUNBOOK §1）；容器内 `python` 落 miniconda base
    （PATH 不含 conda env bin，无 pytest/torch），负载一律用
    `/root/miniconda/envs/python310_torch29_cuda/bin/python`；
  - 镜像 conda env 自带 pytest，判据运行无需网络、无需 pip。环境本体验证（`up.sh --strict` 两轮全绿 + FM 0.3.0 导入）
    2026-11-15 通过；同日判据三件 2/5（机器级 D2H 拷贝路径问题，纯 torch 探针亦挂，与本环境无关），
    详见样例二 ../../../../memory/cases/connector-kunlunxin/REPORT.md。

<!-- changelog
2026-11-15  新建（S4-B）：推理腿由 registered-only 转首个 up-sh 运行环境。同日真机：--strict 两轮全绿、
            FM 0.3.0 导入绿；判据三件 2/5（机器级 D2H 拷贝路径问题，待恢复后重跑）。
-->
