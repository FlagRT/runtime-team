# 新机器迁移清单与两仓同步纪律

## 一、任何平台新机器的通用步骤

```bash
mkdir -p ~/fgw && cd ~/fgw
# 1) 交付仓（代码）
git clone git@github.com:FlagRT/FlagScheduler.git
# 2) 信息仓（本分支；只需要 dev/batch_yxy/ 目录时用 sparse）
git clone -b xianyiyuan/batch_yxy --single-branch --filter=blob:none \
  git@github.com:FlagRT/runtime-team.git
cd runtime-team && git sparse-checkout set dev/batch_yxy
# 3) 昆仑芯机器另需：插件修复版
git clone -b dev-1.0 --single-branch git@github.com:FlagRT/vllm-plugin-FL.git ~/fgw/vllm-plugin-FL
```

SSH key：`ssh-keygen -t ed25519` → 公钥加到 GitHub → `~/.ssh/config` 配
`github.com`（22）与 `github-443`（ssh.github.com:443 备用）。

## 二、按平台的机器侧准备

| 平台 | 必备项 |
| --- | --- |
| 昆仑芯 P800 | flagos `tele4` 镜像（digest `e180a0dd…`）；配额卡约定（P800-A=宿主 2/3/7）；容器配方见 `platforms/kunlunxin-p800/README.md`；模型符号链接视图见下节 |
| 昇腾 910C | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`；容器名/挂载沿用历史口径（见 `platforms/ascend-910c/README.md`） |
| NVIDIA A6000 | conda `vllm` 环境（0.21.0），见 `platforms/nvidia-a6000/README.md` |

## 三、P800 模型符号链接视图（剔除 st 配置，规避 vllm 本地路径误判 bug）

```bash
mkdir -p ~/fgw/models/qwen3-embed-0.6b-xpu
cd /data1/flagperf-kzhang519/models/Qwen3-Embedding-0.6B-modelscope   # 按新机器实际路径
for f in config.json generation_config.json merges.txt model.safetensors \
         tokenizer.json tokenizer_config.json vocab.json; do
  ln -sf "$PWD/$f" ~/fgw/models/qwen3-embed-0.6b-xpu/"$f"
done
# 不链接：config_sentence_transformers.json / modules.json / configuration.json
```

注意：容器需把权重源目录**按宿主同路径**挂载，符号链接才能在容器内解析。

## 四、两仓同步纪律

1. **数据/进度/平台事实** → 本分支 `platforms/<平台>/`（progress.md 记进度、
   data/ 存结果 JSON、investigation/ 存排查与报告、raw/ 存脚本与原始日志）
2. **代码/测试/文档交付** → FlagScheduler（main），交付基座对齐
   dev-1.0 `dev/images/USAGE_MATRIX.md`
3. FlagScheduler 仓的 `reports/`、`*/results/` 已 gitignore 防回流——跑出的
   新数据先落本地，整理后拷入本分支对应平台目录提交
4. commit 信息注明平台（如 `docs(p800): …`），便于跨机器追溯
