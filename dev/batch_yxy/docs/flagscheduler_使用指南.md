# FlagScheduler 使用指南

## 从源码开始

在 `batch_yxy/` 下执行 `python -B -m flagscheduler`。CLI/Mock 只需要 Python ≥3.11。`pyproject.toml` 已定义安装后的同名 `flagscheduler` 入口，但当前不向外部 Conda 环境安装包或修改其文件。

```bash
python -B -m flagscheduler --help
python -B -m flagscheduler --version
python -B -m flagscheduler discover
python -B -m flagscheduler models
python -B -m flagscheduler doctor
```

`doctor` 只读取 NVIDIA 清单。沙箱中 `nvidia-smi` 可能被限制；应在实际节点权限下复核。可发现框架路径不表示已经验证其运行条件或空闲设备。

## 检查当前 Conda vLLM

```bash
python -B -m flagscheduler scheduler inspect \
  --python /home/ldaphome/yuanxianyi/miniconda3/envs/vllm/bin/python
```

其他环境用 `discover --python /path/to/python`；默认根目录之外的环境集合用 `discover --env-root /path/to/envs`。结果包含每个环境的 Python、框架版本/路径、V0/V1/异步候选、扩展字段及源码指纹。目标环境无法读取/启动时返回该环境错误，继续显示其他环境。

## Scheduler 替换计划与原生回退

### 保留 Default 并选择 Currently

查看可选脚本：

```bash
python -B -m flagscheduler scheduler list \
  --python /home/ldaphome/yuanxianyi/miniconda3/envs/vllm/bin/python
```

初次显示：

```text
default (Default) (Currently)
  Class: vllm.v1.core.sched.scheduler.Scheduler
  Script: .../vllm/v1/core/sched/scheduler.py
  Preserved: .../batch_yxy/.state/schedulers/originals/<sha256>.py

flagscheduler
  Class: flagscheduler.integrations.vllm_scheduler.FlagScheduler
  Script: .../batch_yxy/flagscheduler/integrations/vllm_scheduler.py
```

选择项目脚本或恢复默认：

```bash
python -B -m flagscheduler scheduler select flagscheduler \
  --python /home/ldaphome/yuanxianyi/miniconda3/envs/vllm/bin/python
python -B -m flagscheduler scheduler select default \
  --python /home/ldaphome/yuanxianyi/miniconda3/envs/vllm/bin/python
```

选中项目脚本后，原生行保留 `(Default)`，项目行显示 `(Currently)`；选回默认时原生行同时显示两个标记。目录记录的是供后续启动使用的持久选择，不探测或热切换已运行进程。首次列举时保存所有发现的原生脚本快照，原文件内容和位置保持不变；快照不作为脱离框架依赖的独立可执行脚本使用。

省略 `scheduler plan` / `launch` 的 `--scheduler-class` 时使用该选择。`vllm-embed` 节点未显式配置 `llm_options.scheduler_cls` 时也读取同一环境的选择；`status` 中各 binding 的 scheduler 标注则表示启动时配置。显式 `--scheduler-class` 或 `llm_options.scheduler_cls` 是单次/单 binding 覆盖，不改目录里的持久选择。

同步/异步目录分别管理；查看或选择异步脚本加 `--scheduler-mode async`，启动节点时相应设置 `llm_options.async_scheduling: true`。同一环境的 Python 可执行文件别名共享选择，不同环境或安装源码版本分别保存。框架升级后新目录从 Default 开始，旧快照与记录仍保留，不自动沿用旧自定义类。

命令加 `--json` 可供程序/TUI 读取结构化的 `default/current/labels`、源文件、快照和完整性状态。默认状态目录是本项目 `.state/schedulers/`，可用 `--state-dir` 指定本目录内的另一个位置；in-process 后端可在 options 中设置 `scheduler_state_dir` 使用相同目录。

注册自己的脚本（文件与 Python 模块名应对应，不会在注册时执行脚本）：

```bash
python -B -m flagscheduler scheduler add my-policy \
  --python /path/to/conda/envs/vllm/bin/python \
  --script /path/to/batch_yxy/custom_schedulers/my_policy.py \
  --class custom_schedulers.my_policy.MyScheduler --expect-version 0.21.0
python -B -m flagscheduler scheduler select my-policy --python /path/to/conda/envs/vllm/bin/python
```

脚本修改后需要重新 `select` 确认当前内容，保留旧快照；未确认的变化会阻止按旧选择启动。Default 和项目内置脚本 ID 不可被注册覆盖。完整框架接口校验仍在 `scheduler validate` 或显式执行启动时进行。

### 生成、校验和启动

当前示例插件继承 vLLM 原生行为，作为自定义算法入口。同步类为 `flagscheduler.integrations.vllm_scheduler.FlagScheduler`，异步类为 `FlagAsyncScheduler`；二者当前只接受 vLLM 0.21.0。

先生成静态计划，不会加载模型或启动推理服务：

```bash
mkdir -p artifacts
python -B -m flagscheduler scheduler plan \
  --python /home/ldaphome/yuanxianyi/miniconda3/envs/vllm/bin/python \
  --model /path/to/local/Qwen3-Embedding-0.6B \
  --scheduler-class flagscheduler.integrations.vllm_scheduler.FlagScheduler \
  --scheduler-mode sync --expect-version 0.21.0 \
  --arg=--runner --arg=pooling \
  --output artifacts/qwen-vllm-plan.json
```

输出文件默认不覆盖。计划中的 `argv` 是使用自定义类的启动方式，`native_argv` 是原生回退。自定义插件额外目录用 `--plugin-path` 指定。源码运行会将本项目根目录传给目标 Python；以后从已安装 CLI 操作另一个环境时，插件必须在目标环境可用或通过显式源码路径提供，不能把 CLI 所在的整个 site-packages 混入目标环境。插件 import 就会执行 Python 代码，只使用已审查插件。

只校验可导入性、继承与构造签名，不创建模型：

```bash
python -B -m flagscheduler scheduler validate \
  --python /home/ldaphome/yuanxianyi/miniconda3/envs/vllm/bin/python \
  --class flagscheduler.integrations.vllm_scheduler.FlagScheduler
```

在确认指定 GPU 空闲、模型路径和启动参数正确后，显式启动新进程：

```bash
CUDA_VISIBLE_DEVICES=0 python -B -m flagscheduler launch \
  --plan artifacts/qwen-vllm-plan.json --execute
# 原生回退；这是另一个前台进程，先管理原有服务的生命周期和端口。
CUDA_VISIBLE_DEVICES=0 python -B -m flagscheduler launch \
  --plan artifacts/qwen-vllm-plan.json --native --execute
```

不加 `--execute` 只展示计划。启动保留 `CUDA_VISIBLE_DEVICES`，不自动挑卡、不触碰运行中的服务。外部 scheduler 源码不被覆盖；计划执行前复核版本/路径/源码指纹，并检查 argv 与声明设置一致。框架原生日志和进程退出码由该前台进程返回。

## 统一请求入口

```bash
python -B -m flagscheduler config examples/flagscheduler.mock.json
python -B -m flagscheduler run \
  --config examples/flagscheduler.mock.json \
  --input examples/flagscheduler.requests.jsonl
```

`run` 接受文件或 `--input -`，逐行接纳，按完成顺序逐行输出带 ID 的 JSON。文件输入在达到容量时等待空位；单行非法输入会输出 rejected 并继续，已有请求仍会排空。该模式用于离线文件/管道，常驻服务使用 `serve`。

```bash
python -B -m flagscheduler serve --config examples/flagscheduler.mock.json --port 8765
# 另一终端
python -B -m flagscheduler status
python -B -m flagscheduler submit --request examples/flagscheduler.request.json
```

初版 API 仅绑定回环地址，面向可信本机进程，不含身份认证。不要直接暴露到公网。HTTP/1.1 每连接单请求后关闭；请求体上限 1 MiB，头部有长度/读取超时限制，不接受 chunked body。

| API | 语义 |
| --- | --- |
| `GET /healthz` | 进程状态 |
| `GET /v1/status` | 队列、binding、完成/失败数及最近批次，不返回原始 payload |
| `POST /v1/infer` | 提交一个 Request，等待终态结果 |
| `GET /v1/results/{request_id}` | 查询活动或缓存结果；未知/已淘汰返回 404 |

Request 结构：

```json
{
  "request_id": "unique-business-id",
  "binding": "qwen-embed",
  "payload": "待推理文本",
  "priority": 1,
  "slo_ms": 100,
  "metadata": {}
}
```

`priority` 数值越大越先；`slo_ms` 当前仅记录目标，不保证完成时间。结构模型的 payload 是由后端定义的对象。默认 mock 的结构成本字段必须提供相应 `sequence_length` / `nnz` / `sample_count`；这只是夹具协议，不是生产特征校验器。

CLI 退出码：0 成功；1 表示请求存在失败/拒绝、HTTP 错误或 NVIDIA doctor 不就绪；2 为参数/配置/控制面执行错误；130 为中断。HTTP 400 为非法输入或重复 ID，429 为节点容量已满，502 为后端本批执行失败。

## NVIDIA 真实 Embedding 验收

`examples/flagscheduler.nvidia.json` 是模板，先复制到本目录 `artifacts/` 再填入真实本地模型路径、权重版本和预算。模板中的预算不是 RTX A6000 极限，也不是性能推荐值。

在有 vLLM 的 Python 中从本目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 /home/ldaphome/yuanxianyi/miniconda3/envs/vllm/bin/python -B -m flagscheduler serve \
  --config artifacts/flagscheduler.nvidia.local.json
```

`vllm-embed` 会按配置创建自己的 LLM 实例，与 `launch` 启动的独立 vLLM 服务是两种入口。不要同时对同一模型/同一卡重复启动；本阶段没有实现把 runtime 自动连接到 `launch` 服务的远程 Embeddings 适配器。

后端使用执行入口同一 tokenizer，超长显式拒绝、不自动截断。默认使用当前环境目录里保存的 scheduler；也可在本地配置的 `llm_options.scheduler_cls` 显式覆盖并设置同级 `expect_version` 为精确版本，同步/异步选项也需一致。该配置是可信执行配置，不能接受外部用户任意提交。

验收至少覆盖：32 条不同长度、2 档优先级、不同到达序列；逐 ID 向量/数量/维度与直接调用对照；低流量等待、满批、超长、过载、退出排空；原生与自定义 scheduler 的同权重/预热/设备对照。记录时延与吞吐、显存、批次组成、框架版本、权重版本、GPU 型号及观察到的真实设备执行信息。不要以 mock 结果代替这些证据。

## 验证与产物

在现有测试环境执行 `bash scripts/test_flagscheduler.sh`，可用 `FLAGSCHEDULER_TEST_PYTHON` 指定解释器，不会安装依赖。使用标准库接口的 CLI/Mock 可以与 GPU 测试独立运行。当前环境的测试可在 CPU 隔离容器中读取已有 pytest/numpy 依赖；测试缓存与 `--basetemp` 指向本目录 `.tmp/`，不要修改其他目录。

`artifacts/`、`.tmp/` 均忽略入库。框架校验/启动的缓存默认放当前工作目录 `.tmp/`，可用 `FLAGSCHEDULER_STATE_DIR` 显式指定；当前开发约束要求它仍处于 `batch_yxy/` 内。源码包 `sched_engine/models/` 通过本目录的 `.gitignore` 例外保留为可跟踪 Python 源码；没有修改仓库根忽略规则。原始 benchmark 与 probe 证据继续保留。
