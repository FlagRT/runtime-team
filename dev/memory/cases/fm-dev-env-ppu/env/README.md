# env 自建派生方式（教学注解版）

本样例**自建**了一个运行环境目录（不像样例①②那样引用现行环境）——这是三部曲的第三种
方式，适用场景：**你要的环境参数与现行环境不同**。本件的「不同」= FlagMemory 挂载锚。

## 派生声明（自建的第一句该写的话）

> 对齐母本 [`dev/memory/cases/tiering-async-ppu/env/`](../../tiering-async-ppu/env/)
> （FM 在 PPU 的现行定档环境）：镜像（flagrelease vllm0.24 定档）、SDK 三件套、设备映射、
> 容器骨架逐项一致；**唯一差异 = FlagMemory 挂载锚从 v0.3.0 发版 commit（58e60a9）切到
> 开发分支 `xliu969/fix-ppu-copyin-test`（08af9f3）**。

## 为什么必须自建（而不是引用/改现行环境）

- 现行环境（母本）的 `up.sh --strict` 守门在 FM 发版 commit——**开发分支的检出会被拒绝
  启动**，这是设计（现行环境不因个人开发线漂移）；
- 改母本的锚 = 把个人开发线混进公共定档环境，影响所有使用方；
- 自建派生 = 开发者自己的环境、自己的 baseline、**自己的 FM commit 守门**——变更可见、
  机器核对、与现行环境互不干扰。

## 自建的三步（本目录怎么来的）

```bash
# ① 拷贝母本 compose，改派生声明头 + name/container_name + 差异项（本例差异在挂载锚）
# ② 镜像登记桩：同镜像则拷贝母本的 image.anchor.yaml（或指向同镜像的既有锚）
# ③ 宿主检出停到开发分支目标 commit 后，导出自己的版本事实：
cd <仓根>
python3 dev/lib/baseline.py export dev/memory/cases/fm-dev-env-ppu/env
#    → 生成 env.baseline.yaml（FM 的 mounted_repos.commit = 当前检出 HEAD，strict 守此门）
```

## 跟开发线走（维护口径）

开发分支推进后：宿主检出 ff 到新头 → 重跑上面第 ③ 步重导 baseline → strict 门随线头
更新。**baseline 的 FM commit 就是本环境与开发线的绑定关系**，REPORT 里注明当前锚。

## 与前两件样例的关系（三部曲全景）

| 样例 | 环境方式 | 变更登记在哪 | 适合谁 |
|---|---|---|---|
| ① fm-smoke-ascend | 纯引用现行环境 | 现行环境的 env.baseline | 使用方（环境刚好够用） |
| ② fm-wheel-install-ppu | 引用 + run.sh 临时 pip 装 | run.sh 头部声明 + wheel sha256 自证 | 消费方临时验证交付组件 |
| ③ 本样例 | **自建派生**（对齐声明 + 差异清单） | **自己的 env.baseline.yaml** | **开发者跟开发线** |
