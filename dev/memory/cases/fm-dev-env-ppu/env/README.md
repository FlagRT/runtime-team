# env 引用方式

本样例**自建**了一个派生运行环境目录（ENV-SPEC §3.2「各方向自建」形态）。

## 派生声明（对齐的具体版本锚）

| 项 | 母本（对齐对象） | 本环境 |
|---|---|---|
| 环境 | `dev/memory/cases/tiering-async-ppu/env/`（FM 在 PPU 的现行定档环境） | 本目录 |
| 镜像 | flagrelease_ppu_vllm024_gems_540:1012（id `6cd05dcdcf30…`，digest `399937f1…`） | 同母本（image.anchor.yaml 同锚） |
| 容器骨架 | network host / 全 16 卡映射 / SDK 三件套 / entrypoint 装配 | 逐项同母本 |
| **FlagMemory** | `58e60a9`（v0.3.0 发版 commit，母本 env.baseline 锁定） | **`3b7cdb1`（FM dev-1.0 头，含 fix-ppu-copyin-test 修复）** |

即：**唯一差异 = FM 挂载锚**。母本守发版 commit（现行环境不因开发线漂移），本环境锁
FM dev-1.0 线头（开发验证跟线走）——`up.sh --strict` 对两者各自守门。

## 自建三步（本目录怎么来的）

1. 拷贝母本 compose，改 name/container_name 与差异项；
2. 镜像登记桩：同镜像则拷贝母本 image.anchor.yaml；
3. 宿主检出停到目标 commit 后导出自己的版本事实：
   `python3 dev/lib/baseline.py export dev/memory/cases/fm-dev-env-ppu/env`

## 跟线维护

FM dev-1.0 推进后：宿主检出 ff → 重跑第 3 步重导 baseline → 更新 run.sh 的
`FM_COMMIT_EXPECT` → REPORT 注明当前锚。baseline 的 FM commit 即本环境与开发线的绑定。

## 收尾

`WORKSPACE_ROOT=$PWD docker compose -f dev/memory/cases/fm-dev-env-ppu/env/docker-compose.yml down`
