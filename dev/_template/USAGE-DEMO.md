# 使用指南：一次真实跑通的具体步骤

> 用真实存在的 v3 Route A 候选血统（`dev/images/ascend-operator-runtime/v3/`）做例子，
> 下面每一步的命令与输出都是实际跑出来的。规范定义见 `dev/ENV-SPEC.md`。

## 第 1 步：只校验，不启动

```bash
dev/lib/verify_env.sh dev/images/ascend-operator-runtime/v3/docker-compose.routeA.yml
```

实际输出：

```
[PASS] image: flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64 -> sha256:43f3e2f70b4ce49609bb703a2fa10fc9c5b1bfe71add5c965455ebe38562cbab
[PASS] FlagGems: commit=f7ae8e6b934a
[PASS] Torch-FL: commit=162582d678e4
[PASS] FlagCX: commit=4e0e0cbcbf72
[WARN] vllm-plugin-FL: 当前分支 xliu969/dev != 期望 release/0.2（loose，不阻断）
provenance -> dev/lib/.env_provenance.json
EXIT: 0
```

逐行解释：

- `image` 一行 `PASS`：本机 `docker image inspect` 的 digest 与 `pins.routeA.yaml` 里声明
  的一致。
- `FlagGems`/`Torch-FL`/`FlagCX` 三行 `PASS`：宿主 `FlagGems/`、`PyTorch-Plugin-FL/`、
  `FlagCX/` 三个检出的 `git rev-parse HEAD`，与 `pins.routeA.yaml` 声明的 `expect_commit`
  一致。
- `vllm-plugin-FL` 一行 `WARN`：这个仓库是 `class: loose`，宿主当前分支是
  `xliu969/dev`，与声明的 `release/0.2` 不一致——只警告，不影响退出码（`EXIT: 0`）。
- 最后一行 `provenance -> ...`：完整结果（含时间戳）写进了 `dev/lib/.env_provenance.json`
  （不入库，每次跑都会覆盖）。

## 第 2 步：校验通过后启动

```bash
dev/lib/up.sh dev/images/ascend-operator-runtime/v3/docker-compose.routeA.yml
```

`up.sh` 先跑第 1 步那个校验，通过（`EXIT: 0`）才继续执行
`docker compose -f dev/images/ascend-operator-runtime/v3/docker-compose.routeA.yml up -d`。
`docker-compose.routeA.yml` 是自包含的（设备节点、网络、公共仓挂载都在这一个文件里），
不需要额外叠加其他层就能独立启动。

## 第 3 步：故意制造一次 FAIL，看阻断效果

把 `dev/images/ascend-operator-runtime/v3/pins.routeA.yaml` 里 `FlagGems` 的
`expect_commit` 改成一个错误值再跑第 1 步的命令，会看到：

```
[FAIL] FlagGems: commit 不匹配: 期望 <改错的值>, 实际 f7ae8e6b934a
...
EXIT: 1
```

此时再跑 `up.sh`，会在校验阶段就退出，不会执行 `docker compose up`——这就是"总组发版、
子方向预研共用同一套校验，谁都绕不过去"的效果。改完记得把 `expect_commit` 改回来。

## 第 4 步：子方向叠加自己的探索层

假设 `communication` 方向想在 v3 Route A 基础上试一个不同的 `vllm-plugin-FL` 分支，新建
两个文件（同目录、同名后缀不同，见 `dev/ENV-SPEC.md` §2）：

`dev/communication/docker-compose.probe-vllmfl-branch.yml`：

```yaml
name: flagos-communication-probe-vllmfl-branch
services:
  runtime-dev:
    container_name: flagos-communication-probe-vllmfl-branch-dev-910c
```

`dev/communication/pins.probe-vllmfl-branch.yaml`：

```yaml
repos:
  vllm-plugin-FL:
    class: loose
    expect_branch: xliu969/ascend-sync-fix
```

然后：

```bash
dev/lib/up.sh dev/images/ascend-operator-runtime/v3/docker-compose.routeA.yml \
              dev/communication/docker-compose.yml \
              dev/communication/docker-compose.probe-vllmfl-branch.yml
```

`vllm-plugin-FL` 这一条会按最后一层（`probe-vllmfl-branch`）声明的
`expect_branch: xliu969/ascend-sync-fix` 校验，`FlagGems`/`Torch-FL`/`FlagCX`/`image` 仍然
沿用 `routeA` 层的声明，不受影响——这就是 §5 的字典递归合并规则在实际使用中的样子。

## 第 5 步：想知道各层之间某个库的 pin 是否一致

```bash
python3 dev/lib/list_pins.py vllm-plugin-FL FlagCX
```

会把所有 `pins.*.yaml` 里对这两个库的声明并排列出来，用于事后核对某个库在不同候选血统/
探索层之间到底 pin 没 pin 到同一个版本，不需要一个个打开文件去看。
