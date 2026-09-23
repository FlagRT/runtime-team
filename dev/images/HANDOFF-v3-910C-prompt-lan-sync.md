# Prompt 2／2：从旧机器局域网同步可复用大文件

> 用法：在新机器上开一个新的 Claude Code session（可以和 Prompt 1 是同一个 session，
> 建议先做完 Prompt 1 腾出空间再做这个），把下面「Prompt 正文」整段贴给它。
> 目的：910C v3 这个任务里有几个大文件（docker 镜像、模型权重）在旧机器上已经
> 构建/下载好了，走局域网直接同步比在新机器上重新构建/重新下载快得多。

## 背景信息（正文里已经带了，供你核对）

- 旧机器：`npu1-27`，局域网 IP 候选（哪个通就用哪个，不确定就都 ping 一下）：
  `10.120.73.79` / `10.120.73.80` / `10.120.72.27`，用户名 `xliu969`。
- 交接文档在旧机器的 `~/runtime-team` 仓库（`xliu969/dev` 分支）：
  `dev/images/HANDOFF-v3-910C.md`。

## Prompt 正文（直接复制粘贴）

```
我在续接一个从另一台机器（局域网内，主机名 npu1-27，IP 可能是 10.120.73.79 /
10.120.73.80 / 10.120.72.27 中的一个，用户名 xliu969）交接过来的任务
（详见本机 runtime-team 仓库 xliu969/dev 分支的 dev/images/HANDOFF-v3-910C.md，
先读一下这个文件了解全貌）。那边已经构建/下载好几个大文件，局域网同步比在这台
机器上重新构建/重新下载快很多，帮我把它们同步过来。

前提：docker 的存储位置已经迁移到大容量盘了（如果还没做，先做那件事，
见 dev/images/HANDOFF-v3-910C-prompt-docker-storage.md）。

请按下面顺序做，每一步先验证连通性/权限，不要假设一定能连上或有权限：

1. 连通性确认：
   - 依次 ping 10.120.73.79 / 10.120.73.80 / 10.120.72.27，找出这台机器能到达
     哪个（局域网内可能走不同网段，不一定三个都通）。
   - `ssh xliu969@<通的那个IP> echo ok` 确认 SSH 免密或至少能连上；如果连不上
     （没有 key、需要密码），停下来问我要怎么处理，不要尝试绕过认证。

2. 同步两个 v3 镜像（旧机器上已构建好，直接 docker save | ssh | docker load
   流式传输，不落盘中间 tar 文件，省两边磁盘）：
   - 先 `ssh xliu969@<IP> docker images` 确认下面这两个 tag 是否还在（旧机器上
     的任务还在推进，tag 可能已经变化，不要假设一定还是这两个）：
     - `flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64`
     - `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`
   - 确认存在后：
     ```
     ssh xliu969@<IP> 'docker save flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64 flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64' | docker load
     ```
     这两个 tag 一起传，第二个是从第一个 build 出来的，公共 layer 会自动去重，
     总量大概 31GB 左右（不是两个 31GB 相加）。局域网内预计比重新跑一遍
     build.sh（要下载基座镜像 + 编译 triton + 装一堆东西，很慢）快得多。
   - 传完 `docker images` 核对 image id 是不是 `9ad551058f2f`（operator-runtime）
     和 `43f3e2f70b4c`（comm）——如果对不上说明旧机器那边镜像已经更新了，
     以旧机器当前实际的 id 为准，不要死板对照这两个旧 id。

3.（可选，先判断要不要）基座镜像：
   `harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64:202608-torch2.10.0-vllm0.20.2`
   （19.2GB）。先试 `docker pull` 这个地址，如果这台机器访问 harbor.baai.ac.cn
   速度正常，就不用走局域网搬了；只有确认这条路径慢/不通的时候，才用类似第 2
   步的 `docker save | ssh | docker load` 从旧机器搬这个。

4. 同步模型文件（旧机器上路径是 `/mnt/raid/hliu553/models/`，注意这个目录
   属主是 hliu553 不是 xliu969，用 xliu969 账号能不能读要先试一下）：
   - `ssh xliu969@<IP> 'ls -la /mnt/raid/hliu553/models/Qwen2.5-1.5B /mnt/raid/hliu553/models/Qwen3-Embedding-0.6B'`
     确认能读到（如果权限不够，跟我说，这两个模型也可以从 modelscope/huggingface
     重新下载，不是非局域网同步不可，只是局域网更快）。
   - 能读的话：
     ```
     rsync -avP --info=progress2 xliu969@<IP>:/mnt/raid/hliu553/models/Qwen2.5-1.5B/ <这台机器上的目标路径>/Qwen2.5-1.5B/
     rsync -avP --info=progress2 xliu969@<IP>:/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B/ <这台机器上的目标路径>/Qwen3-Embedding-0.6B/
     ```
     两个加起来大概 4GB（Qwen2.5-1.5B 约 2.9GB，Qwen3-Embedding-0.6B 约 1.2GB）。
     目标路径放哪你自己定（这台机器的大容量盘上即可），记下来，下一步要用。

5. 私有 fork（`FlagRT/vllm-plugin-FL`，含关键 sync 修复 commit `5d545c9`）
   **不用走局域网同步**，只有 6.9MB，直接重新 clone 更简单：
   ```
   git clone git@github.com:FlagRT/vllm-plugin-FL.git
   cd vllm-plugin-FL
   git remote add flagos-upstream https://github.com/flagos-ai/vllm-plugin-FL.git
   git log --oneline -5   # 确认能看到 5d545c9 / f34b4e7 这两个 commit
   ```

6. 收尾：模型文件落地后，回到 runtime-team 仓库，把
   `dev/images/v3-pending-validation/race_and_validate.sh`、
   `dev/images/v3-pending-validation/race_task1.sh`、
   `dev/images/v3-pending-validation/v3_step5_validate.py` 里写死的
   `/mnt/raid/hliu553/models/...` 路径改成这台机器上第 4 步实际落地的路径
   （这几个文件里搜 `/mnt/raid/hliu553` 就能定位到全部要改的地方）。

7. 最后汇报：哪些同步成功了、哪些走了 docker pull 而不是局域网、模型文件
   落地在哪、脚本路径改了没有、总共花了多少时间——方便我知道下一步能不能
   直接开始真机验证。
```
