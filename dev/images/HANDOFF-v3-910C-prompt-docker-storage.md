# Prompt 1／2：新机器 Docker 存储位置迁移

> 用法：在新机器上开一个新的 Claude Code session，把下面「Prompt 正文」整段贴给它。
> 这是两件事里的第一件，建议先做——第二件事（局域网同步大文件）依赖这里腾出来的空间。

## Prompt 正文（直接复制粘贴）

```
这台机器 home 目录空间有限，docker 默认会把镜像/容器数据存到系统盘（通常是
/var/lib/docker），我要在这台机器上继续一个需要构建多个 ~30GB 级别 910C
镜像的任务，必须先把 docker 的存储位置从 home/系统盘搬到一个空间够大的盘。

请帮我做这件事，按下面的步骤，每一步先看清楚现状再动手，不要在没确认空间
安全的情况下删除任何现有数据：

1. 先摸清现状：
   - `df -h` 看所有挂载点的可用空间，找出哪个盘空间最大最合适（参考：另一台
     同类机器上是挂了一个 11TB 的 raid 在 /mnt/raid，docker root 指到了
     /mnt/raid/docker；这台机器不一定叫这个名字，需要实地确认，不要照抄路径）。
   - `docker info | grep "Docker Root Dir"` 看当前 docker 数据存在哪。
   - `du -sh <当前 docker root dir>` 看现在已经占了多少（可能是全新环境，
     也可能已经有一些镜像了）。
   - `docker ps -a` 和 `docker images` 看现在有没有在跑的容器、已有哪些镜像——
     这些如果有价值，之后要么留在原地要么原样搬过去，不能丢。
   - 检查现有 `/etc/docker/daemon.json`（如果存在）的完整内容——里面可能已经有
     registry mirror、insecure-registries 之类的配置，改的时候要在原文件基础上
     合并，不能整个覆盖掉。

2. 确认迁移方案后（选定目标路径，比如 `<大盘挂载点>/docker`），执行迁移：
   - 确认目标路径所在盘的可用空间明显大于当前 docker root 已占用空间。
   - 停 docker 服务前，如果 `docker ps` 里有其他人在用的容器，先跟我确认能不能停，
     不要unilaterally 停掉别人正在用的服务。
   - `systemctl stop docker`（或对应的服务管理方式）。
   - 如果当前 docker root 下已有数据需要保留：用 `rsync -a` 把旧目录内容搬到新
     目标路径（不要用 `mv` 跨盘，用 rsync 更安全，可断点续传，出问题不会丢原数据）；
     确认 rsync 完整跑完、抽查几个文件/目录大小对得上之后，再考虑要不要删除旧目录
     （删除前跟我确认一次，给我看确认结果）。
   - 编辑 `/etc/docker/daemon.json`，加入或更新 `"data-root": "<目标路径>"` 字段，
     保留原有其它字段不动。
   - `systemctl start docker`，`docker info | grep "Docker Root Dir"` 确认已经
     指向新路径。
   - `docker images` / `docker ps -a` 确认之前的镜像/容器（如果保留了）都还在。

3. 迁移完成后，简单汇报：新的 docker root 路径、迁移前后各自的可用空间、
   有没有旧数据被保留/删除、整个过程有没有遇到需要我决定的情况。

不要跳过第 1 步直接动手；如果这台机器没有明显的大容量盘（比如 df -h 里所有
挂载点可用空间都不大），先停下来告诉我实际情况，我们再决定方案，不要将就用
一个空间也紧张的盘。
```
