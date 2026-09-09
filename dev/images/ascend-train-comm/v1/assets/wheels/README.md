# flagcx wheel 缺口
构建时被 `rm`，镜像内已无。补齐方式（二选一）：
- 向镜像 owner（performance）索取 `flagcx-0.13.0-cp311-cp311-linux_aarch64.whl`，校验 `../flagcx-wheel.sha256`
- 按 `../lock.yaml` 的 commit `55eb2ff` + 两个 patch sha256 从 FlagRT/FlagCX 源码构建
