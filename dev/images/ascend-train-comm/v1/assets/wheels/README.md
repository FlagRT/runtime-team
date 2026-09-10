# FlagCX 制品

- **重建用（已验证）**：从原镜像回收的「已装包 + `flagcx-0.13.0.dist-info`」直接 vendor，
  `.so` 与原镜像逐字节一致（校验 `../provenance/flagcx-vendor.sha256`）。
  本机实体位置见 `../../BACKUP.local`；清单见 `../../ARCHIVE.md`。
- **逐字节复现（未做）**：原始 `flagcx-0.13.0-cp311-cp311-linux_aarch64.whl`（sha256 见
  `../flagcx-wheel.sha256`）已被 `rm`；需 owner 提供，或按 commit `55eb2ff` + 2 patch
  （`../flagcx-source-artifacts.sha256` = 打补丁后源码树哈希）自建。
