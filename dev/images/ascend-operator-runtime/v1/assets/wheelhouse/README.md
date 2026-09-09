# wheelhouse（36 个离线 wheel，~574M）

不入仓（体积，`.gitignore` 已挡）。逐文件 sha256 见 `../wheelhouse.sha256`。
本机实体位置见 `../../BACKUP.local`；离线包清单见 `../../ARCHIVE.md`。

无回收件时重建：`pip download -r ../requirements-runtime.txt -d .`（arm64 / py311，全 pin）。
`build.sh` 需把它放进构建上下文的 `wheelhouse/`。
