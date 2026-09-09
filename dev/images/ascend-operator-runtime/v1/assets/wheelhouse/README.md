# wheelhouse 缺口
离线 wheel 集（约 601MB），构建时被 `rm`，镜像内已无。
按 `../requirements-runtime.txt`（全 pin）重建：`pip download -r requirements-runtime.txt -d .`（arm64 / py311 环境）。
