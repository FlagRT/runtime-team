#!/usr/bin/env bash
# connector-ppu 容器侧执行体入口：source SDK envsetup 后跑字节级探针。
# 口径同 dev/memory/cases/tiering-async-ppu/scripts/run_tiering_ppu.sh 的 SDK 三件套。
set -eo pipefail
cd /workspace/FlagMemory
source /usr/local/PPU_SDK/envsetup.sh
set -u
export UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD
python3 /tmp/ppu_offload_byte_probe.py
