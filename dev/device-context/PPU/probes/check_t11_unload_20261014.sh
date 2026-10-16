#!/bin/bash
# T11 取证 2：① LD_DEBUG=libs 看退出期卸载顺序（容器路径为准）② 找带调试器的镜像
set -u
HO=/bmcp_lvm_fs/hliu553/probes_ppu/dc_t11_20261014
O=/workspace/probes_ppu/dc_t11_20261014
P=/workspace/runtime-team/dev/device-context/prototype
C=hliu553-dc-dev

echo "===== [A] T4 载体 + LD_DEBUG=libs（3 次，看是否稳定复现）====="
for i in 1 2 3; do
    docker exec -e DC_BACKEND=ppu -e DC_ROOT=$P -e CUDA_VISIBLE_DEVICES=1 \
        -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONFAULTHANDLER=1 \
        -e LD_DEBUG=libs -e LD_DEBUG_OUTPUT=$O/ldd$i \
        "$C" bash -lc "cd $P && /usr/local/bin/python3 -u runtime/proto/proto_error_recovery_loop.py --backend ppu --out $O/t4_lddebug$i.json" \
        > "$HO/vehicle_t4_lddebug$i.log" 2>&1
    echo "  run$i rc=$?"
done

echo
echo "===== [B] 退出期 fini/unload 序列（最后一次运行）====="
F=$(ls -t $HO/ldd3.* 2>/dev/null | head -1)
echo "文件：$F"
if [ -n "${F:-}" ]; then
    echo "--- 最后 30 条 fini/destroy/unload ---"
    grep -aE "calling fini|destroying link map|unloading" "$F" | tail -30
    echo "--- libcuda / libcudart / libtorch_cuda 的全部 fini 行 ---"
    grep -aE "calling fini: .*(libcuda|libcudart|libtorch_cuda|libtorch_python|libpython)" "$F" | tail -12
fi

echo
echo "===== [C] 镜像里有没有调试器 ====="
for img in flagtree-ppu-compile-research:20261013-profile-experiment flagtree-ppu-repair-base:20260920 \
           flagtree-ppu-graph-compile-full:20260920 flagtree-ppu-xbench-runtime:20260920 \
           torchrec-dev-img:v3 cuml-ppu-img:v3 easyrec-ppu-img:v2 xgb-ppu-img:v2 \
           harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.24.0-thead-ci; do
    r=$(timeout 60 docker run --rm --entrypoint sh "$img" -c 'command -v gdb lldb eu-stack addr2line 2>/dev/null | tr "\n" " "' 2>/dev/null)
    echo "  ${img}  -> ${r:-（无）}"
done
