#!/usr/bin/env sh
set -eu

ROOT=${ROOT:-/experiment/legacy_sdnn}
OUT=${OUT:-/experiment/results/acceptance}
MM=/env/FlagGems/src/flag_gems/runtime/backend/_kunlunxin/ops/mm.py

mkdir -p "$OUT" "$OUT/ir" "$OUT/cache"
cp -p "$MM" "$OUT/mm_original.py"
ORIGINAL_SHA=$(sha256sum "$MM" | awk '{print $1}')
restore() {
    cp -p "$OUT/mm_original.py" "$MM"
    test "$(sha256sum "$MM" | awk '{print $1}')" = "$ORIGINAL_SHA"
}
trap restore EXIT

patch --forward --batch "$MM" < "$ROOT/04-flaggems-mm-block-pointer.patch"
test "$(sha256sum "$MM" | awk '{print $1}')" != "$ORIGINAL_SHA"

export PYTHONPATH=/env/FlagGems/src${PYTHONPATH:+:$PYTHONPATH}
export FLAGTREE_BACKEND=xpu
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export KLX_USE_AUTOTUNE=0
export XMLIR_MATMUL_FAST_MODE=1
export XPU_SDNN_PRESERVE_TTXIR_ERRORS=1
export TRITON_CACHE_DIR="$OUT/cache"
export XPU_FIX_IR_DIR="$OUT/ir"

/root/miniconda/envs/python310_torch29_cuda/bin/python -B \
    "$ROOT/run_pytest.py" \
    --profile diagnostic \
    --output "$OUT/compiler_copy.py" \
    /env/FlagGems/tests/test_blas_ops.py::test_accuracy_mm -q -s
