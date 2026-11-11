#!/usr/bin/env sh
set -u

ROOT=${ROOT:-/experiment/legacy_sdnn}
OUT=${OUT:-/experiment/results/acceptance-blas-all}
PYTHON=${PYTHON:-/root/miniconda/envs/python310_torch29_cuda/bin/python}
MM=/env/FlagGems/src/flag_gems/runtime/backend/_kunlunxin/ops/mm.py
BMM=/env/FlagGems/src/flag_gems/runtime/backend/_kunlunxin/ops/bmm.py
ADDMM=/env/FlagGems/src/flag_gems/runtime/backend/_kunlunxin/ops/addmm.py

mkdir -p "$OUT/ir" "$OUT/cache"
for f in "$MM" "$BMM" "$ADDMM"; do
    cp -p "$f" "$OUT/$(basename "$f").original"
    sha256sum "$f"
done

restore() {
    cp -p "$OUT/mm.py.original" "$MM"
    cp -p "$OUT/bmm.py.original" "$BMM"
    cp -p "$OUT/addmm.py.original" "$ADDMM"
    test "$(sha256sum "$MM" | awk '{print $1}')" = "$(sha256sum "$OUT/mm.py.original" | awk '{print $1}')"
    test "$(sha256sum "$BMM" | awk '{print $1}')" = "$(sha256sum "$OUT/bmm.py.original" | awk '{print $1}')"
    test "$(sha256sum "$ADDMM" | awk '{print $1}')" = "$(sha256sum "$OUT/addmm.py.original" | awk '{print $1}')"
}
trap restore EXIT INT TERM

set -e
patch --forward --batch "$MM" < "$ROOT/04-flaggems-mm-block-pointer.patch"
patch --forward --batch "$BMM" < "$ROOT/05-flaggems-bmm-block-pointer.patch"
# addmm is intentionally composed through the fixed FP32 mm path. A direct
# block-pointer bias load is still kept as an unaccepted diagnostic variant in
# 05-flaggems-addmm-block-pointer.patch because this SDK rejects its masked
# bias load during TritonConvertType/bufferization.
patch --forward --batch "$ADDMM" < "$ROOT/06-flaggems-addmm-composition.patch"

export PYTHONPATH=/env/FlagGems/src${PYTHONPATH:+:$PYTHONPATH}
export FLAGTREE_BACKEND=${FLAGTREE_BACKEND:-xpu}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export KLX_USE_AUTOTUNE=${KLX_USE_AUTOTUNE:-0}
export XMLIR_MATMUL_FAST_MODE=${XMLIR_MATMUL_FAST_MODE:-1}
export XPU_SDNN_PRESERVE_TTXIR_ERRORS=${XPU_SDNN_PRESERVE_TTXIR_ERRORS:-1}
export TRITON_CACHE_DIR="$OUT/cache"
export XPU_FIX_IR_DIR="$OUT/ir"

set +e
"$PYTHON" -B "$ROOT/run_pytest.py" \
    --profile diagnostic \
    --output "$OUT/compiler_copy.py" \
    /env/FlagGems/tests/test_blas_ops.py::test_accuracy_mm \
    /env/FlagGems/tests/test_blas_ops.py::test_accuracy_bmm \
    /env/FlagGems/tests/test_blas_ops.py::test_accuracy_addmm \
    -q -s > "$OUT/run.log" 2>&1
RC=$?
cat "$OUT/run.log"
printf '%s\n' "$RC" > "$OUT/exit"
restore
trap - EXIT INT TERM
exit "$RC"
