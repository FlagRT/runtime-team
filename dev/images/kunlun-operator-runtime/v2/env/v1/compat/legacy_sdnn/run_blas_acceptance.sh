#!/usr/bin/env sh
# Run the direct ADDMM kernel, not 06-flaggems-addmm-composition.patch.
set -eu
ROOT=${ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)}
OUT=${OUT:-/experiment/results/direct-blas-$(date +%Y%m%d-%H%M%S)}
OPS=${OPS:-/env/FlagGems/src/flag_gems/runtime/backend/_kunlunxin/ops}
PYTHON=${PYTHON:-/root/miniconda/envs/python310_torch29_cuda/bin/python}
ADDMM_PATCH=${ADDMM_PATCH:-$ROOT/05-flaggems-addmm-block-pointer.patch}

# A fresh directory protects original backups and makes cold-cache runs explicit.
mkdir "$OUT"
mkdir "$OUT/ir" "$OUT/cache"
EXPECTED_MM=848afc83d2a541b88c29fa66ab3396a60d4c73838dac0352d866d31dcc94c33b
EXPECTED_BMM=7cff38c02136e5e4978d3c0ad796fc07ad1b73a8ad98b73b0cce61a94e333171
EXPECTED_ADDMM=24cf49b9b042e6eb9a97ec7bc4dd3637b1a805f99b5ca10d6e1dc15cd18f5c1f
check_hash() {
    test "$(sha256sum "$1" | awk '{print $1}')" = "$2"
}
check_hash "$OPS/mm.py" "$EXPECTED_MM"
check_hash "$OPS/bmm.py" "$EXPECTED_BMM"
check_hash "$OPS/addmm.py" "$EXPECTED_ADDMM"
for name in mm bmm addmm; do
    cp -p "$OPS/$name.py" "$OUT/$name.py.original"
done
restore() {
    rc=$?
    trap - EXIT
    for name in mm bmm addmm; do
        if ! cp -p "$OUT/$name.py.original" "$OPS/$name.py"; then rc=125; fi
        if ! cmp -s "$OPS/$name.py" "$OUT/$name.py.original"; then rc=125; fi
    done
    sha256sum "$OPS/mm.py" "$OPS/bmm.py" "$OPS/addmm.py" > "$OUT/restored.sha256"
    printf '%s\n' "$rc" > "$OUT/exit"
    exit "$rc"
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 04 is an existing historical hunk (fuzz 1); ADDMM uses exact hunk matching.
patch --forward --batch "$OPS/mm.py" < "$ROOT/04-flaggems-mm-block-pointer.patch"
patch --forward --batch --fuzz=0 "$OPS/bmm.py" < "$ROOT/05-flaggems-bmm-block-pointer.patch"
patch --forward --batch --fuzz=0 "$OPS/addmm.py" < "$ADDMM_PATCH"
for name in mm bmm addmm; do
    cp -p "$OPS/$name.py" "$OUT/$name.py.modified"
done
sha256sum "$OPS/mm.py" "$OPS/bmm.py" "$OPS/addmm.py" "$ADDMM_PATCH" > "$OUT/modified.sha256"
export PYTHONPATH=/env/FlagGems/src
export PYTHONDONTWRITEBYTECODE=1
export FLAGTREE_BACKEND=xpu
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export KLX_USE_AUTOTUNE=0
export XMLIR_MATMUL_FAST_MODE=1
export XPU_SDNN_PRESERVE_TTXIR_ERRORS=1
# Keep the default Pipeline. This is not the diagnostic Pass-skipping variant.
export XPU_SDNN_SKIP_EARLY_COMBINE=0
export TRITON_CACHE_DIR="$OUT/cache"
export XPU_FIX_IR_DIR="$OUT/ir"
if [ "$#" -eq 0 ]; then
    set -- /env/FlagGems/tests/test_blas_ops.py::test_accuracy_mm \
        /env/FlagGems/tests/test_blas_ops.py::test_accuracy_bmm \
        /env/FlagGems/tests/test_blas_ops.py::test_accuracy_addmm -q -s
fi
set +e
"$PYTHON" -B "$ROOT/run_pytest.py" --profile diagnostic \
    --output "$OUT/compiler_copy.py" "$@" > "$OUT/run.log" 2>&1
rc=$?
set -e
cat "$OUT/run.log"
printf '%s\n' "$rc" > "$OUT/pytest.exit"
exit "$rc"
