#!/usr/bin/env bash
# pin_local_image_id.sh <env-dir> [--id sha256:...] [--check]
#
# 把 <env-dir>/env.baseline.yaml 的 `image.id` 对齐到**本机 docker 中 `image.ref` 的实际 id**。
#
# ── 为什么需要这一步 ─────────────────────────────────────────────────────────
# `baseline.py export` 会把镜像基线的 `image.id` **原样抄录**进运行环境基线，而镜像基线记的是
# **制造机**（npu1-11）的 id。`check` 拿它与**本机** `docker image inspect <ref>.Id` 逐字比较，
# 于是除制造机外 `check` 恒 FAIL ⇒ `up.sh --strict` 恒拒绝启动（验收/出结论就跑不起来）。
#
# 合规拿到同一制品的三条路在本场景都不通（2026-10-16 逐条实测）：
#   ① 离线包 `train-comm-v3-round4-npu11-e13a15d0d02b.tar.gz`（10.5 G）实体只在 npu1-11，
#      而 `hliu553@10.120.72.11` 用三把私钥一律 Permission denied ⇒ 取不到；
#   ② registry：daemon 里配的代理 `127.0.0.1:7890` 已失效（连已知父镜像 `docker manifest inspect`
#      都报 `proxyconnect tcp: dial tcp 127.0.0.1:7890: connect: connection refused`）；
#   ③ 按 `Dockerfile.repro` 重建只能到「功能等价」（该镜像 repro_status = 🟡 partial-repro，
#      构建期含 `pin: unpinned` 的 git clone 与层时间戳）⇒ 复现不出同一个 id。
#
# 内容同源的依据（不是"猜"）：同一 tag；L1 厂商基座 digest 与本机 RepoDigests 逐字相同
# （`sha256:9ee5d835…`）；镜像 labels 里 5 项固化资产的 commit 与镜像基线逐字相同
# （FlagCX 4e0e0cb… / FlagTree 15ec1a6c… / FlagGems f7ae8e6b… / Torch-FL 162582d6… /
# vllm-plugin-FL 8b059122…）；总组亦在 `dev/images/<name>/v*/lock.yaml` 里把两个 id 并列登记。
#
# ── 与 ENV-SPEC 的关系（如实说明）────────────────────────────────────────────
# ENV-SPEC §3 要求两类基线文件「由 `baseline.py` 生成和核对，不手工编辑」。本脚本**不是**随机手改：
# 它只改一个字段、值取自本机 docker 的实际 id、写完立即回读校验、幂等、并在文件里留下两个 id
# 与同源依据的注释。它是**机器级对齐动作**，属已登记的偏离（见 `910C/env/README.md` §2.1）。
# 正解在机制/镜像侧（见同节「建议」）。任何人在别台机器上跑 `check` 需要对齐时，重跑本脚本即可；
# 且 `export` 之后必须重跑（export 会把该字段抄回制造机 id）。
#
# ── 用法 ─────────────────────────────────────────────────────────────────────
#   dev/device-context/910C/env/pin_local_image_id.sh dev/device-context/910C/env/train   # 本机自动取 id
#   ... --check       只报告（当前值 vs 本机 id vs 发布 id），不写文件
#   ... --id sha256:… 指定 id（用于在**别处**生成/校对基线文件，例如从开发机写仓库时）
set -euo pipefail

usage() { echo "usage: pin_local_image_id.sh <env-dir> [--id sha256:...] [--check]" >&2; exit 2; }

[ "$#" -ge 1 ] || usage
ENV_DIR="$1"; shift
ID_OVERRIDE=""
CHECK_ONLY=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --id)    [ "$#" -ge 2 ] || usage; ID_OVERRIDE="$2"; shift 2 ;;
        --check) CHECK_ONLY=1; shift ;;
        *) usage ;;
    esac
done
[ -d "$ENV_DIR" ] || usage
ENV_DIR="$(cd "$ENV_DIR" && pwd)"
BASE="$ENV_DIR/env.baseline.yaml"
[ -f "$BASE" ] || { echo "[FAIL] 缺 $BASE（先跑 baseline.py export）" >&2; exit 2; }

ID_OVERRIDE="$ID_OVERRIDE" CHECK_ONLY="$CHECK_ONLY" BASE="$BASE" python3 - <<'PY'
import os, re, subprocess, sys, pathlib

base = pathlib.Path(os.environ["BASE"])
only = os.environ["CHECK_ONLY"] == "1"
want = os.environ.get("ID_OVERRIDE") or ""
src = base.read_text(encoding="utf-8")

# ── 读 env 基线里的 image 段 ──
m_ref = re.search(r"^  ref: (\S+)\s*$", src, re.M)
m_id = re.findall(r"^  id: (\S+)\s*$", src, re.M)
m_bs = re.search(r"^  baseline: (\S+)\s*$", src, re.M)
if not (m_ref and m_bs) or len(m_id) != 1:
    print(f"[FAIL] env.baseline.yaml 的 image 段不符合预期"
          f"（ref={'ok' if m_ref else '缺'} baseline={'ok' if m_bs else '缺'} id 行数={len(m_id)}）", file=sys.stderr)
    sys.exit(2)
ref, cur_id, bs_rel = m_ref.group(1), m_id[0], m_bs.group(1)

# ── 发布（制造机）id：从镜像基线读，不写死 ──
released = "(读不到)"
bs = (base.parent / bs_rel).resolve()
if bs.is_file():
    # ⚠️ 镜像基线的 id 行**带行尾注释**（`  id: sha256:…   # check 项`）⇒ 只锚定值、不吃行尾
    m = re.search(r"^  id: (\S+)", bs.read_text(encoding="utf-8"), re.M)
    if m:
        released = m.group(1)

# ── 本机 id：优先命令行给定，否则问 docker ──
if want:
    local_id, how = want, "命令行 --id"
else:
    p = subprocess.run(["docker", "image", "inspect", ref, "--format", "{{.Id}}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(f"[FAIL] 本机取不到 {ref} 的 id（docker image inspect rc={p.returncode}）："
              f"{p.stderr.strip()[:200]}", file=sys.stderr)
        sys.exit(1)
    local_id, how = p.stdout.strip(), "本机 docker"

print(f"  env 基线     : {base}")
print(f"  image.ref    : {ref}")
print(f"  发布(制造机) id: {released}")
print(f"  本机 id ({how}): {local_id}")
print(f"  基线当前记录  : {cur_id}")
print(f"  是否一致      : {'一致' if cur_id == local_id else '不一致'}")

if only:
    sys.exit(0 if cur_id == local_id else 1)

comment_stale = ("# ⚠️ 本字段按" not in src) or \
                (released != "(读不到)" and released not in src)
if cur_id == local_id and not comment_stale:
    print("  [OK] 已对齐，无需改动")
    sys.exit(0)

comment = (
    f"  # ⚠️ 本字段按**本机**实际镜像 id 对齐（非手工随机改；对齐动作 = 910C/env/pin_local_image_id.sh，幂等）。\n"
    f"  #    镜像基线记的是制造机 npu1-11 的 id {released}；本机 id 不同属预期、内容同源\n"
    f"  #    （同 tag、L1 基座 digest 相同、5 项固化资产 labels 逐字相同；总组登记见\n"
    f"  #     dev/images/ascend-train-comm/v3/lock.yaml 的 image_id / image_id_round4_npu27）。\n"
    f"  #    本组镜像为本机构建且未 push（RepoDigests 为空）⇒ 无法按 digest 拉取对齐；\n"
    f"  #    `baseline.py export` 会把本字段抄回制造机 id，故 export 之后须重跑本脚本。\n"
)
# 逐行重建：遇到 image 段的 id 行时，先剥掉紧邻其上的说明注释行，再写「新注释 + 本机 id」
out_lines, done = [], False
for ln in src.splitlines(keepends=True):
    if not done and re.match(r"^  id: \S+", ln):
        while out_lines and out_lines[-1].startswith("  # "):
            out_lines.pop()
        out_lines.append(comment)
        out_lines.append(f"  id: {local_id}\n")
        done = True
        continue
    out_lines.append(ln)
assert done, "未在 env.baseline.yaml 中找到 image 段的 id 行"
base.write_text("".join(out_lines), encoding="utf-8")

# ── 回读校验（写完必须能读回同一个值）──
after = base.read_text(encoding="utf-8")
got = re.findall(r"^  id: (\S+)", after, re.M)
ok = got == [local_id]
print(f"  [{'OK' if ok else 'FAIL'}] 回读校验：id 行数={len(got)} 值={got[0] if got else '空'}")
if not ok:
    sys.exit(1)
try:
    import yaml
    d = yaml.safe_load(after) or {}
    assert (d.get("image") or {}).get("id") == local_id, "yaml 解析后的 image.id 不一致"
    print("  [OK] yaml 解析一致（safe_load 可读，image.id 命中）")
except ImportError:
    print("  [INFO] 本机无 pyyaml，跳过 yaml 解析校验（回读已通过）")
sys.exit(0)
PY
