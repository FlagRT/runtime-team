#!/usr/bin/env python3
"""环境基线管理：镜像基线（image.baseline.yaml）与运行环境基线（env.baseline.yaml）
的生成（export）、核对（check）、按基线切换宿主 repo（load）、只读查看（show/list）。

规范与判定规则见 docs/环境基线管理-改造实施方案-20261010.md（§3 文件形态、
§4.2 Dockerfile 判定规则、§4.3 运行环境行为）。

用法：
  baseline.py export --image <image-dir> [--demo <path>] [--image-ref <ref>]
  baseline.py check  --image <image-dir>
  baseline.py show   --image <image-dir>
  baseline.py export <env-dir> [--require-clean]
  baseline.py check  <env-dir>
  baseline.py load   <env-dir>
  baseline.py show   <env-dir>
  baseline.py list

退出码：0 一致/成功；1 存在不一致（check）或交叉核对失败（export）；
2 用法或文件格式错误；3 load 被安全规则中止。
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from verify_env import REPO_DIR  # 仓库名 -> 宿主目录（相对仓库根）注册表
except Exception:  # noqa: BLE001  —— 单测或裁剪环境下的兜底
    REPO_DIR = {
        "FlagTree": "FlagTree",
        "FlagGems": "FlagGems",
        "Torch-FL": "PyTorch-Plugin-FL",
        "FlagCX": "FlagCX",
        "vllm-plugin-FL": "vllm-plugin-FL",
        "FlagPerf": "FlagPerf",
    }

SCHEMA = "flagrt.baseline/v1"

# 固化 repo -> 镜像标签（commit 以标签为准，Dockerfile ARG 只做交叉核对）
COMMIT_LABEL = {
    "FlagTree": "io.flagrt.flagtree.commit",
    "FlagGems": "io.flagrt.flag_gems.commit",
    "Torch-FL": "io.flagrt.torch_fl.commit",
    "vllm-plugin-FL": "io.flagrt.vllm_plugin_fl.ref",
    "FlagCX": "io.flagrt.flagcx.commit",
}
BRANCH_LABEL = {
    "FlagTree": "io.flagrt.flagtree.branch",
}
# 四层模型层级（层级与交付方式正交，见方案 §1/§3）
LAYER = {
    "vllm": "L3",
    "FlagTree": "L2",
    "FlagGems": "L2",
    "Torch-FL": "L3",
    "vllm-plugin-FL": "L3",
    "FlagCX": "L3",
    "llvm-toolchain": "L2",
    "triton-build-deps": "L2",
    "patch_triton_ascend_flagtree.py": "L2",
    "FlagGems-DSA-__init__.py": "L2",
    "verify_runtime.py": "L2",
    "verify_flagcx_runtime.py": "L3",
    "verify_flagcx_p2p.py": "L3",
    "json": "L3",
    "googletest": "L3",
}
DEFAULT_LAYER = "L3"

# 外部基座的 provides 关注哪些包
PROVIDES_PKGS = ("torch", "torch_npu", "vllm")

# COPY src/<Repo>/ 的目录名与基线 repo 名不一致时的映射（目前一致，留作登记位）
SRC_DIR_REPO = {}


class BaselineError(Exception):
    """用法/格式/判定失败：对应退出码 2（格式、用法）或 1（交叉核对）。"""

    def __init__(self, msg, code=2):
        super().__init__(msg)
        self.code = code


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------

def workspace_root() -> Path:
    """挂载 repo 的宿主根目录；FLAGRT_WORKSPACE_ROOT 可覆盖（测试用临时仓库）。"""
    env = os.environ.get("FLAGRT_WORKSPACE_ROOT")
    return Path(env).resolve() if env else REPO_ROOT


def cache_root() -> Path:
    env = os.environ.get("FLAGRT_BASELINE_CACHE")
    base = Path(env) if env else Path.home() / ".cache" / "flagrt-baseline"
    return base


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def rel_to_repo(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(*args, cwd, timeout=60):
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=timeout
        )
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    if proc.returncode != 0:
        return None, (proc.stderr or proc.stdout).strip()
    return proc.stdout.strip(), None


def y(value) -> str:
    """把一个 python 值渲染成单行 YAML 标量/流式映射。"""
    s = yaml.safe_dump(
        value, default_flow_style=True, allow_unicode=True, width=10 ** 6, sort_keys=False
    ).strip()
    if s.endswith("..."):
        s = s[:-3].strip()
    return s


def load_yaml(path: Path) -> dict:
    if not Path(path).is_file():
        raise BaselineError(f"文件不存在: {path}")
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise BaselineError(f"顶层不是映射: {path}")
    return data


# --------------------------------------------------------------------------
# docker 读取（单测里被 mock）
# --------------------------------------------------------------------------

def inspect_image(ref: str) -> dict:
    proc = subprocess.run(
        ["docker", "image", "inspect", ref], capture_output=True, text=True, timeout=120
    )
    if proc.returncode != 0:
        raise BaselineError(
            f"本机没有该镜像或 ref 有误: {ref} ({(proc.stderr or '').strip()})", code=1
        )
    return json.loads(proc.stdout)[0]


def image_python_report(ref: str, python_bin: str) -> dict:
    """无设备、无挂载地读取镜像内的 pip 元数据（editable 安装 + 关键包版本）。"""
    def _run(args):
        proc = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", python_bin, ref, *args],
            capture_output=True, text=True, timeout=900,
        )
        if proc.returncode != 0:
            raise BaselineError(
                f"读取镜像 pip 元数据失败: {ref} {' '.join(args)}\n{(proc.stderr or '').strip()}",
                code=1,
            )
        return proc.stdout

    inspect_out = _run(["-m", "pip", "inspect"])
    freeze_out = _run(["-m", "pip", "freeze"])
    data = json.loads(inspect_out)
    versions = {}
    editables = {}
    for pkg in data.get("installed", []):
        meta = pkg.get("metadata") or {}
        name = (meta.get("name") or "").replace("-", "_")
        versions[name] = meta.get("version")
        direct = pkg.get("direct_url") or {}
        if (direct.get("dir_info") or {}).get("editable"):
            editables[name] = {"source": direct.get("url", ""), "commit": None}
    for line in freeze_out.splitlines():
        m = re.match(r"^-e\s+(\S+?)@([0-9a-f]{40})#egg=(\S+)$", line.strip())
        if m:
            url, commit, name = m.group(1), m.group(2), m.group(3).replace("-", "_")
            e = editables.setdefault(name, {"source": "", "commit": None})
            e["commit"] = commit
            e["repo_url"] = url[len("git+"):] if url.startswith("git+") else url
    env = data.get("environment") or {}
    return {
        "python_version": env.get("python_full_version") or env.get("python_version"),
        "versions": versions,
        "editables": editables,
    }


# --------------------------------------------------------------------------
# 资源 sha256：assets 文件直接算，URL 走 ~/.cache/flagrt-baseline 下载缓存
# --------------------------------------------------------------------------

def _content_length(url: str):
    proc = subprocess.run(["curl", "-sIL", "--max-time", "120", url],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    sizes = re.findall(r"(?im)^content-length:\s*(\d+)", proc.stdout)
    return int(sizes[-1]) if sizes else None


def cached_download_sha256(url: str) -> str:
    cache = cache_root() / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    target = cache / f"{key}-{url.rsplit('/', 1)[-1]}"
    side = Path(str(target) + ".sha256")
    if side.is_file():
        txt = side.read_text(encoding="utf-8").split()
        if txt and re.fullmatch(r"[0-9a-f]{64}", txt[0]):
            return txt[0]
    if not target.is_file():
        print(f"[INFO] 下载资源以计算 sha256（大文件，缓存在 {target}）: {url}")
        part = Path(str(target) + ".part")
        expect = _content_length(url)
        if part.is_file() and expect and part.stat().st_size > expect:
            part.unlink()  # 断点续传写坏过的残留，重下
        proc = subprocess.run(
            ["curl", "-fL", "-C", "-", "--retry", "5", "--retry-all-errors",
             "--retry-delay", "5", "-o", str(part), url],
            capture_output=True, text=True, timeout=36000,
        )
        if proc.returncode != 0:
            raise BaselineError(
                f"下载失败，无法计算 sha256: {url}\n{(proc.stderr or '').strip()}", code=1
            )
        if expect and part.stat().st_size != expect:
            got = part.stat().st_size
            part.unlink()
            raise BaselineError(
                f"下载长度与 Content-Length 不一致（{got} != {expect}），已丢弃: {url}", code=1
            )
        part.rename(target)
    digest = sha256_file(target)
    side.write_text(digest + "\n", encoding="utf-8")
    return digest


# --------------------------------------------------------------------------
# info 块（只供阅读，不参与校验）
# --------------------------------------------------------------------------

def _branches_containing(repo_dir: Path, commit: str):
    out, _ = run_git("branch", "-r", "--contains", commit, cwd=repo_dir)
    names = []
    for line in (out or "").splitlines():
        line = line.strip()
        if not line or "->" in line:
            continue
        names.append(re.sub(r"^[^/]+/", "", line))
    if names:
        return sorted(set(names))
    out, _ = run_git("branch", "--contains", commit, cwd=repo_dir)
    for line in (out or "").splitlines():
        line = line.strip().lstrip("* ").strip()
        if line and not line.startswith("("):
            names.append(line)
    return sorted(set(names))


def _format_branches(names):
    """同一 commit 可能在多个分支上。优先 master/main，分支很多时只列前三个并给出总数
    （info 仅供阅读，不参与校验）。"""
    if not names:
        return None
    pri = next((n for n in ("master", "main") if n in names), None)
    ordered = ([pri] if pri else []) + [n for n in names if n != pri]
    if len(ordered) <= 3:
        return ", ".join(ordered)
    return ", ".join(ordered[:3]) + f" 等共 {len(ordered)} 个分支"


def _committed_at(repo_dir: Path, commit: str):
    out, _ = run_git("show", "-s", "--format=%ci", commit, cwd=repo_dir)
    return out


def repo_info(name: str, commit: str, repo_url=None, branch_hint=None):
    """branch / committed_at，只供阅读。取不到返回 None（写成 info: unavailable）。"""
    if not commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
        return None
    host = workspace_root() / REPO_DIR.get(name, name)
    if (host / ".git").exists():
        ok, _ = run_git("cat-file", "-e", commit + "^{commit}", cwd=host)
        if ok is not None:
            info = {}
            branch = _format_branches(_branches_containing(host, commit)) or branch_hint
            if branch:
                info["branch"] = branch
            at = _committed_at(host, commit)
            if at:
                info["committed_at"] = at
            return info or None
    if repo_url:
        info = _info_from_cache_repo(name, commit, repo_url)
        if info is not None:
            if "branch" not in info and branch_hint:
                info["branch"] = branch_hint
            return {k: info[k] for k in ("branch", "committed_at") if k in info}
    if branch_hint:
        return {"branch": branch_hint}
    return None


def _info_from_cache_repo(name: str, commit: str, repo_url: str):
    bare = cache_root() / "repos" / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.git"
    if not (bare / "HEAD").exists():
        bare.mkdir(parents=True, exist_ok=True)
        out, err = run_git("init", "--bare", "-q", cwd=bare)
        if out is None:
            return None
    ok, _ = run_git("cat-file", "-e", commit + "^{commit}", cwd=bare)
    if ok is None:
        print(f"[INFO] 本机未检出 {name}，从 {repo_url} 浅取 {commit[:12]} 以读取提交元数据")
        out, err = run_git("fetch", "--depth", "1", repo_url, commit, cwd=bare, timeout=900)
        if out is None:
            print(f"[WARN] 浅取失败，{name} 的 info 标为 unavailable: {err}")
            return None
    info = {}
    branch = _format_branches(_branches_containing(bare, commit))
    if branch:
        info["branch"] = branch
    at = _committed_at(bare, commit)
    if at:
        info["committed_at"] = at
    return info or None


# --------------------------------------------------------------------------
# Dockerfile 解析（方案 §4.2：只识别列出的模式，其余一律报错）
# --------------------------------------------------------------------------

KNOWN_OPS = {
    "FROM", "ARG", "LABEL", "ENV", "RUN", "COPY", "SHELL", "WORKDIR", "USER",
    "EXPOSE", "ENTRYPOINT", "CMD", "VOLUME", "STOPSIGNAL", "HEALTHCHECK",
    "MAINTAINER",
}


def parse_instructions(text: str):
    lines, cur = [], ""
    for raw in text.splitlines():
        s = raw.strip()
        if not cur and (not s or s.startswith("#")):
            continue
        if cur and s.startswith("#"):
            continue
        if s.endswith("\\"):
            cur += s[:-1].strip() + " "
            continue
        cur += s
        lines.append(cur.strip())
        cur = ""
    if cur.strip():
        lines.append(cur.strip())
    out = []
    for line in lines:
        m = re.match(r"^([A-Za-z]+)\s+(.*)$", line)
        if not m:
            raise BaselineError(f"无法解析的 Dockerfile 指令: {line[:80]}")
        op = m.group(1).upper()
        if op not in KNOWN_OPS:
            raise BaselineError(f"不识别的 Dockerfile 指令 {op}（§4.2 只识别受支持的模式）")
        out.append((op, m.group(2).strip()))
    return out


def expand(s: str, args: dict) -> str:
    def _sub(m):
        name, mod = m.group(1), m.group(2) or ""
        if name not in args:
            return m.group(0)
        val = args[name]
        ms = re.fullmatch(r":(\d+):(\d+)", mod)
        if ms:
            start, length = int(ms.group(1)), int(ms.group(2))
            return val[start:start + length]
        return val

    prev = None
    out = s
    for _ in range(5):
        prev, out = out, re.sub(r"\$\{(\w+)(:[^}]*)?\}", _sub, out)
        if out == prev:
            break
    return out


def _copy_operands(body: str):
    toks = [t for t in shlex.split(body) if not t.startswith("--")]
    if len(toks) < 2:
        raise BaselineError(f"COPY 操作数不足: {body}")
    return toks[:-1], toks[-1]


def _segments(body: str):
    """把一条 RUN 拆成以 && / ; / | 分隔的小段，保留顺序。"""
    return [s.strip() for s in re.split(r"&&|\|\||;|\n", body) if s.strip()]


def analyze_dockerfile(image_dir: Path) -> dict:
    """返回该 Dockerfile 的固化判定结果；遇到不识别的代码入镜方式即报错。"""
    image_dir = Path(image_dir).resolve()
    df = image_dir / "Dockerfile.repro"
    if not df.is_file():
        raise BaselineError(f"未找到 Dockerfile.repro: {df}")
    instructions = parse_instructions(df.read_text(encoding="utf-8"))

    args: dict = {}
    labels: dict = {}
    from_ref = None
    repos: dict = {}       # name -> info
    dir_repo: dict = {}    # 镜像内目录 -> repo 名
    resources: list = []
    patch_dirs: dict = {}  # 镜像内 patches 目录 -> 仓内 assets 相对路径
    warnings: list = []
    order = 0

    def _add_repo(name, **kw):
        nonlocal order
        if name in repos:
            raise BaselineError(f"同一 repo 在本层出现两次，无法判定: {name}")
        order += 1
        repos[name] = {"name": name, "order": order, "patches": [], **kw}
        return repos[name]

    for op, body in instructions:
        if op == "ARG":
            m = re.match(r"^(\w+)\s*=\s*(.*)$", body)
            if m:
                args[m.group(1)] = expand(m.group(2).strip().strip('"\''), args)
            else:
                args.setdefault(body.strip(), "")
        elif op == "FROM":
            from_ref = expand(body.split()[0], args)
        elif op == "LABEL":
            for k, v in re.findall(r'([\w.\-]+)\s*=\s*("(?:[^"\\]|\\.)*"|\S+)', body):
                val = v[1:-1].replace('\\"', '"') if v.startswith('"') else v
                labels[k] = expand(val, args)
        elif op == "COPY":
            srcs, dest = _copy_operands(expand(body, args))
            for src in srcs:
                _handle_copy(image_dir, src, dest, dir_repo, resources, patch_dirs, _add_repo)
        elif op == "RUN":
            _handle_run(body, expand(body, args), args, dir_repo, repos, resources,
                        patch_dirs, warnings, _add_repo)

    for d, name in dir_repo.items():
        if not repos[name].get("install"):
            raise BaselineError(
                f"{name}: 源码进入镜像（{d}）但未找到对应的 pip install，无法判定固化方式（§4.2）"
            )
    return {
        "image_dir": image_dir,
        "args": args,
        "labels": labels,
        "from_ref": from_ref,
        "repos": repos,
        "resources": resources,
        "warnings": warnings,
    }


def _handle_copy(image_dir, src, dest, dir_repo, resources, patch_dirs, add_repo):
    s = src.lstrip("./")
    if s.startswith("src/"):
        parts = [p for p in s.split("/") if p]
        if len(parts) != 2:
            raise BaselineError(f"COPY 源码路径不符合 src/<Repo>/ 模式: {src}")
        name = SRC_DIR_REPO.get(parts[1], parts[1])
        add_repo(name, source="copy", src=s, dir=dest.rstrip("/"))
        dir_repo[dest.rstrip("/")] = name
        return
    if s.startswith("assets/"):
        local = image_dir / s
        if s.startswith("assets/patches/"):
            patch_dirs[dest.rstrip("/")] = s
            return
        if src.endswith("/") or local.is_dir():
            if not local.is_dir():
                raise BaselineError(f"COPY 的 assets 目录不存在: {local}")
            for f in sorted(local.rglob("*")):
                if f.is_file():
                    resources.append({"name": f.name, "kind": "asset",
                                      "path": rel_to_repo(f)})
            return
        if not local.is_file():
            raise BaselineError(f"COPY 的 assets 文件不存在: {local}")
        resources.append({"name": local.name, "kind": "asset", "path": rel_to_repo(local)})
        return
    raise BaselineError(
        f"COPY 来源不在 §4.2 识别范围内（只识别 src/<Repo>/ 与 assets/…）: {src}"
    )


def _handle_run(raw, body, args, dir_repo, repos, resources, patch_dirs, warnings, add_repo):
    """raw: 未展开 ARG 的原文（用于识别 ${<X>_COMMIT|REF|URL} 引用）；
    body: 展开后的文本（用于路径、URL 等实际值）。"""
    if re.search(r"pip\s+install\b[^&|;]*(\s-e\s|\s--editable\b)", body):
        raise BaselineError(
            "本构建层出现 pip install -e：固化资产不允许 editable 安装（方案 §2.1）"
        )
    if re.search(r"\bwget\b", body):
        raise BaselineError("RUN 中出现 wget，§4.2 只识别 curl ... ${<X>_URL}")

    segs = _segments(body)

    # 2. git remote add origin <url> + git fetch ... ${<X>_COMMIT|REF}
    m_remote = re.search(r"git\s+remote\s+add\s+origin\s+(\S+)", body)
    cur_repo = None
    if m_remote:
        url = m_remote.group(1)
        name = re.sub(r"\.git$", "", url.rsplit("/", 1)[-1])
        m_fetch_raw = re.search(r"git\s+fetch\b([^&|;]*)", raw)
        m_fetch = re.search(r"git\s+fetch\b([^&|;]*)", body)
        if not m_fetch or not m_fetch_raw:
            raise BaselineError(f"{name}: 有 git remote add 但没有 git fetch，无法判定 commit 来源")
        fetch_args = m_fetch.group(1)
        m_arg = re.search(r"\$\{(\w+)\}", m_fetch_raw.group(0))
        arg_name = None
        ref_value = None
        if m_arg:
            arg_name = m_arg.group(1)
            ref_value = args.get(arg_name)
            if not ref_value:
                raise BaselineError(f"{name}: ARG {arg_name} 没有默认值，无法确定 commit")
        else:
            m_sha = re.search(r"\b([0-9a-f]{40})\b", fetch_args)
            if not m_sha:
                raise BaselineError(
                    f"{name}: git fetch 未固定到 ARG 或 40 位 commit，拒绝猜测（§4.2）"
                )
            ref_value = m_sha.group(1)
        m_branch = re.search(r"git\s+checkout[^&|;]*-b\s+(\S+)", body)
        m_dir = re.search(r"\bcd\s+(\S+)", body)
        cur_repo = add_repo(
            name, source="fetch", url=url, arg=arg_name, arg_value=ref_value,
            branch_hint=m_branch.group(1) if m_branch else None,
            dir=m_dir.group(1).rstrip("/") if m_dir else None,
        )
        if cur_repo["dir"]:
            dir_repo[cur_repo["dir"]] = name

    # 4a. git clone（不带 pin）：FlagCX 的公开 submodule。记录 + 警告，不猜 commit。
    for m in re.finditer(r"git\s+clone\b([^&|;]*)", body):
        toks, skip = [], False
        for t in m.group(1).split():
            if skip:
                skip = False
                continue
            if t.startswith("-"):
                skip = t in ("--depth", "-b", "--branch", "-c", "--config", "-o", "--origin")
                continue
            toks.append(t)
        if not toks:
            raise BaselineError(f"无法解析的 git clone: {m.group(0)}")
        url = toks[0]
        name = re.sub(r"\.git$", "", url.rsplit("/", 1)[-1])
        target = toks[1] if len(toks) > 1 else name
        resources.append({"name": name, "kind": "git-clone-floating", "url": url,
                          "path_in_image": target})
        warnings.append(
            f"{name}: 构建期 git clone 未固定 commit（{url}），只能记录 URL，"
            "版本事实由 image id 兜底"
        )

    # 4b. curl ... ${<X>_URL}
    for seg in _segments(raw):
        if not re.search(r"\bcurl\b", seg):
            continue
        m_url = re.search(r"\$\{(\w+_URL)\}", seg)
        if not m_url:
            raise BaselineError(
                f"RUN 中的 curl 未使用 ${{<X>_URL}} 形式的 ARG，拒绝猜测（§4.2）: {seg[:80]}"
            )
        arg_name = m_url.group(1)
        url = args.get(arg_name)
        if not url:
            raise BaselineError(f"ARG {arg_name} 没有默认值，无法确定下载来源")
        name = arg_name[:-len("_URL")].lower().replace("_", "-")
        if not any(r.get("url") == url for r in resources):
            resources.append({"name": name, "kind": "url", "url": url, "arg": arg_name})

    # 1/2 的安装侧：pip install <已登记目录|.>
    cd_dir = None
    for seg in segs:
        m_cd = re.match(r"cd\s+(\S+)$", seg)
        if m_cd:
            cd_dir = m_cd.group(1).rstrip("/")
        m_cd2 = re.search(r"\bcd\s+(\S+)", seg)
        if m_cd2 and "pip" not in seg:
            cd_dir = m_cd2.group(1).rstrip("/")
        if not re.search(r"pip\s+install\b", seg):
            continue
        toks = shlex.split(seg)
        try:
            i = next(i for i, t in enumerate(toks)
                     if t == "install" and i and toks[i - 1].endswith("pip"))
        except StopIteration:
            raise BaselineError(f"无法解析的 pip install: {seg[:80]}")
        env_prefix = [t for t in toks[:i] if re.match(r"^USE_[A-Z0-9_]*=", t)]
        targets = [t for t in toks[i + 1:] if not t.startswith("-")]
        if len(targets) != 1:
            raise BaselineError(
                f"pip install 的目标不唯一，拒绝猜测: {seg[:80]}"
            )
        target = targets[0]
        d = cd_dir if target in (".", "./") else target.rstrip("/")
        if d not in dir_repo:
            raise BaselineError(
                f"pip install 的来源未登记为固化 repo（§4.2 只识别 COPY src/<Repo>/ 与"
                f" 构建期 git fetch 两种来源）: {seg[:80]}"
            )
        repo = repos[dir_repo[d]]
        install = "pip (non-editable"
        if env_prefix:
            install += ", " + " ".join(env_prefix)
        repo["install"] = install + ")"
        # git apply 的补丁归属到同一条 RUN 安装的 repo
        for m in re.finditer(r"git\s+apply\b([^&|;]*)", body):
            for tok in m.group(1).split():
                if tok.startswith("-"):
                    continue
                pdir = tok.rsplit("/", 1)[0]
                src_dir = patch_dirs.get(pdir.rstrip("/"))
                if src_dir is None:
                    raise BaselineError(f"git apply 的补丁目录不是 COPY 进来的 assets: {tok}")
                repo["patch_src"] = src_dir


# --------------------------------------------------------------------------
# 镜像基线 export
# --------------------------------------------------------------------------

def build_tag_from_build_sh(image_dir: Path):
    bs = Path(image_dir) / "build.sh"
    if not bs.is_file():
        return None
    tags = re.findall(r"-t\s+(\S+)", bs.read_text(encoding="utf-8"))
    return tags[-1] if tags else None


def find_image_baseline_by_ref(ref: str):
    for p in sorted((REPO_ROOT / "dev" / "images").glob("*/v*/image.baseline.yaml")):
        try:
            data = load_yaml(p)
        except BaselineError:
            continue
        if ((data.get("image") or {}).get("ref")) == ref:
            return p, data
    return None, None


def _packages_file(image_dir: Path):
    prov = Path(image_dir) / "assets" / "provenance"
    if not prov.is_dir():
        return None
    files = sorted(f for f in prov.glob("pipfreeze-*.txt") if f.is_file())
    if not files:
        return None
    npu = [f for f in files if "npu" in f.name]
    return rel_to_repo(max(npu or files, key=lambda f: f.name))


def _from_label(image_dir: Path, kind: str) -> str:
    d = Path(image_dir)
    tag = f"{d.parent.name}/{d.name}"
    return f"{tag} Dockerfile.repro({kind})"


def export_image(image_dir, demo=None, image_ref=None):
    image_dir = Path(image_dir).resolve()
    recipe = analyze_dockerfile(image_dir)
    ref = image_ref or build_tag_from_build_sh(image_dir)
    if not ref:
        raise BaselineError(
            f"无法确定镜像 ref（build.sh 里没有 -t，可用 --image-ref 指定）: {image_dir}"
        )
    info = inspect_image(ref)
    image_id = info.get("Id")
    img_labels = (info.get("Config") or {}).get("Labels") or {}

    warnings = list(recipe["warnings"])

    # Dockerfile 的 LABEL（ARG 展开后）与已构建镜像标签交叉核对
    for key, val in recipe["labels"].items():
        if key in COMMIT_LABEL.values():
            actual = img_labels.get(key)
            if actual is None:
                raise BaselineError(
                    f"镜像缺少标签 {key}（Dockerfile 声明为 {val}）", code=1
                )
            if actual != val:
                raise BaselineError(
                    f"ARG/标签交叉核对失败: {key} Dockerfile={val} 镜像={actual}", code=1
                )

    base_block = None
    baked_repos: list = []
    baked_resources: list = []
    recipe_files: list = []

    parent_path, parent_data = (None, None)
    if recipe["from_ref"]:
        parent_path, parent_data = find_image_baseline_by_ref(recipe["from_ref"])

    if parent_data:
        # 父镜像是本仓自建镜像：展开继承（flatten），层级与来源层原样保留
        parent_img = parent_data.get("image") or {}
        parent_id = parent_img.get("id")
        try:
            local_parent = inspect_image(recipe["from_ref"])
        except BaselineError as e:
            raise BaselineError(f"父镜像不可用: {e}", code=1)
        if parent_id and local_parent.get("Id") != parent_id:
            raise BaselineError(
                f"父镜像 id 与其基线不一致: 基线 {parent_id} 本机 {local_parent.get('Id')}"
                f"（{rel_to_repo(parent_path)}）", code=1,
            )
        base_block = parent_data.get("base_image")
        for name, item in (parent_data.get("baked_repos") or {}).items():
            entry = dict(item)
            entry["_name"] = name
            baked_repos.append(entry)
        for res in parent_data.get("baked_resources") or []:
            baked_resources.append(dict(res))
        recipe_files = list(parent_img.get("recipe") or [])
    elif recipe["from_ref"]:
        # 外部基座：无设备地读取镜像内 pip 元数据补全 L1
        base_info = inspect_image(recipe["from_ref"])
        digest = None
        for rd in base_info.get("RepoDigests") or []:
            if "@" in rd:
                digest = rd.split("@", 1)[1]
                break
        python_bin = recipe["args"].get("PYTHON_BIN")
        if not python_bin:
            raise BaselineError("Dockerfile 未声明 ARG PYTHON_BIN，无法读取基座 pip 元数据")
        report = image_python_report(recipe["from_ref"], python_bin)
        provides = {}
        cann = None
        for e in (base_info.get("Config") or {}).get("Env") or []:
            m = re.match(r"ASCEND_TOOLKIT_HOME=.*/cann-([0-9.]+)$", e)
            if m:
                cann = m.group(1)
        if cann:
            provides["cann"] = cann
        if report["python_version"]:
            provides["python"] = report["python_version"]
        for pkg in PROVIDES_PKGS:
            v = report["versions"].get(pkg)
            if v:
                provides[pkg] = v.replace("+empty", "")
        base_block = {"ref": recipe["from_ref"], "digest": digest, "provides": provides}
        for name, e in sorted(report["editables"].items()):
            src = e.get("source", "")
            inside = src.startswith("file:/")
            if not inside:
                warnings.append(
                    f"{name}: 父镜像的 editable 安装源码不在镜像内（{src}），不记为固化资产"
                )
                continue
            commit = e.get("commit")
            baked_repos.append({
                "_name": name, "layer": LAYER.get(name, DEFAULT_LAYER), "from": "base_image",
                "commit": commit, "install": "editable-in-image",
                "source_path": src[len("file://"):] if src.startswith("file://") else src,
                "info": repo_info(name, commit, repo_url=e.get("repo_url")),
            })
            warnings.append(
                f"{name}: 继承自父镜像的 editable 安装，源码位于镜像内 "
                f"{src}，记为固化资产（install: editable-in-image，方案 §2.1）"
            )

    recipe_files.append(rel_to_repo(image_dir / "Dockerfile.repro"))

    # 本层的固化 repo
    for item in sorted(recipe["repos"].values(), key=lambda r: r["order"]):
        name = item["name"]
        label_key = COMMIT_LABEL.get(name)
        if not label_key:
            raise BaselineError(
                f"{name}: 未登记 commit 标签（请在 baseline.py 的 COMMIT_LABEL 里登记）"
            )
        commit = img_labels.get(label_key)
        if not commit:
            raise BaselineError(f"{name}: 镜像缺少 {label_key} 标签，无法确定 commit", code=1)
        if item.get("arg_value") and item["arg_value"] != commit:
            raise BaselineError(
                f"{name}: ARG {item.get('arg')}={item['arg_value']} 与镜像标签 "
                f"{label_key}={commit} 不一致", code=1,
            )
        kind = "COPY 构建上下文" if item["source"] == "copy" else "构建期 git fetch"
        patches = []
        if item.get("patch_src"):
            pdir = image_dir / item["patch_src"]
            for pf in sorted(pdir.glob("*.patch")):
                patches.append({"file": pf.name, "sha256": sha256_file(pf)})
            if not patches:
                raise BaselineError(f"{name}: 补丁目录内没有 .patch 文件: {pdir}")
            kind += " + 补丁"
        entry = {
            "_name": name,
            "layer": LAYER.get(name, DEFAULT_LAYER),
            "from": _from_label(image_dir, kind),
            "commit": commit,
            "install": item["install"],
            "info": repo_info(name, commit, repo_url=item.get("url"),
                              branch_hint=item.get("branch_hint")
                              or recipe["labels"].get(BRANCH_LABEL.get(name, ""))
                              or img_labels.get(BRANCH_LABEL.get(name, ""))),
        }
        if patches:
            entry["patches"] = patches
        if LAYER.get(name) is None:
            warnings.append(f"{name}: 层级未登记，按默认 {DEFAULT_LAYER} 记录")
        baked_repos.append(entry)

    # 本层的固化资源
    for res in recipe["resources"]:
        entry = {"name": res["name"], "layer": LAYER.get(res["name"], DEFAULT_LAYER)}
        if res["kind"] == "asset":
            entry["path"] = res["path"]
            entry["sha256"] = sha256_file(REPO_ROOT / res["path"])
        elif res["kind"] == "url":
            entry["url"] = res["url"]
            entry["sha256"] = cached_download_sha256(res["url"])
        elif res["kind"] == "git-clone-floating":
            entry["url"] = res["url"]
            entry["path_in_image"] = res["path_in_image"]
            entry["pin"] = "unpinned (构建期 git clone --depth 1)"
        baked_resources.append(entry)

    doc = {
        "image": {
            "ref": ref,
            "id": image_id,
            "recipe": recipe_files,
        },
        "base_image": base_block,
        "baked_repos": baked_repos,
        "baked_resources": baked_resources,
        "packages": _packages_file(image_dir),
    }
    if demo:
        doc["image"]["demo"] = demo

    out = image_dir / "image.baseline.yaml"
    out.write_text(render_image_baseline(doc, image_dir), encoding="utf-8")
    for w in warnings:
        print(f"[WARN] {w}")
    print(f"[OK] 镜像基线已生成 -> {rel_to_repo(out)}")
    return 0


def render_image_baseline(doc: dict, image_dir: Path) -> str:
    L = []
    L.append(f"schema: {SCHEMA}")
    L.append("kind: image")
    L.append(f"generated_by: dev/lib/baseline.py export --image {rel_to_repo(image_dir)}")
    L.append(f"generated_at: {now_iso()}")
    L.append("")
    L.append("image:")
    img = doc["image"]
    L.append(f"  ref: {y(img['ref'])}")
    L.append(f"  id: {img['id']}                 # check 项")
    L.append(f"  recipe: {y(img['recipe'])}")
    if img.get("demo"):
        L.append(f"  demo: {y(img['demo'])}")
    if doc.get("base_image"):
        b = doc["base_image"]
        L.append("")
        L.append("base_image:                                    # L1")
        L.append(f"  ref: {y(b.get('ref'))}")
        L.append(f"  digest: {y(b.get('digest'))}")
        L.append(f"  provides: {y(b.get('provides') or {})}")
    L.append("")
    L.append("baked_repos:")
    if not doc["baked_repos"]:
        L[-1] = "baked_repos: {}"
    for e in doc["baked_repos"]:
        e = dict(e)
        name = e.pop("_name")
        L.append(f"  {name}:")
        L.append(f"    layer: {e.pop('layer', DEFAULT_LAYER)}")
        L.append(f"    from: {y(e.pop('from', 'unknown'))}")
        L.append(f"    commit: {y(e.pop('commit', None))}")
        if e.get("patches"):
            L.append(f"    patches: {y(e.pop('patches'))}")
        if e.get("source_path"):
            L.append(f"    source_path: {y(e.pop('source_path'))}")
        L.append(f"    install: {y(e.pop('install', None))}")
        info = e.pop("info", None)
        L.append(f"    info: {y(info) if info else 'unavailable'}")
        for k, v in e.items():
            L.append(f"    {k}: {y(v)}")
    L.append("")
    L.append("baked_resources:")
    if not doc["baked_resources"]:
        L[-1] = "baked_resources: []"
    for r in doc["baked_resources"]:
        L.append(f"  - {y(r)}")
    if doc.get("packages"):
        L.append("")
        L.append(f"packages: {doc['packages']}   # 完整包清单，仅供查阅")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# 镜像基线 check
# --------------------------------------------------------------------------

def check_image(image_dir) -> int:
    image_dir = Path(image_dir).resolve()
    data = load_yaml(image_dir / "image.baseline.yaml")
    img = data.get("image") or {}
    ref, want_id = img.get("ref"), img.get("id")
    report = []
    ok = True
    try:
        info = inspect_image(ref)
    except BaselineError as e:
        print(f"[FAIL] image: {e}")
        return 1
    actual = info.get("Id")
    if actual != want_id:
        report.append(("FAIL", "image", f"id 不一致: 基线 {want_id} 本机 {actual} (ref={ref})"))
        ok = False
    else:
        report.append(("PASS", "image", f"{ref} -> {actual}"))
    labels = (info.get("Config") or {}).get("Labels") or {}
    for name, item in (data.get("baked_repos") or {}).items():
        key = COMMIT_LABEL.get(name)
        if not key or key not in labels:
            report.append(("INFO", name, "镜像无对应 commit 标签，跳过（由 image id 兜底）"))
            continue
        if labels[key] != item.get("commit"):
            report.append(("FAIL", name, f"标签 {key} 不一致: 基线 {item.get('commit')} "
                                         f"镜像 {labels[key]}"))
            ok = False
        else:
            report.append(("PASS", name, f"{key}={labels[key][:12]}"))
    for status, item, msg in report:
        print(f"[{status}] {item}: {msg}")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# 运行环境场景
# --------------------------------------------------------------------------

def load_env_declaration(env_dir: Path) -> dict:
    env_dir = Path(env_dir).resolve()
    compose = env_dir / "docker-compose.yml"
    if not compose.is_file():
        raise BaselineError(f"运行环境目录缺少 docker-compose.yml: {env_dir}")
    data = load_yaml(compose)
    x = data.get("x-flagrt")
    if not isinstance(x, dict):
        raise BaselineError(
            f"{rel_to_repo(compose)} 缺少顶层 x-flagrt 声明（方案 §3.2）"
        )
    baseline_rel = x.get("image_baseline")
    if not baseline_rel:
        raise BaselineError("x-flagrt 缺少 image_baseline")
    image_baseline = (env_dir / baseline_rel).resolve()
    mounted = x.get("mounted_repos") or {}
    if not isinstance(mounted, dict):
        raise BaselineError("x-flagrt.mounted_repos 不是映射")
    return {
        "env_dir": env_dir,
        "compose": compose,
        "compose_data": data,
        "image_baseline_rel": baseline_rel,
        "image_baseline": image_baseline,
        "mounted": mounted,
    }


def _container_path_for(compose_data: dict, host_dir_name: str):
    for svc in (compose_data.get("services") or {}).values():
        for vol in (svc.get("volumes") or []):
            if isinstance(vol, str):
                parts = vol.split(":")
                if len(parts) >= 2:
                    src, target = parts[0], parts[1]
                else:
                    continue
            elif isinstance(vol, dict):
                src, target = vol.get("source", ""), vol.get("target", "")
            else:
                continue
            if src.rstrip("/").endswith("/" + host_dir_name) or src.rstrip("/") == host_dir_name:
                return target
    return None


def _mounted_state(name: str, spec: dict, decl: dict, baked: dict):
    host_rel = spec.get("host_path") or REPO_DIR.get(name, name)
    host = (workspace_root() / host_rel).resolve()
    entry = {
        "host_path": host_rel,
        "container_path": spec.get("container_path")
        or _container_path_for(decl["compose_data"], Path(host_rel).name)
        or f"/workspace/{Path(host_rel).name}",
        "activation": spec.get("activation", "unknown"),
    }
    if not (host / ".git").exists():
        entry["error"] = f"宿主 repo 不存在或不是 git 仓库: {host}"
        return entry
    head, err = run_git("rev-parse", "HEAD", cwd=host)
    if head is None:
        entry["error"] = f"读取 HEAD 失败: {err}"
        return entry
    dirty_out, _ = run_git("status", "--porcelain", cwd=host)
    entry["commit"] = head
    entry["dirty"] = bool(dirty_out)
    if name in baked:
        entry["overrides_baked"] = True
        entry["baked_commit"] = baked[name].get("commit")
    info = repo_info(name, head)
    if info:
        entry["info"] = info
    return entry


def export_env(env_dir, require_clean=False) -> int:
    decl = load_env_declaration(env_dir)
    image_baseline = load_yaml(decl["image_baseline"])
    baked = image_baseline.get("baked_repos") or {}
    mounted = {}
    dirty_names = []
    for name, spec in decl["mounted"].items():
        spec = spec or {}
        entry = _mounted_state(name, spec, decl, baked)
        if entry.get("error"):
            raise BaselineError(f"{name}: {entry['error']}")
        if entry.get("dirty"):
            dirty_names.append(name)
        mounted[name] = entry
    if dirty_names and require_clean:
        print(f"[FAIL] --require-clean: 以下挂载 repo 工作区不干净，拒绝导出: "
              f"{', '.join(dirty_names)}", file=sys.stderr)
        return 1
    for name in dirty_names:
        print(f"[WARN] {name}: 工作区不干净，已记录 dirty: true")
    doc = {
        "compose": "docker-compose.yml",
        "image": {
            "ref": (image_baseline.get("image") or {}).get("ref"),
            "id": (image_baseline.get("image") or {}).get("id"),
            "baseline": decl["image_baseline_rel"],
        },
        "baked_repos": baked,
        "mounted_repos": mounted,
    }
    out = decl["env_dir"] / "env.baseline.yaml"
    out.write_text(render_env_baseline(doc, decl["env_dir"]), encoding="utf-8")
    print(f"[OK] 运行环境基线已生成 -> {rel_to_repo(out)}")
    return 0


def render_env_baseline(doc: dict, env_dir: Path) -> str:
    L = []
    L.append(f"schema: {SCHEMA}")
    L.append("kind: container")
    L.append(f"generated_by: dev/lib/baseline.py export {rel_to_repo(env_dir)}")
    L.append(f"generated_at: {now_iso()}")
    L.append(f"compose: {doc['compose']}")
    L.append("")
    L.append("image:                       # 抄录自 image.baseline.yaml；check 只核对本地镜像 id")
    for k in ("ref", "id", "baseline"):
        L.append(f"  {k}: {y(doc['image'].get(k))}")
    L.append("")
    L.append("baked_repos:                 # 原样抄录自镜像基线，只读；check 与 load 均不处理")
    if not doc["baked_repos"]:
        L[-1] = L[-1].replace("baked_repos:", "baked_repos: {}", 1)
    for name, item in doc["baked_repos"].items():
        L.append(f"  {name}:")
        for k, v in item.items():
            L.append(f"    {k}: {y(v) if v is not None else 'unavailable'}")
    L.append("")
    L.append("mounted_repos:               # check 与 load 的对象")
    if not doc["mounted_repos"]:
        L[-1] = L[-1].replace("mounted_repos:", "mounted_repos: {}", 1)
    for name, item in doc["mounted_repos"].items():
        L.append(f"  {name}:")
        for k, v in item.items():
            L.append(f"    {k}: {y(v)}")
    return "\n".join(L) + "\n"


def check_env(env_dir, write_last_check=True):
    decl = load_env_declaration(env_dir)
    base = load_yaml(decl["env_dir"] / "env.baseline.yaml")
    report = []
    ok = True
    img = base.get("image") or {}
    try:
        info = inspect_image(img.get("ref"))
        actual = info.get("Id")
        if actual != img.get("id"):
            report.append(("FAIL", "image", f"id 不一致: 基线 {img.get('id')} 本机 {actual}"))
            ok = False
        else:
            report.append(("PASS", "image", f"{img.get('ref')} -> {actual}"))
    except BaselineError as e:
        report.append(("FAIL", "image", str(e)))
        ok = False

    for name, item in (base.get("mounted_repos") or {}).items():
        host = (workspace_root() / item.get("host_path", name)).resolve()
        if not (host / ".git").exists():
            report.append(("FAIL", name, f"宿主 repo 不存在或不是 git 仓库: {host}"))
            ok = False
            continue
        head, err = run_git("rev-parse", "HEAD", cwd=host)
        if head is None:
            report.append(("FAIL", name, f"读取 HEAD 失败: {err}"))
            ok = False
            continue
        if head != item.get("commit"):
            report.append(("FAIL", name, f"commit 不一致: 基线 {str(item.get('commit'))[:12]} "
                                         f"实际 {head[:12]}"))
            ok = False
        else:
            report.append(("PASS", name, f"commit={head[:12]}"))
        dirty_out, _ = run_git("status", "--porcelain", cwd=host)
        if dirty_out:
            report.append(("WARN", name, "工作区有未提交改动（不阻断）"))
        if item.get("overrides_baked"):
            report.append(("INFO", name, f"覆盖镜像内固化版本 "
                                         f"{str(item.get('baked_commit'))[:12]}，"
                                         f"生效方式 {item.get('activation')}"))
    if (base.get("baked_repos")):
        report.append(("INFO", "baked", "固化资产不与宿主比对，由 image id 约束（方案 §2）"))

    for status, item, msg in report:
        print(f"[{status}] {item}: {msg}")

    result = {
        "checked_at": now_iso(),
        "env_dir": rel_to_repo(decl["env_dir"]),
        "baseline": "env.baseline.yaml",
        "report": [{"status": s, "item": i, "detail": m} for s, i, m in report],
        "result": "PASS" if ok else "FAIL",
    }
    if write_last_check:
        out = decl["env_dir"] / ".last-check.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"check 结果 -> {rel_to_repo(out)}")
    return 0 if ok else 1


def load_env(env_dir) -> int:
    decl = load_env_declaration(env_dir)
    base = load_yaml(decl["env_dir"] / "env.baseline.yaml")
    repos = base.get("mounted_repos") or {}
    if not repos:
        print("[INFO] 基线没有声明挂载 repo，load 无事可做")
        return 0

    plan = []
    dirty = []
    for name, item in repos.items():
        host = (workspace_root() / item.get("host_path", name)).resolve()
        if not (host / ".git").exists():
            print(f"[FAIL] {name}: 宿主 repo 不存在或不是 git 仓库: {host}", file=sys.stderr)
            return 3
        dirty_out, _ = run_git("status", "--porcelain", cwd=host)
        if dirty_out:
            dirty.append(name)
        head, _ = run_git("rev-parse", "HEAD", cwd=host)
        branch, _ = run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=host)
        plan.append((name, host, item.get("commit"), head, branch))
    if dirty:
        print("[ABORT] 以下挂载 repo 工作区不干净，整体中止，未做任何切换（方案 §4.3）: "
              + ", ".join(dirty), file=sys.stderr)
        return 3

    for name, host, want, head, branch in plan:
        if not want:
            print(f"[FAIL] {name}: 基线没有 commit，无法切换", file=sys.stderr)
            return 2
        print(f"[INFO] {name}: 切换前 commit={str(head)[:12]} branch={branch}（回退用）")
        ok, _ = run_git("cat-file", "-e", want + "^{commit}", cwd=host)
        if ok is None:
            print(f"[INFO] {name}: 本机缺少 {want[:12]}，先 git fetch")
            out, err = run_git("fetch", "--all", "--tags", cwd=host, timeout=900)
            if out is None:
                print(f"[FAIL] {name}: git fetch 失败: {err}", file=sys.stderr)
                return 1
        out, err = run_git("checkout", "--detach", want, cwd=host, timeout=300)
        if out is None:
            print(f"[FAIL] {name}: checkout {want[:12]} 失败: {err}", file=sys.stderr)
            return 1
        new, _ = run_git("rev-parse", "HEAD", cwd=host)
        print(f"[OK] {name}: 切换后 commit={str(new)[:12]} (detached HEAD)")
    return 0


# --------------------------------------------------------------------------
# show / list
# --------------------------------------------------------------------------

def _dw(s: str) -> int:
    """显示宽度：CJK 全角字符占两列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _table(rows, headers):
    rows = [[("" if c is None else str(c)) for c in r] for r in rows]
    widths = [_dw(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], _dw(c))

    def pad(cell, i):
        return cell + " " * max(0, widths[i] - _dw(cell))

    out = ["  ".join(pad(h, i) for i, h in enumerate(headers)).rstrip()]
    out.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        out.append("  ".join(pad(c, i) for i, c in enumerate(r)).rstrip())
    return "\n".join(out)


def _info_str(info):
    if not isinstance(info, dict):
        return "unavailable"
    return " / ".join(str(info.get(k)) for k in ("branch", "committed_at") if info.get(k))


def show_image(image_dir) -> int:
    image_dir = Path(image_dir).resolve()
    data = load_yaml(image_dir / "image.baseline.yaml")
    img = data.get("image") or {}
    print(f"镜像: {img.get('ref')}")
    print(f"id:   {img.get('id')}")
    print(f"配方: {', '.join(img.get('recipe') or [])}")
    if img.get("demo"):
        print(f"demo: {img.get('demo')}")
    b = data.get("base_image") or {}
    if b:
        print(f"基座: {b.get('ref')}")
        print(f"      digest={b.get('digest')} provides={b.get('provides')}")
    print()
    rows = []
    for name, item in (data.get("baked_repos") or {}).items():
        rows.append([item.get("layer"), name, str(item.get("commit"))[:12],
                     item.get("install"), _info_str(item.get("info")), item.get("from")])
    print("固化 repo（baked，由 image id 约束，不比对宿主）:")
    print(_table(rows, ["层级", "资产", "commit", "安装方式", "info（只读）", "来源层"]))
    res = data.get("baked_resources") or []
    if res:
        print()
        print("固化资源（baked_resources）:")
        rows = [[r.get("layer"), r.get("name"),
                 str(r.get("sha256"))[:16] if r.get("sha256") else "unpinned",
                 r.get("url") or r.get("path")] for r in res]
        print(_table(rows, ["层级", "名称", "sha256/pin", "来源"]))
    if data.get("packages"):
        print()
        print(f"完整包清单: {data['packages']}")
    return 0


def show_env(env_dir) -> int:
    env_dir = Path(env_dir).resolve()
    data = load_yaml(env_dir / "env.baseline.yaml")
    img = data.get("image") or {}
    print(f"运行环境: {rel_to_repo(env_dir)}")
    print(f"镜像: {img.get('ref')}")
    print(f"id:   {img.get('id')}  (基线 {img.get('baseline')})")
    print()
    rows = []
    for name, item in (data.get("mounted_repos") or {}).items():
        rows.append([name, str(item.get("commit"))[:12], item.get("activation"),
                     "是" if item.get("dirty") else "否",
                     ("是 <- " + str(item.get("baked_commit"))[:12])
                     if item.get("overrides_baked") else "否",
                     item.get("container_path"), _info_str(item.get("info"))])
    print("挂载 repo（mounted，check/load 的对象）:")
    print(_table(rows, ["资产", "commit", "生效方式", "dirty", "覆盖固化", "容器内路径",
                        "info（只读）"]))
    rows = []
    for name, item in (data.get("baked_repos") or {}).items():
        rows.append([item.get("layer"), name, str(item.get("commit"))[:12],
                     item.get("install"),
                     "被挂载覆盖" if name in (data.get("mounted_repos") or {}) else ""])
    if rows:
        print()
        print("固化 repo（baked，只读抄录）:")
        print(_table(rows, ["层级", "资产", "commit", "安装方式", "备注"]))
    return 0


def list_baselines() -> int:
    files = sorted((REPO_ROOT / "dev" / "images").glob("*/v*/image.baseline.yaml"))
    if not files:
        print("未找到任何 image.baseline.yaml")
        return 0
    rows = []
    for f in files:
        try:
            data = load_yaml(f)
        except BaselineError as e:
            print(f"[跳过] {rel_to_repo(f)}: {e}", file=sys.stderr)
            continue
        img = data.get("image") or {}
        image_name = f"{f.parent.parent.name}/{f.parent.name}"
        items = list((data.get("baked_repos") or {}).items())
        res = data.get("baked_resources") or []
        first = True
        for name, item in items:
            rows.append([image_name if first else "", str(img.get("id", ""))[7:19] if first else "",
                         item.get("layer"), name, str(item.get("commit"))[:12],
                         item.get("install")])
            first = False
        for r in res:
            rows.append([image_name if first else "", "", r.get("layer"), r.get("name"),
                         str(r.get("sha256"))[:12] if r.get("sha256") else "unpinned",
                         "资源"])
            first = False
        if first:
            rows.append([image_name, str(img.get("id", ""))[7:19], "", "(无固化资产)", "", ""])
    print("仓内全部镜像基线（dev/images/*/v*/image.baseline.yaml）:")
    print(_table(rows, ["镜像", "image id", "层级", "资产", "版本/sha256", "安装方式"]))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="baseline.py",
        description="环境基线管理（镜像基线 / 运行环境基线）。规范见 "
                    "docs/环境基线管理-改造实施方案-20261010.md",
    )
    sub = p.add_subparsers(dest="action")

    def add_target(sp):
        sp.add_argument("--image", metavar="IMAGE_DIR", help="镜像场景：镜像配方目录")
        sp.add_argument("env_dir", nargs="?", help="运行环境场景：运行环境目录")

    sp = sub.add_parser("export", help="生成基线文件")
    add_target(sp)
    sp.add_argument("--demo", help="镜像场景：demo 运行环境目录（写入 image.demo）")
    sp.add_argument("--image-ref", help="镜像场景：显式指定镜像 ref（默认取 build.sh 的 -t）")
    sp.add_argument("--require-clean", action="store_true",
                    help="运行环境场景：挂载 repo 不干净时拒绝导出")

    sp = sub.add_parser("check", help="核对基线与本机实际")
    add_target(sp)

    sp = sub.add_parser("load", help="按基线切换挂载 repo（仅运行环境场景）")
    sp.add_argument("env_dir", help="运行环境目录")

    sp = sub.add_parser("show", help="只读打印基线")
    add_target(sp)

    sub.add_parser("list", help="列出仓内全部镜像基线的资产与版本")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.action:
        parser.print_usage(sys.stderr)
        return 2
    try:
        if args.action == "list":
            return list_baselines()
        if args.action == "load":
            return load_env(args.env_dir)
        image = getattr(args, "image", None)
        env_dir = getattr(args, "env_dir", None)
        if bool(image) == bool(env_dir):
            print("用法错误: 必须且只能给一个场景（--image <image-dir> 或 <env-dir>）",
                  file=sys.stderr)
            return 2
        if args.action == "export":
            if image:
                return export_image(image, demo=args.demo, image_ref=args.image_ref)
            return export_env(env_dir, require_clean=args.require_clean)
        if args.action == "check":
            return check_image(image) if image else check_env(env_dir)
        if args.action == "show":
            return show_image(image) if image else show_env(env_dir)
    except BaselineError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return e.code
    parser.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
