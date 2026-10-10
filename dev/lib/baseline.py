#!/usr/bin/env python3
"""环境基线管理：镜像基线（image.baseline.yaml）与运行环境基线（env.baseline.yaml）
的生成（export）、核对（check）、按基线切换宿主 repo（load）、只读查看（show/list）。
规范见 docs/环境基线管理-改造实施方案-20261010.md（§3 文件形态、§4.2 判定规则、§4.3 运行环境）。

用法：
  baseline.py export --image <image-dir> [--demo <path>] [--image-ref <ref>]
  baseline.py check|show --image <image-dir>
  baseline.py export <env-dir> [--require-clean]
  baseline.py check|load|show <env-dir>
  baseline.py list

退出码：0 一致/成功；1 不一致（check）或交叉核对失败（export/load）；
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
SCHEMA = "flagrt.baseline/v1"

# 仓库名 -> 宿主目录（相对 workspace 根）
REPO_DIR = {
    "FlagTree": "FlagTree",
    "FlagGems": "FlagGems",
    "Torch-FL": "PyTorch-Plugin-FL",
    "FlagCX": "FlagCX",
    "vllm-plugin-FL": "vllm-plugin-FL",
    "FlagPerf": "FlagPerf",
}
# 固化 repo 的 commit 以镜像标签为准，Dockerfile ARG 只做交叉核对
COMMIT_LABEL = {
    "FlagTree": "io.flagrt.flagtree.commit",
    "FlagGems": "io.flagrt.flag_gems.commit",
    "Torch-FL": "io.flagrt.torch_fl.commit",
    "vllm-plugin-FL": "io.flagrt.vllm_plugin_fl.ref",
    "FlagCX": "io.flagrt.flagcx.commit",
}
BRANCH_LABEL = {"FlagTree": "io.flagrt.flagtree.branch"}
# 四层模型层级（与交付方式正交，方案 §1/§3）；未登记的按 L3
LAYER = {
    "FlagTree": "L2", "FlagGems": "L2", "llvm-toolchain": "L2", "triton-build-deps": "L2",
    "patch_triton_ascend_flagtree.py": "L2", "FlagGems-DSA-__init__.py": "L2",
    "verify_runtime.py": "L2",
}
DEFAULT_LAYER = "L3"
SHA40 = re.compile(r"[0-9a-f]{40}")


class BaselineError(Exception):
    """code 2：用法/格式错误；code 1：交叉核对失败。"""

    def __init__(self, msg, code=2):
        super().__init__(msg)
        self.code = code


# ---- 基础工具 ----

def workspace_root() -> Path:
    """挂载 repo 的宿主根目录。up.sh 以同一值导出 compose 的 WORKSPACE_ROOT。"""
    env = os.environ.get("FLAGRT_WORKSPACE_ROOT")
    return Path(env).resolve() if env else REPO_ROOT


def cache_root() -> Path:
    env = os.environ.get("FLAGRT_BASELINE_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "flagrt-baseline"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def rel_to_repo(p) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(*args, cwd, timeout=60):
    """返回 (stdout, None) 或 (None, 错误信息)。"""
    try:
        proc = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                              text=True, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    if proc.returncode != 0:
        return None, (proc.stderr or proc.stdout).strip()
    return proc.stdout.strip(), None


def y(value) -> str:
    """python 值 -> 单行 YAML（流式）。"""
    s = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True,
                       width=10 ** 6, sort_keys=False).strip()
    return s[:-3].strip() if s.endswith("...") else s


def load_yaml(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise BaselineError(f"文件不存在: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise BaselineError(f"YAML 解析失败: {path}\n{e}")
    if not isinstance(data, dict):
        raise BaselineError(f"顶层不是映射: {path}")
    return data


# ---- docker 读取（单测里 mock） ----

def inspect_image(ref: str) -> dict:
    proc = subprocess.run(["docker", "image", "inspect", ref], capture_output=True,
                          text=True, timeout=120)
    if proc.returncode != 0:
        raise BaselineError(f"本机没有该镜像或 ref 有误: {ref} ({proc.stderr.strip()})", code=1)
    return json.loads(proc.stdout)[0]


def image_python_report(ref: str, python_bin: str) -> dict:
    """无设备、无挂载地读取镜像内 pip 元数据（关键包版本 + editable 安装）。"""
    def _run(*args):
        proc = subprocess.run(["docker", "run", "--rm", "--entrypoint", python_bin, ref, *args],
                              capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            raise BaselineError(f"读取镜像 pip 元数据失败: {ref} {' '.join(args)}\n"
                                f"{proc.stderr.strip()}", code=1)
        return proc.stdout

    data = json.loads(_run("-m", "pip", "inspect"))
    versions, editables = {}, {}
    for pkg in data.get("installed", []):
        meta = pkg.get("metadata") or {}
        name = (meta.get("name") or "").replace("-", "_")
        versions[name] = meta.get("version")
        direct = pkg.get("direct_url") or {}
        if (direct.get("dir_info") or {}).get("editable"):
            editables[name] = {"source": direct.get("url", ""), "commit": None}
    for line in _run("-m", "pip", "freeze").splitlines():
        m = re.match(r"^-e\s+(\S+?)@([0-9a-f]{40})#egg=(\S+)$", line.strip())
        if m:
            e = editables.setdefault(m.group(3).replace("-", "_"), {"source": ""})
            e["commit"] = m.group(2)
            e["repo_url"] = re.sub(r"^git\+", "", m.group(1))
    env = data.get("environment") or {}
    return {"python_version": env.get("python_full_version") or env.get("python_version"),
            "versions": versions, "editables": editables}


def compose_binds(compose: Path):
    """docker compose 实际解析出的 bind 挂载 [(宿主源, 容器目标)]。

    WORKSPACE_ROOT 未设置时取 workspace_root()，与 up.sh 一致；已设置则照用，
    从而能发现它与 FLAGRT_WORKSPACE_ROOT 不一致的情况。"""
    env = dict(os.environ)
    env.setdefault("WORKSPACE_ROOT", str(workspace_root()))
    try:
        proc = subprocess.run(["docker", "compose", "-f", str(compose), "config",
                               "--format", "json"], capture_output=True, text=True,
                              timeout=120, env=env)
    except OSError as e:
        raise BaselineError(f"无法执行 docker compose config: {e}")
    if proc.returncode != 0:
        raise BaselineError(f"docker compose config 失败: {proc.stderr.strip()}")
    binds = []
    for svc in (json.loads(proc.stdout).get("services") or {}).values():
        for v in svc.get("volumes") or []:
            if v.get("type") == "bind":
                binds.append((Path(v["source"]).resolve(), Path(v["target"])))
    return binds


def _translate(pairs, path: Path):
    """按覆盖 path 的最长前缀 a 把 path 从 a 换到 b；无覆盖返回 None。"""
    hits = [(a, b) for a, b in pairs if a == path or a in path.parents]
    if not hits:
        return None
    a, b = max(hits, key=lambda ab: len(ab[0].parts))
    return b / path.relative_to(a)


# ---- 资源 sha256：URL 走 ~/.cache/flagrt-baseline 下载缓存 ----

def _content_length(url: str):
    proc = subprocess.run(["curl", "-sIL", "--max-time", "120", url],
                          capture_output=True, text=True)
    sizes = re.findall(r"(?im)^content-length:\s*(\d+)", proc.stdout)
    return int(sizes[-1]) if proc.returncode == 0 and sizes else None


def cached_download_sha256(url: str) -> str:
    cache = cache_root() / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}-{url.rsplit('/', 1)[-1]}"
    side = Path(f"{target}.sha256")
    if side.is_file():
        txt = side.read_text(encoding="utf-8").split()
        if txt and re.fullmatch(r"[0-9a-f]{64}", txt[0]):
            return txt[0]
    if not target.is_file():
        print(f"[INFO] 下载资源以计算 sha256（缓存在 {target}）: {url}")
        part = Path(f"{target}.part")
        expect = _content_length(url)
        if part.is_file() and expect and part.stat().st_size > expect:
            part.unlink()  # 断点续传写坏的残留
        proc = subprocess.run(["curl", "-fL", "-C", "-", "--retry", "5", "--retry-all-errors",
                               "--retry-delay", "5", "-o", str(part), url],
                              capture_output=True, text=True, timeout=36000)
        if proc.returncode != 0:
            raise BaselineError(f"下载失败，无法计算 sha256: {url}\n{proc.stderr.strip()}", code=1)
        got = part.stat().st_size
        if expect and got != expect:
            part.unlink()
            raise BaselineError(f"下载长度与 Content-Length 不一致（{got} != {expect}），"
                                f"已丢弃: {url}", code=1)
        part.rename(target)
    digest = sha256_file(target)
    side.write_text(digest + "\n", encoding="utf-8")
    return digest


# ---- info 块（只供阅读，不参与校验） ----

def _git_info(repo_dir: Path, commit: str) -> dict:
    """branch（优先 master/main，超过 3 个只列前三 + 总数）与 committed_at。"""
    out, _ = run_git("branch", "-r", "--contains", commit, cwd=repo_dir)
    names = {re.sub(r"^[^/]+/", "", s.strip()) for s in (out or "").splitlines()
             if s.strip() and "->" not in s}
    if not names:
        out, _ = run_git("branch", "--contains", commit, cwd=repo_dir)
        names = {s.strip().lstrip("* ").strip() for s in (out or "").splitlines()}
        names = {n for n in names if n and not n.startswith("(")}
    names = sorted(names)
    info = {}
    if names:
        pri = next((n for n in ("master", "main") if n in names), None)
        ordered = ([pri] if pri else []) + [n for n in names if n != pri]
        info["branch"] = ", ".join(ordered[:3]) + (
            f" 等共 {len(ordered)} 个分支" if len(ordered) > 3 else "")
    at, _ = run_git("show", "-s", "--format=%ci", commit, cwd=repo_dir)
    if at:
        info["committed_at"] = at
    return info


def _has_commit(repo_dir: Path, commit: str) -> bool:
    return run_git("cat-file", "-e", commit + "^{commit}", cwd=repo_dir)[0] is not None


def repo_info(name: str, commit: str, repo_url=None, branch_hint=None):
    """优先宿主 checkout，其次按 repo_url 浅取到缓存裸仓库；取不到返回 None（unavailable）。"""
    if not commit or not SHA40.fullmatch(commit):
        return None
    info = None
    host = workspace_root() / REPO_DIR.get(name, name)
    if (host / ".git").exists() and _has_commit(host, commit):
        info = _git_info(host, commit)
    elif repo_url:
        bare = cache_root() / "repos" / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.git"
        if not (bare / "HEAD").exists():
            bare.mkdir(parents=True, exist_ok=True)
            run_git("init", "--bare", "-q", cwd=bare)
        if not _has_commit(bare, commit):
            print(f"[INFO] 本机未检出 {name}，从 {repo_url} 浅取 {commit[:12]} 以读取提交元数据")
            out, err = run_git("fetch", "--depth", "1", repo_url, commit, cwd=bare, timeout=900)
            if out is None:
                print(f"[WARN] 浅取失败，{name} 的 info 标为 unavailable: {err}")
        if _has_commit(bare, commit):
            info = _git_info(bare, commit)
    info = info or {}
    if "branch" not in info and branch_hint:
        info = {"branch": branch_hint, **info}
    return info or None


# ---- Dockerfile 解析（方案 §4.2：只识别列出的模式，其余一律报错） ----

KNOWN_OPS = {"FROM", "ARG", "LABEL", "ENV", "RUN", "COPY", "SHELL", "WORKDIR", "USER", "EXPOSE",
             "ENTRYPOINT", "CMD", "VOLUME", "STOPSIGNAL", "HEALTHCHECK", "MAINTAINER"}


def parse_instructions(text: str):
    lines, cur = [], ""
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("#") or (not cur and not s):
            continue
        if s.endswith("\\"):
            cur += s[:-1].strip() + " "
            continue
        lines.append((cur + s).strip())
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
    """展开 ${X} 与 ${X:start:len}；未定义的保持原样。"""
    def _sub(m):
        name, mod = m.group(1), m.group(2) or ""
        if name not in args:
            return m.group(0)
        ms = re.fullmatch(r":(\d+):(\d+)", mod)
        val = args[name]
        return val[int(ms.group(1)):int(ms.group(1)) + int(ms.group(2))] if ms else val

    for _ in range(5):
        new = re.sub(r"\$\{(\w+)(:[^}]*)?\}", _sub, s)
        if new == s:
            break
        s = new
    return s


def _segments(body: str):
    return [s.strip() for s in re.split(r"&&|\|\||;|\n", body) if s.strip()]


def _repo_name(url: str) -> str:
    return re.sub(r"\.git$", "", url.rsplit("/", 1)[-1])


class _Recipe:
    """单个 Dockerfile 的固化判定结果。"""

    def __init__(self, image_dir: Path):
        self.image_dir = image_dir
        self.args, self.labels, self.from_ref = {}, {}, None
        self.repos = {}       # name -> 判定信息（按出现顺序）
        self.dir_repo = {}    # 镜像内目录 -> repo 名
        self.resources = []
        self.patch_dirs = {}  # 镜像内 patches 目录 -> assets 相对路径
        self.warnings = []

    def add_repo(self, name, **kw):
        if name in self.repos:
            raise BaselineError(f"同一 repo 在本层出现两次，无法判定: {name}")
        self.repos[name] = {"name": name, **kw}
        if kw.get("dir"):
            self.dir_repo[kw["dir"]] = name
        return self.repos[name]

    def copy(self, src: str, dest: str):
        s, dest = src.lstrip("./"), dest.rstrip("/")
        if s.startswith("src/"):
            parts = [p for p in s.split("/") if p]
            if len(parts) != 2:
                raise BaselineError(f"COPY 源码路径不符合 src/<Repo>/ 模式: {src}")
            self.add_repo(parts[1], source="copy", dir=dest)
        elif s.startswith("assets/patches/"):
            self.patch_dirs[dest] = s
        elif s.startswith("assets/"):
            local = self.image_dir / s
            if src.endswith("/") or local.is_dir():
                if not local.is_dir():
                    raise BaselineError(f"COPY 的 assets 目录不存在: {local}")
                files = sorted(f for f in local.rglob("*") if f.is_file())
            elif local.is_file():
                files = [local]
            else:
                raise BaselineError(f"COPY 的 assets 文件不存在: {local}")
            self.resources += [{"name": f.name, "kind": "asset", "path": rel_to_repo(f)}
                               for f in files]
        else:
            raise BaselineError(
                f"COPY 来源不在 §4.2 识别范围内（只识别 src/<Repo>/ 与 assets/…）: {src}")

    def run(self, raw: str):
        """raw 未展开 ARG（用于识别 ${<X>_COMMIT|REF|URL}），body 为展开后的实际值。"""
        body = expand(raw, self.args)
        if re.search(r"pip\s+install\b[^&|;]*(\s-e\s|\s--editable\b)", body):
            raise BaselineError("本构建层出现 pip install -e：固化资产不允许 editable 安装（方案 §2.1）")
        if re.search(r"\bwget\b", body):
            raise BaselineError("RUN 中出现 wget，§4.2 只识别 curl ... ${<X>_URL}")

        # git remote add origin <url> + git fetch ${<X>_COMMIT|REF} 或 40 位 commit
        m_remote = re.search(r"git\s+remote\s+add\s+origin\s+(\S+)", body)
        if m_remote:
            url = m_remote.group(1)
            name = _repo_name(url)
            m_fetch = re.search(r"git\s+fetch\b([^&|;]*)", raw)
            if not m_fetch:
                raise BaselineError(f"{name}: 有 git remote add 但没有 git fetch，无法判定 commit 来源")
            m_arg = re.search(r"\$\{(\w+)\}", m_fetch.group(1))
            arg_name = m_arg.group(1) if m_arg else None
            if arg_name:
                ref_value = self.args.get(arg_name)
                if not ref_value:
                    raise BaselineError(f"{name}: ARG {arg_name} 没有默认值，无法确定 commit")
            else:
                m_sha = re.search(r"\b([0-9a-f]{40})\b", m_fetch.group(1))
                if not m_sha:
                    raise BaselineError(f"{name}: git fetch 未固定到 ARG 或 40 位 commit，拒绝猜测（§4.2）")
                ref_value = m_sha.group(1)
            m_branch = re.search(r"git\s+checkout[^&|;]*-b\s+(\S+)", body)
            m_dir = re.search(r"\bcd\s+(\S+)", body)
            self.add_repo(name, source="fetch", url=url, arg=arg_name, arg_value=ref_value,
                          branch_hint=m_branch.group(1) if m_branch else None,
                          dir=m_dir.group(1).rstrip("/") if m_dir else None)

        # 未固定 commit 的 git clone（FlagCX 公开 submodule）：记录 URL + 警告，不猜 commit
        for m in re.finditer(r"git\s+clone\b([^&|;]*)", body):
            toks, skip = [], False
            for t in m.group(1).split():
                if skip:
                    skip = False
                elif t.startswith("-"):
                    skip = t in ("--depth", "-b", "--branch", "-c", "--config", "-o", "--origin")
                else:
                    toks.append(t)
            if not toks:
                raise BaselineError(f"无法解析的 git clone: {m.group(0)}")
            name = _repo_name(toks[0])
            self.resources.append({"name": name, "kind": "git-clone-floating", "url": toks[0],
                                   "path_in_image": toks[1] if len(toks) > 1 else name})
            self.warnings.append(f"{name}: 构建期 git clone 未固定 commit（{toks[0]}），只能记录 URL，"
                                 "版本事实由 image id 兜底")

        # curl ... ${<X>_URL}
        for seg in _segments(raw):
            if not re.search(r"\bcurl\b", seg):
                continue
            m_url = re.search(r"\$\{(\w+_URL)\}", seg)
            if not m_url:
                raise BaselineError(
                    f"RUN 中的 curl 未使用 ${{<X>_URL}} 形式的 ARG，拒绝猜测（§4.2）: {seg[:80]}")
            url = self.args.get(m_url.group(1))
            if not url:
                raise BaselineError(f"ARG {m_url.group(1)} 没有默认值，无法确定下载来源")
            if not any(r.get("url") == url for r in self.resources):
                name = m_url.group(1)[:-len("_URL")].lower().replace("_", "-")
                self.resources.append({"name": name, "kind": "url", "url": url})

        # 安装侧：pip install <已登记目录|.>
        cd_dir = None
        for seg in _segments(body):
            m_cd = re.search(r"\bcd\s+(\S+)", seg)
            if m_cd and "pip" not in seg:
                cd_dir = m_cd.group(1).rstrip("/")
            if not re.search(r"pip\s+install\b", seg):
                continue
            toks = shlex.split(seg)
            i = next((i for i, t in enumerate(toks)
                      if t == "install" and i and toks[i - 1].endswith("pip")), None)
            if i is None:
                raise BaselineError(f"无法解析的 pip install: {seg[:80]}")
            targets = [t for t in toks[i + 1:] if not t.startswith("-")]
            if len(targets) != 1:
                raise BaselineError(f"pip install 的目标不唯一，拒绝猜测: {seg[:80]}")
            d = cd_dir if targets[0] in (".", "./") else targets[0].rstrip("/")
            if d not in self.dir_repo:
                raise BaselineError(f"pip install 的来源未登记为固化 repo（§4.2 只识别 COPY src/<Repo>/ 与"
                                    f" 构建期 git fetch 两种来源）: {seg[:80]}")
            repo = self.repos[self.dir_repo[d]]
            # 只记录 USE_* 构建开关（偏离已接受）
            use = [t for t in toks[:i] if re.match(r"^USE_[A-Z0-9_]*=", t)]
            repo["install"] = f"pip (non-editable{', ' + ' '.join(use) if use else ''})"
            # 同一条 RUN 中的 git apply 补丁归属该 repo
            for m in re.finditer(r"git\s+apply\b([^&|;]*)", body):
                for tok in m.group(1).split():
                    if tok.startswith("-"):
                        continue
                    src_dir = self.patch_dirs.get(tok.rsplit("/", 1)[0].rstrip("/"))
                    if src_dir is None:
                        raise BaselineError(f"git apply 的补丁目录不是 COPY 进来的 assets: {tok}")
                    repo["patch_src"] = src_dir


def analyze_dockerfile(image_dir) -> _Recipe:
    image_dir = Path(image_dir).resolve()
    df = image_dir / "Dockerfile.repro"
    if not df.is_file():
        raise BaselineError(f"未找到 Dockerfile.repro: {df}")
    r = _Recipe(image_dir)
    for op, body in parse_instructions(df.read_text(encoding="utf-8")):
        if op == "ARG":
            m = re.match(r"^(\w+)\s*=\s*(.*)$", body)
            if m:
                r.args[m.group(1)] = expand(m.group(2).strip().strip('"\''), r.args)
            else:
                r.args.setdefault(body.strip(), "")
        elif op == "FROM":
            r.from_ref = expand(body.split()[0], r.args)
        elif op == "LABEL":
            for k, v in re.findall(r'([\w.\-]+)\s*=\s*("(?:[^"\\]|\\.)*"|\S+)', body):
                r.labels[k] = expand(v[1:-1].replace('\\"', '"') if v.startswith('"') else v, r.args)
        elif op == "COPY":
            toks = [t for t in shlex.split(expand(body, r.args)) if not t.startswith("--")]
            if len(toks) < 2:
                raise BaselineError(f"COPY 操作数不足: {body}")
            for src in toks[:-1]:
                r.copy(src, toks[-1])
        elif op == "RUN":
            r.run(body)
    for d, name in r.dir_repo.items():
        if not r.repos[name].get("install"):
            raise BaselineError(
                f"{name}: 源码进入镜像（{d}）但未找到对应的 pip install，无法判定固化方式（§4.2）")
    return r


# ---- 镜像基线 export / check ----

def image_baselines():
    return sorted((REPO_ROOT / "dev" / "images").glob("*/v*/image.baseline.yaml"))


def _parent_baseline(ref: str):
    for p in image_baselines():
        try:
            data = load_yaml(p)
        except BaselineError:
            continue
        if (data.get("image") or {}).get("ref") == ref:
            return p, data
    return None, None


def _external_base(recipe: _Recipe, warnings: list):
    """外部基座：无设备读取 pip 元数据补全 L1，镜像内 editable 安装记为固化资产。"""
    base_info = inspect_image(recipe.from_ref)
    digest = next((rd.split("@", 1)[1] for rd in base_info.get("RepoDigests") or []
                   if "@" in rd), None)
    python_bin = recipe.args.get("PYTHON_BIN")
    if not python_bin:
        raise BaselineError("Dockerfile 未声明 ARG PYTHON_BIN，无法读取基座 pip 元数据")
    report = image_python_report(recipe.from_ref, python_bin)
    provides = {}
    for e in (base_info.get("Config") or {}).get("Env") or []:
        m = re.match(r"ASCEND_TOOLKIT_HOME=.*/cann-([0-9.]+)$", e)
        if m:
            provides["cann"] = m.group(1)
    if report["python_version"]:
        provides["python"] = report["python_version"]
    for pkg in ("torch", "torch_npu", "vllm"):
        if report["versions"].get(pkg):
            provides[pkg] = report["versions"][pkg].replace("+empty", "")
    repos = []
    for name, e in sorted(report["editables"].items()):
        src = e.get("source", "")
        if not src.startswith("file:/"):
            warnings.append(f"{name}: 父镜像的 editable 安装源码不在镜像内（{src}），不记为固化资产")
            continue
        repos.append({"_name": name, "layer": LAYER.get(name, DEFAULT_LAYER), "from": "base_image",
                      "commit": e.get("commit"),
                      "source_path": re.sub(r"^file://", "", src),
                      "install": "editable-in-image",
                      "info": repo_info(name, e.get("commit"), repo_url=e.get("repo_url"))})
        warnings.append(f"{name}: 继承自父镜像的 editable 安装，源码位于镜像内 {src}，"
                        "记为固化资产（install: editable-in-image，方案 §2.1）")
    return {"ref": recipe.from_ref, "digest": digest, "provides": provides}, repos


def export_image(image_dir, demo=None, image_ref=None) -> int:
    image_dir = Path(image_dir).resolve()
    recipe = analyze_dockerfile(image_dir)
    ref = image_ref
    if not ref and (image_dir / "build.sh").is_file():
        tags = re.findall(r"-t\s+(\S+)", (image_dir / "build.sh").read_text(encoding="utf-8"))
        ref = tags[-1] if tags else None
    if not ref:
        raise BaselineError(f"无法确定镜像 ref（build.sh 里没有 -t，可用 --image-ref 指定）: {image_dir}")
    info = inspect_image(ref)
    img_labels = (info.get("Config") or {}).get("Labels") or {}
    warnings = list(recipe.warnings)

    # Dockerfile LABEL（ARG 展开后）与镜像标签交叉核对
    for key, val in recipe.labels.items():
        if key in COMMIT_LABEL.values():
            if key not in img_labels:
                raise BaselineError(f"镜像缺少标签 {key}（Dockerfile 声明为 {val}）", code=1)
            if img_labels[key] != val:
                raise BaselineError(f"ARG/标签交叉核对失败: {key} Dockerfile={val} 镜像={img_labels[key]}",
                                    code=1)

    parent_path, parent = _parent_baseline(recipe.from_ref) if recipe.from_ref else (None, None)
    if parent:
        # 父镜像是本仓自建镜像：展开继承，层级与来源层原样保留
        pimg = parent.get("image") or {}
        try:
            local_id = inspect_image(recipe.from_ref).get("Id")
        except BaselineError as e:
            raise BaselineError(f"父镜像不可用: {e}", code=1)
        if pimg.get("id") and local_id != pimg["id"]:
            raise BaselineError(f"父镜像 id 与其基线不一致: 基线 {pimg['id']} 本机 {local_id}"
                                f"（{rel_to_repo(parent_path)}）", code=1)
        base_block = parent.get("base_image")
        baked_repos = [{"_name": n, **item} for n, item in (parent.get("baked_repos") or {}).items()]
        baked_resources = [dict(r) for r in parent.get("baked_resources") or []]
        recipe_files = list(pimg.get("recipe") or [])
    elif recipe.from_ref:
        base_block, baked_repos = _external_base(recipe, warnings)
        baked_resources, recipe_files = [], []
    else:
        base_block, baked_repos, baked_resources, recipe_files = None, [], [], []
    recipe_files.append(rel_to_repo(image_dir / "Dockerfile.repro"))

    for item in recipe.repos.values():
        name = item["name"]
        label_key = COMMIT_LABEL.get(name)
        if not label_key:
            raise BaselineError(f"{name}: 未登记 commit 标签（请在 baseline.py 的 COMMIT_LABEL 里登记）")
        commit = img_labels.get(label_key)
        if not commit:
            raise BaselineError(f"{name}: 镜像缺少 {label_key} 标签，无法确定 commit", code=1)
        if item.get("arg_value") and item["arg_value"] != commit:
            raise BaselineError(f"{name}: ARG {item['arg']}={item['arg_value']} 与镜像标签 "
                                f"{label_key}={commit} 不一致", code=1)
        kind = "COPY 构建上下文" if item["source"] == "copy" else "构建期 git fetch"
        entry = {"_name": name, "layer": LAYER.get(name, DEFAULT_LAYER), "commit": commit}
        if item.get("patch_src"):
            pdir = image_dir / item["patch_src"]
            entry["patches"] = [{"file": pf.name, "sha256": sha256_file(pf)}
                                for pf in sorted(pdir.glob("*.patch"))]
            if not entry["patches"]:
                raise BaselineError(f"{name}: 补丁目录内没有 .patch 文件: {pdir}")
            kind += " + 补丁"
        entry["from"] = f"{image_dir.parent.name}/{image_dir.name} Dockerfile.repro({kind})"
        entry["install"] = item["install"]
        bl = BRANCH_LABEL.get(name, "")
        entry["info"] = repo_info(name, commit, repo_url=item.get("url"),
                                  branch_hint=item.get("branch_hint") or recipe.labels.get(bl)
                                  or img_labels.get(bl))
        baked_repos.append(entry)

    for res in recipe.resources:
        entry = {"name": res["name"], "layer": LAYER.get(res["name"], DEFAULT_LAYER)}
        if res["kind"] == "asset":
            entry.update(path=res["path"], sha256=sha256_file(REPO_ROOT / res["path"]))
        elif res["kind"] == "url":
            entry.update(url=res["url"], sha256=cached_download_sha256(res["url"]))
        else:
            entry.update(url=res["url"], path_in_image=res["path_in_image"],
                         pin="unpinned (构建期 git clone --depth 1)")
        baked_resources.append(entry)

    # 完整包清单：assets/provenance/pipfreeze-*.txt，优先带 npu 的最新一份
    freezes = sorted((image_dir / "assets" / "provenance").glob("pipfreeze-*.txt"))
    packages = rel_to_repo(max([f for f in freezes if "npu" in f.name] or freezes,
                               key=lambda f: f.name)) if freezes else None

    L = [f"schema: {SCHEMA}", "kind: image",
         f"generated_by: dev/lib/baseline.py export --image {rel_to_repo(image_dir)}",
         f"generated_at: {now_iso()}", "", "image:",
         f"  ref: {y(ref)}",
         f"  id: {info.get('Id')}                 # check 项",
         f"  recipe: {y(recipe_files)}"]
    if demo:
        L.append(f"  demo: {y(demo)}")
    if base_block:
        L += ["", "base_image:                                    # L1",
              f"  ref: {y(base_block.get('ref'))}",
              f"  digest: {y(base_block.get('digest'))}",
              f"  provides: {y(base_block.get('provides') or {})}"]
    L += ["", "baked_repos: {}" if not baked_repos else "baked_repos:"]
    for e in baked_repos:
        e = dict(e)
        L.append(f"  {e.pop('_name')}:")
        L.append(f"    layer: {e.pop('layer', DEFAULT_LAYER)}")
        L.append(f"    from: {y(e.pop('from', 'unknown'))}")
        L.append(f"    commit: {y(e.pop('commit', None))}")
        for k in ("patches", "source_path"):
            if e.get(k):
                L.append(f"    {k}: {y(e.pop(k))}")
        L.append(f"    install: {y(e.pop('install', None))}")
        info_v = e.pop("info", None)
        L.append(f"    info: {y(info_v) if info_v else 'unavailable'}")
        L += [f"    {k}: {y(v)}" for k, v in e.items()]
    L += ["", "baked_resources: []" if not baked_resources else "baked_resources:"]
    L += [f"  - {y(r)}" for r in baked_resources]
    if packages:
        L += ["", f"packages: {packages}   # 完整包清单，仅供查阅"]
    out = image_dir / "image.baseline.yaml"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    for w in warnings:
        print(f"[WARN] {w}")
    print(f"[OK] 镜像基线已生成 -> {rel_to_repo(out)}")
    return 0


def _check_image_id(ref, want_id):
    """返回 (报告行, 镜像标签；镜像不可用时为 None)。"""
    try:
        info = inspect_image(ref)
    except BaselineError as e:
        return ("FAIL", "image", str(e)), None
    labels = (info.get("Config") or {}).get("Labels") or {}
    if info.get("Id") != want_id:
        return ("FAIL", "image", f"id 不一致: 基线 {want_id} 本机 {info.get('Id')} (ref={ref})"), labels
    return ("PASS", "image", f"{ref} -> {want_id}"), labels


def _print_report(report) -> bool:
    for status, item, msg in report:
        print(f"[{status}] {item}: {msg}")
    return not any(s == "FAIL" for s, _, _ in report)


def check_image(image_dir) -> int:
    data = load_yaml(Path(image_dir).resolve() / "image.baseline.yaml")
    img = data.get("image") or {}
    line, labels = _check_image_id(img.get("ref"), img.get("id"))
    report = [line]
    baked = (data.get("baked_repos") or {}) if labels is not None else {}
    for name, item in baked.items():
        key = COMMIT_LABEL.get(name)
        if key not in labels:
            report.append(("INFO", name, "镜像无对应 commit 标签，跳过（由 image id 兜底）"))
        elif labels[key] != item.get("commit"):
            report.append(("FAIL", name, f"标签 {key} 不一致: 基线 {item.get('commit')} 镜像 {labels[key]}"))
        else:
            report.append(("PASS", name, f"{key}={labels[key][:12]}"))
    return 0 if _print_report(report) else 1


# ---- 运行环境场景 ----

def _env(env_dir):
    """读取 <env-dir>/docker-compose.yml 的 x-flagrt 声明。"""
    env_dir = Path(env_dir).resolve()
    compose = env_dir / "docker-compose.yml"
    if not compose.is_file():
        raise BaselineError(f"运行环境目录缺少 docker-compose.yml: {env_dir}")
    x = load_yaml(compose).get("x-flagrt")
    if not isinstance(x, dict):
        raise BaselineError(f"{rel_to_repo(compose)} 缺少顶层 x-flagrt 声明（方案 §3.2）")
    if not x.get("image_baseline"):
        raise BaselineError("x-flagrt 缺少 image_baseline")
    mounted = x.get("mounted_repos") or {}
    if not isinstance(mounted, dict):
        raise BaselineError("x-flagrt.mounted_repos 不是映射")
    return env_dir, compose, x["image_baseline"], mounted


def _host_head(host: Path):
    """返回 (HEAD, 是否 dirty, 错误信息)。"""
    if not (host / ".git").exists():
        return None, False, f"宿主 repo 不存在或不是 git 仓库: {host}"
    head, err = run_git("rev-parse", "HEAD", cwd=host)
    if head is None:
        return None, False, f"读取 HEAD 失败: {err}"
    return head, bool(run_git("status", "--porcelain", cwd=host)[0]), None


def export_env(env_dir, require_clean=False) -> int:
    env_dir, compose, image_baseline_rel, declared = _env(env_dir)
    image_baseline = load_yaml(env_dir / image_baseline_rel)
    baked = image_baseline.get("baked_repos") or {}
    binds = compose_binds(compose) if declared else []
    mounted, dirty = {}, []
    for name, spec in declared.items():
        spec = spec or {}
        host_rel = spec.get("host_path") or REPO_DIR.get(name, name)
        host = (workspace_root() / host_rel).resolve()
        head, is_dirty, err = _host_head(host)
        if err:
            raise BaselineError(f"{name}: {err}")
        container = spec.get("container_path") or _translate(binds, host)
        if not container:
            raise BaselineError(f"{name}: docker-compose.yml 中没有覆盖宿主目录 {host} 的挂载")
        entry = {"host_path": host_rel, "container_path": str(container),
                 "activation": spec.get("activation", "unknown"),
                 "commit": head, "dirty": is_dirty}
        if name in baked:
            entry["overrides_baked"] = True
            entry["baked_commit"] = baked[name].get("commit")
            if baked[name].get("patches"):
                entry["baked_patches"] = baked[name]["patches"]
        info = repo_info(name, head)
        if info:
            entry["info"] = info
        mounted[name] = entry
        if is_dirty:
            dirty.append(name)
    if dirty and require_clean:
        print(f"[FAIL] --require-clean: 以下挂载 repo 工作区不干净，拒绝导出: {', '.join(dirty)}",
              file=sys.stderr)
        return 1
    for name in dirty:
        print(f"[WARN] {name}: 工作区不干净，已记录 dirty: true")

    img = image_baseline.get("image") or {}
    L = [f"schema: {SCHEMA}", "kind: container",
         f"generated_by: dev/lib/baseline.py export {rel_to_repo(env_dir)}",
         f"generated_at: {now_iso()}", "compose: docker-compose.yml", "",
         "image:                       # 抄录自 image.baseline.yaml；check 只核对本地镜像 id",
         f"  ref: {y(img.get('ref'))}", f"  id: {y(img.get('id'))}",
         f"  baseline: {y(image_baseline_rel)}", ""]
    for key, items, comment in (
            ("baked_repos", baked, "                 # 原样抄录自镜像基线，只读；check 与 load 均不处理"),
            ("mounted_repos", mounted, "               # check 与 load 的对象")):
        L.append(f"{key}:{' {}' if not items else ''}{comment}")
        for name, item in items.items():
            L.append(f"  {name}:")
            L += [f"    {k}: {'unavailable' if v is None else y(v)}" for k, v in item.items()]
        L.append("")
    out = env_dir / "env.baseline.yaml"
    out.write_text("\n".join(L[:-1]) + "\n", encoding="utf-8")
    print(f"[OK] 运行环境基线已生成 -> {rel_to_repo(out)}")
    return 0


def check_env(env_dir) -> int:
    env_dir, compose, _, _ = _env(env_dir)
    base = load_yaml(env_dir / "env.baseline.yaml")
    img = base.get("image") or {}
    report = [_check_image_id(img.get("ref"), img.get("id"))[0]]
    mounted = base.get("mounted_repos") or {}
    # 容器实际挂载的目录必须就是本次核对的宿主目录（防 WORKSPACE_ROOT 分叉）
    binds = [(t, s) for s, t in compose_binds(compose)] if mounted else []
    for name, item in mounted.items():
        host = (workspace_root() / item.get("host_path", name)).resolve()
        src = _translate(binds, Path(item.get("container_path", "")))
        if src != host:
            report.append(("FAIL", name, f"容器 {item.get('container_path')} 的挂载来源为 {src}，"
                                         f"与核对的宿主目录 {host} 不一致"
                                         "（检查 WORKSPACE_ROOT / FLAGRT_WORKSPACE_ROOT）"))
        head, dirty, err = _host_head(host)
        if err:
            report.append(("FAIL", name, err))
            continue
        if head != item.get("commit"):
            report.append(("FAIL", name, f"commit 不一致: 基线 {str(item.get('commit'))[:12]} "
                                         f"实际 {head[:12]}"))
        else:
            report.append(("PASS", name, f"commit={head[:12]}"))
        if dirty:
            report.append(("WARN", name, "工作区有未提交改动（不阻断）"))
        if item.get("overrides_baked"):
            report.append(("INFO", name, f"覆盖镜像内固化版本 {str(item.get('baked_commit'))[:12]}，"
                                         f"生效方式 {item.get('activation')}"))
        if item.get("baked_patches"):
            report.append(("WARN", name, "挂载副本不含镜像内固化的补丁: "
                           + ", ".join(p.get("file", "?") for p in item["baked_patches"])))
    if base.get("baked_repos"):
        report.append(("INFO", "baked", "固化资产不与宿主比对，由 image id 约束（方案 §2）"))
    ok = _print_report(report)
    out = env_dir / ".last-check.json"
    out.write_text(json.dumps({
        "checked_at": now_iso(), "env_dir": rel_to_repo(env_dir), "baseline": "env.baseline.yaml",
        "report": [{"status": s, "item": i, "detail": m} for s, i, m in report],
        "result": "PASS" if ok else "FAIL",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"check 结果 -> {rel_to_repo(out)}")
    return 0 if ok else 1


def load_env(env_dir) -> int:
    env_dir = _env(env_dir)[0]
    repos = load_yaml(env_dir / "env.baseline.yaml").get("mounted_repos") or {}
    if not repos:
        print("[INFO] 基线没有声明挂载 repo，load 无事可做")
        return 0
    plan, dirty = [], []
    for name, item in repos.items():
        host = (workspace_root() / item.get("host_path", name)).resolve()
        head, is_dirty, err = _host_head(host)
        if err:
            print(f"[FAIL] {name}: {err}", file=sys.stderr)
            return 3
        if not item.get("commit"):
            print(f"[FAIL] {name}: 基线没有 commit，无法切换", file=sys.stderr)
            return 2
        if is_dirty:
            dirty.append(name)
        plan.append((name, host, item["commit"], head))
    if dirty:
        print("[ABORT] 以下挂载 repo 工作区不干净，整体中止，未做任何切换（方案 §4.3）: "
              + ", ".join(dirty), file=sys.stderr)
        return 3

    for name, host, want, head in plan:
        branch, _ = run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=host)
        print(f"[INFO] {name}: 切换前 commit={head[:12]} branch={branch}（回退用）")
        if not _has_commit(host, want):
            print(f"[INFO] {name}: 本机缺少 {want[:12]}，先 git fetch")
            out, err = run_git("fetch", "--all", "--tags", cwd=host, timeout=900)
            if out is None:
                print(f"[FAIL] {name}: git fetch 失败: {err}", file=sys.stderr)
                return 1
        out, err = run_git("checkout", "--detach", want, cwd=host, timeout=300)
        if out is None:
            print(f"[FAIL] {name}: checkout {want[:12]} 失败: {err}", file=sys.stderr)
            return 1
        # checkout 成功返回不代表检出完整（如部分克隆的克隆缺对象，会留下被删的跟踪文件）
        new, _ = run_git("rev-parse", "HEAD", cwd=host)
        status, _ = run_git("status", "--porcelain", cwd=host)
        if new != want or status != "":
            print(f"[FAIL] {name}: 切换后检出不完整（HEAD={str(new)[:12]}，期望 {want[:12]}；"
                  f"git status 有 {len((status or '').splitlines())} 项改动），请检查 {host}",
                  file=sys.stderr)
            return 1
        print(f"[OK] {name}: 切换后 commit={new[:12]} (detached HEAD)")
    return 0


# ---- show / list ----

def _dw(s: str) -> int:
    """显示宽度：CJK 全角字符占两列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _table(rows, headers):
    rows = [["" if c is None else str(c) for c in r] for r in rows]
    widths = [max([_dw(h)] + [_dw(r[i]) for r in rows]) for i, h in enumerate(headers)]

    def line(cells):
        return "  ".join(c + " " * (widths[i] - _dw(c)) for i, c in enumerate(cells)).rstrip()

    return "\n".join([line(headers), "  ".join("-" * w for w in widths)] + [line(r) for r in rows])


def _info_str(info):
    if not isinstance(info, dict):
        return "unavailable"
    return " / ".join(str(info[k]) for k in ("branch", "committed_at") if info.get(k))


def _short(sha, n=12):
    return str(sha)[:n] if sha else "unpinned"


def show(target, image: bool) -> int:
    d = Path(target).resolve()
    data = load_yaml(d / ("image.baseline.yaml" if image else "env.baseline.yaml"))
    img = data.get("image") or {}
    baked = data.get("baked_repos") or {}
    if image:
        print(f"镜像: {img.get('ref')}\nid:   {img.get('id')}")
        print(f"配方: {', '.join(img.get('recipe') or [])}")
        if img.get("demo"):
            print(f"demo: {img['demo']}")
        b = data.get("base_image") or {}
        if b:
            print(f"基座: {b.get('ref')}\n      digest={b.get('digest')} provides={b.get('provides')}")
        print("\n固化 repo（baked，由 image id 约束，不比对宿主）:")
        print(_table([[i.get("layer"), n, str(i.get("commit"))[:12], i.get("install"),
                       _info_str(i.get("info")), i.get("from")] for n, i in baked.items()],
                     ["层级", "资产", "commit", "安装方式", "info（只读）", "来源层"]))
        res = data.get("baked_resources") or []
        if res:
            print("\n固化资源（baked_resources）:")
            print(_table([[r.get("layer"), r.get("name"), _short(r.get("sha256"), 16),
                           r.get("url") or r.get("path")] for r in res],
                         ["层级", "名称", "sha256/pin", "来源"]))
        if data.get("packages"):
            print(f"\n完整包清单: {data['packages']}")
        return 0
    mounted = data.get("mounted_repos") or {}
    print(f"运行环境: {rel_to_repo(d)}\n镜像: {img.get('ref')}")
    print(f"id:   {img.get('id')}  (基线 {img.get('baseline')})\n")
    print("挂载 repo（mounted，check/load 的对象）:")
    print(_table([[n, str(i.get("commit"))[:12], i.get("activation"), "是" if i.get("dirty") else "否",
                   "是 <- " + str(i.get("baked_commit"))[:12] if i.get("overrides_baked") else "否",
                   i.get("container_path"), _info_str(i.get("info"))] for n, i in mounted.items()],
                 ["资产", "commit", "生效方式", "dirty", "覆盖固化", "容器内路径", "info（只读）"]))
    if baked:
        print("\n固化 repo（baked，只读抄录）:")
        print(_table([[i.get("layer"), n, str(i.get("commit"))[:12], i.get("install"),
                       "被挂载覆盖" if n in mounted else ""] for n, i in baked.items()],
                     ["层级", "资产", "commit", "安装方式", "备注"]))
    return 0


def list_baselines() -> int:
    rows = []
    for f in image_baselines():
        try:
            data = load_yaml(f)
        except BaselineError as e:
            print(f"[跳过] {rel_to_repo(f)}: {e}", file=sys.stderr)
            continue
        head = [f"{f.parent.parent.name}/{f.parent.name}",
                str((data.get("image") or {}).get("id", ""))[7:19]]
        items = [[i.get("layer"), n, str(i.get("commit"))[:12], i.get("install")]
                 for n, i in (data.get("baked_repos") or {}).items()]
        items += [[r.get("layer"), r.get("name"), _short(r.get("sha256")), "资源"]
                  for r in data.get("baked_resources") or []]
        for k, it in enumerate(items or [["", "(无固化资产)", "", ""]]):
            rows.append((head if k == 0 else ["", ""]) + it)
    if not rows:
        print("未找到任何 image.baseline.yaml")
        return 0
    print("仓内全部镜像基线（dev/images/*/v*/image.baseline.yaml）:")
    print(_table(rows, ["镜像", "image id", "层级", "资产", "版本/sha256", "安装方式"]))
    return 0


# ---- CLI ----

def build_parser():
    p = argparse.ArgumentParser(prog="baseline.py", description="环境基线管理（镜像基线 / 运行环境基线）。"
                                "规范见 docs/环境基线管理-改造实施方案-20261010.md")
    sub = p.add_subparsers(dest="action")
    for action, help_ in (("export", "生成基线文件"), ("check", "核对基线与本机实际"),
                          ("show", "只读打印基线")):
        sp = sub.add_parser(action, help=help_)
        sp.add_argument("--image", metavar="IMAGE_DIR", help="镜像场景：镜像配方目录")
        sp.add_argument("env_dir", nargs="?", help="运行环境场景：运行环境目录")
        if action == "export":
            sp.add_argument("--demo", help="镜像场景：demo 运行环境目录（写入 image.demo）")
            sp.add_argument("--image-ref", help="镜像场景：显式指定镜像 ref（默认取 build.sh 的 -t）")
            sp.add_argument("--require-clean", action="store_true",
                            help="运行环境场景：挂载 repo 不干净时拒绝导出")
    sub.add_parser("load", help="按基线切换挂载 repo（仅运行环境场景）").add_argument(
        "env_dir", help="运行环境目录")
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
        if bool(args.image) == bool(args.env_dir):
            print("用法错误: 必须且只能给一个场景（--image <image-dir> 或 <env-dir>）", file=sys.stderr)
            return 2
        if args.action == "export":
            return (export_image(args.image, demo=args.demo, image_ref=args.image_ref) if args.image
                    else export_env(args.env_dir, require_clean=args.require_clean))
        if args.action == "check":
            return check_image(args.image) if args.image else check_env(args.env_dir)
        return show(args.image or args.env_dir, image=bool(args.image))
    except BaselineError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
