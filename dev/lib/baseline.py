#!/usr/bin/env python3
"""环境基线管理：镜像基线（image.baseline.yaml）与运行环境基线（env.baseline.yaml）
的生成（export）、核对（check）、按基线切换宿主 repo（load）、只读查看（show/list）。
规范见 dev/ENV-SPEC.md（§3 文件形态、§4.2 Dockerfile 判定规则、§4.3 运行环境场景）。

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
import html
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
# 固化资产的版本锚以镜像标签为准，Dockerfile ARG / pip pin 只做交叉核对。
# 每个固化资产至少要有一个可机器核对的版本锚：commit 标签或 version 标签（ENV-SPEC §4.2）。
COMMIT_LABEL = {
    "FlagTree": "io.flagrt.flagtree.commit",
    "FlagGems": "io.flagrt.flag_gems.commit",
    "Torch-FL": "io.flagrt.torch_fl.commit",
    "vllm-plugin-FL": "io.flagrt.vllm_plugin_fl.ref",
    "FlagCX": "io.flagrt.flagcx.commit",
}
# 与 COMMIT_LABEL 平行的版本锚：没有仓库身份的 pip 索引发行包以 pip 发行名为键
VERSION_LABEL = {"flagtree": "io.flagrt.flagtree.version"}
BRANCH_LABEL = {"FlagTree": "io.flagrt.flagtree.branch"}
# PYTHON_BIN 指向的解释器版本声明；存在时与镜像内实际读数交叉核对（§4.2 模式 3）
PYTHON_VERSION_LABEL = "io.flagrt.python.version"
# pip 发行名 -> 仓库名。只登记确认是同一个仓的；不确定的保持 pip 发行名作键，
# 否则 REPO_DIR 会指向错误的宿主 checkout（例：/env/xvllm-plugin-FL 的远端是厂商
# 内网仓，与仓内 vllm-plugin-FL 不是同一个仓，故不登记）。
PKG_TO_REPO = {"flag_gems": "FlagGems", "flagcx": "FlagCX", "vllm": "vllm"}
# pip install 的「带取值选项」：解析目标时必须跳过它们的取值（否则 -i <url> 被当成目标）
PIP_OPTS_WITH_VALUE = {"-i", "--index-url", "--extra-index-url", "-f", "--find-links",
                       "-c", "--constraint", "-r", "--requirement", "--target", "-t",
                       "--prefix", "--root"}
PIP_INDEX_OPTS = {"-i", "--index-url"}
# L1 基座事实的探针，按芯片族扩展，不在代码里硬编码某一款芯片
BASE_PROVIDES_ENV = {        # 环境变量名 -> (provides 键, 取值正则)
    "ASCEND_TOOLKIT_HOME": ("cann", r".*/cann-([0-9.]+)$"),
}
BASE_PROVIDES_FILES = [      # 镜像内文件探针：KEY: VALUE 文件，取哪几个键
    {"path": "torch_xmlir/_versions.txt", "relative_to": "site-packages",
     "keys": {"XTDK": "xtdk", "XRE": "xre", "XCCL": "xccl"}},
]
BASE_PROVIDES_PKGS = ("torch", "torch_npu", "torchvision", "torchaudio", "vllm",
                      "xmlir", "xtorch_ops")
LOCAL_VERSION_STRIP = ("+empty",)   # 需要从包版本尾部剥掉的 local version 后缀
# 完整包清单（pip freeze）的优先标记，按顺序匹配文件名；各血统的命名惯例登记在这里。
# 都不命中时取名字最大的一份并告警。("npu" = 昇腾验收机快照，"repro" = 重建校验快照)
PACKAGES_PREFER = ("npu", "repro")
# 四层模型层级（与交付方式正交，ENV-SPEC §2）；未登记的按 L3
LAYER = {
    # flagtree 是 FlagTree 的 pip 索引发行物（提供 triton 编译器/运行时），与 FlagTree 同层
    "FlagTree": "L2", "flagtree": "L2", "FlagGems": "L2",
    "llvm-toolchain": "L2", "triton-build-deps": "L2",
    "patch_triton_ascend_flagtree.py": "L2", "FlagGems-DSA-__init__.py": "L2",
    "verify_runtime.py": "L2",
}
DEFAULT_LAYER = "L3"
SHA40 = re.compile(r"[0-9a-f]{40}")
# baked_repos 的 provenance 分组头（ENV-SPEC §3.1「怎么读镜像基线」维度 B）。划分依据就是
# 已有的 from 字段，不新增字段：`base_image` 是被动继承的现状，其余（含父层配方继承来的
# Dockerfile.repro(...)）是本组配方带入、可控的资产。风格同 env.baseline.yaml 的
# 「环境挂载（env）」分割线，缩进到 baked_repos 条目所在层级。
BAKED_GROUP_HEADERS = {
    "base": ["  # " + "─" * 16 + " 厂商基座自带（被动继承，不可重建） " + "─" * 16,
             "  # 本组配方未声明这些资产，只能记录其版本事实；要变更须按 §2.2 在运行环境挂载覆盖。"],
    "recipe": ["  # " + "─" * 16 + " 本组配方带入（可控，重建即可变更） " + "─" * 16,
               "  # 由 image.recipe 列出的 Dockerfile.repro 声明并固化（含父层配方继承来的）。"],
}


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


def image_python_report(ref: str, python_bin: str, probes=()) -> dict:
    """无设备、无挂载地读取镜像内 pip 元数据（关键包版本 + editable 安装）与文件探针。

    `probes` 是 BASE_PROVIDES_FILES 那样的声明式表，读出的内容放在返回值的 `base_files`
    里（文件不存在的探针不出现 —— 别的芯片族没有该文件，不是错误）。与 pip 元数据共用
    同一个 ref，但各自一次 docker run。"""
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
        # fragment 可带多个参数（#egg=flagcx&subdirectory=plugin/torch），egg 名只取 & 前的部分
        m = re.match(r"^-e\s+(\S+?)@([0-9a-f]{40})#(\S+)$", line.strip())
        if not m:
            continue
        frag = dict(kv.split("=", 1) for kv in m.group(3).split("&") if "=" in kv)
        if not frag.get("egg"):
            continue
        e = editables.setdefault(frag["egg"].replace("-", "_"), {"source": ""})
        e["commit"] = m.group(2)
        e["repo_url"] = re.sub(r"^git\+", "", m.group(1))
        if frag.get("subdirectory"):
            e["package_subdir"] = frag["subdirectory"]
    env = data.get("environment") or {}
    return {"python_version": env.get("python_full_version") or env.get("python_version"),
            "versions": versions, "editables": editables,
            "base_files": image_base_files(ref, python_bin, probes)}


def image_base_files(ref: str, python_bin: str, probes) -> dict:
    """一次 docker run 读齐镜像内的全部文件探针，返回 {probe path: 文件内容}。

    文件不存在的探针不出现在结果里（别的芯片族没有该文件，不是错误）。直接读文件而不
    import 任何厂商包：import 会往 stdout 打加载噪声（实测 torch_xmlir 即如此），
    故这里只取最后一行 JSON。"""
    want = [[p.get("path"), p.get("relative_to") or ""] for p in probes or [] if p.get("path")]
    if not want:
        return {}
    code = ("import json, os, sysconfig\n"
            "roots = {'': '', 'site-packages': sysconfig.get_paths()['purelib']}\n"
            "out = {}\n"
            "for path, rel in " + json.dumps(want) + ":\n"
            "    root = roots.get(rel)\n"
            "    if root is None:\n"
            "        continue\n"
            "    try:\n"
            "        with open(os.path.join(root, path), encoding='utf-8', errors='replace') as f:\n"
            "            out[path] = f.read()\n"
            "    except OSError:\n"
            "        pass\n"
            "print(json.dumps(out))\n")
    proc = subprocess.run(["docker", "run", "--rm", "--entrypoint", python_bin, ref, "-c", code],
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise BaselineError(f"读取镜像内文件探针失败: {ref}\n{proc.stderr.strip()}", code=1)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise BaselineError(f"读取镜像内文件探针无输出: {ref}", code=1)
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        raise BaselineError(f"文件探针输出不是 JSON: {ref}\n{e}", code=1)


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


# ---- 索引发行包的 wheel sha256：PEP 503 索引页的 `#sha256=` fragment ----
#
# 模式 6（ENV-SPEC §4.2）的资产以 version 为锚，但上游以同一版本号重发时版本锚察觉不到。
# 索引页按 PEP 503 在 href 上直接公布 `#sha256=`，一次小 HTTP 请求即可取，无需下载 wheel
# （flagtree 那个 wheel 是 3.3GB）。这里只取、只比对，不下载。
#
# 代理：**不在命令里硬编码 --noproxy**，走 curl 的标准环境变量。需要绕过代理的机器
# （实测本机访问 resource.flagos.net 即需要）在调用前 `export no_proxy='*'`。
INDEX_HTTP_TIMEOUT = 30


def _packaging():
    """软依赖 packaging：只有 wheel sha256 这条路径需要它，缺它不影响其余任何功能。

    baseline.py 的硬依赖只有 yaml（包版本的常规归一化由 _strip_local 自己处理）。
    wheel 文件名解析与 PEP 440 版本相等属规范细节，手搓容易错，故这一条按需引入。"""
    try:
        from packaging.utils import canonicalize_name, parse_wheel_filename
        from packaging.version import Version
    except ImportError as e:
        raise BaselineError(
            f"wheel sha256 核对需要 packaging 包（仅此功能需要，其余功能不依赖它）: {e}\n"
            f"安装：python3 -m pip install packaging", code=1)
    return canonicalize_name, parse_wheel_filename, Version


def index_wheel_entries(index_url: str, dist: str):
    """取 PEP 503 索引页，返回 [(wheel 文件名, sha256 或 None)]，只含 .whl 条目。"""
    canonicalize_name, _, _ = _packaging()
    url = f"{index_url.rstrip('/')}/{canonicalize_name(dist)}/"
    proc = subprocess.run(["curl", "-fsSL", "--max-time", str(INDEX_HTTP_TIMEOUT), url],
                          capture_output=True, text=True, timeout=INDEX_HTTP_TIMEOUT + 30)
    if proc.returncode != 0:
        raise BaselineError(
            f"取索引页失败: {url}\n{proc.stderr.strip()}\n"
            f"若本机需绕过代理，先 export no_proxy='*' 再重试", code=1)
    out = []
    for href in re.findall(r'href="([^"]+)"', proc.stdout):
        name = html.unescape(href).rsplit("/", 1)[-1]
        base, _, frag = name.partition("#")
        if not base.endswith(".whl"):
            continue
        m = re.search(r"sha256=([0-9a-f]{64})", frag)
        out.append((base, m.group(1) if m else None))
    if not out:
        raise BaselineError(f"索引页没有任何 .whl 条目，无法取 sha256: {url}", code=1)
    return out


def resolve_index_wheel(index_url: str, dist: str, version: str, tag=None) -> dict:
    """在索引页上定位 dist==version 的那个 wheel，返回 {filename, sha256}。

    **按解析匹配，不按字符串匹配**，两个坑都是实测到的（见 kunlun-operator-runtime/v2/
    lock.yaml known_issues: flagtree-wheel-sha256-in-check-chain）：
      · 子串匹配会误命中 —— 该索引上子串 `0.6.1` 命中 20 条，还有 `0.6.1a1+xpu3.6` 这种
        近邻干扰项，故按 PEP 440 的 Version 相等比较。
      · 不能「自己拼出文件名再比字符串」—— PEP 427 规定文件名的 version 段应把非字母
        数字串转义成 `_`，而该索引实际未转义（原样 `0.6.1+xpu3.6`），上游换构建工具即失配。
    `tag` 是镜像内 dist-info 的 `WHEEL: Tag`，用于同版本多 wheel 时消歧；命中唯一时用它
    反向核对，不一致即报错（说明装进镜像的那个 wheel 与索引现在的不是同一个）。
    """
    canonicalize_name, parse_wheel_filename, Version = _packaging()
    want_name, want_ver = canonicalize_name(dist), Version(version)
    hits = []
    for base, sha in index_wheel_entries(index_url, dist):
        try:
            n, v, _build, tags = parse_wheel_filename(base)
        except Exception:
            continue          # 非规范文件名不参与匹配，也不报错（索引上可能挂着别的东西）
        if canonicalize_name(str(n)) == want_name and v == want_ver:
            hits.append((base, sha, {str(t) for t in tags}))
    if not hits:
        raise BaselineError(f"索引页上没有 {dist}=={version} 的 wheel: {index_url}", code=1)
    if len(hits) > 1:
        if not tag:
            raise BaselineError(
                f"{dist}=={version} 在索引页上有 {len(hits)} 个 wheel 且无 tag 可消歧: "
                f"{', '.join(h[0] for h in hits)}", code=1)
        narrowed = [h for h in hits if tag in h[2]]
        if len(narrowed) != 1:
            raise BaselineError(
                f"{dist}=={version} 按 tag {tag} 消歧后仍有 {len(narrowed)} 个候选: "
                f"{', '.join(h[0] for h in hits)}", code=1)
        hits = narrowed
    base, sha, tags = hits[0]
    if not sha:
        raise BaselineError(f"索引页条目没有 #sha256= fragment，拿不到指纹: {base}", code=1)
    if tag and tag not in tags:
        raise BaselineError(
            f"{dist}=={version}: 镜像内 dist-info 的 WHEEL Tag={tag} 与索引页条目 {base} "
            f"的 tag {sorted(tags)} 不一致 —— 装进镜像的 wheel 与索引现在的不是同一个", code=1)
    return {"filename": base, "sha256": sha}


def image_wheel_tags(ref: str, python_bin: str, dists) -> dict:
    """读镜像内各 dist 的 dist-info/WHEEL 的 `Tag:` 行，返回 {dist: [tag, ...]}。

    wheel 文件名由「dist 名 + dist 版本 + tag」三段构成，前两段 pip 元数据已有，tag 只在
    dist-info 里 —— 这是把文件名从镜像内事实完整推出的最后一段。多条 Tag 行（压缩 tag 集）
    原样返回，由调用方决定报错还是消歧，不在这里猜。"""
    dists = sorted(set(dists))
    if not dists:
        return {}
    code = ("import json\n"
            "import importlib.metadata as m\n"
            "out = {}\n"
            "for d in " + json.dumps(dists) + ":\n"
            "    try:\n"
            "        dist = m.distribution(d)\n"
            "    except Exception:\n"
            "        continue\n"
            "    for f in dist.files or []:\n"
            "        if not str(f).endswith('.dist-info/WHEEL'):\n"
            "            continue\n"
            "        try:\n"
            "            text = dist.read_text(str(f))\n"
            "        except Exception:\n"
            "            text = None\n"
            "        if text is None:\n"
            "            break\n"
            "        out[d] = [ln.split(':', 1)[1].strip() for ln in text.splitlines()\n"
            "                  if ln.startswith('Tag:')]\n"
            "        break\n"
            "print(json.dumps(out))\n")
    proc = subprocess.run(["docker", "run", "--rm", "--entrypoint", python_bin, ref, "-c", code],
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise BaselineError(f"读取镜像内 dist-info 的 WHEEL Tag 失败: {ref}\n"
                            f"{proc.stderr.strip()}", code=1)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise BaselineError(f"读取镜像内 WHEEL Tag 无输出: {ref}", code=1)
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        raise BaselineError(f"WHEEL Tag 探针输出不是 JSON: {ref}\n{e}", code=1)


def image_git_heads(ref: str, python_bin: str, paths) -> dict:
    """读镜像内若干 git 工作树的 HEAD commit，返回 {路径: 40 位 sha}；取不到的键不出现。

    用途是给 `install: editable-in-image` 的继承资产一个**镜像自证**的第二版本锚：它们的
    commit 目前只来自基座 pip 元数据，而镜像标签只是「本组对基座内容的断言」、标签不存在
    时不核对（ENV-SPEC §3.1）。不要求镜像里有 git 可执行文件，纯读 .git 下的文件。"""
    paths = sorted(set(paths))
    if not paths:
        return {}
    code = ("import json, os, re\n"
            "def head(repo):\n"
            "    g = os.path.join(repo, '.git')\n"
            "    if os.path.isfile(g):\n"
            "        try:\n"
            "            line = open(g).read().strip()\n"
            "        except OSError:\n"
            "            return None\n"
            "        if not line.startswith('gitdir:'):\n"
            "            return None\n"
            "        g = line.split(':', 1)[1].strip()\n"
            "    if not os.path.isdir(g):\n"
            "        return None\n"
            "    try:\n"
            "        h = open(os.path.join(g, 'HEAD')).read().strip()\n"
            "    except OSError:\n"
            "        return None\n"
            "    if re.fullmatch('[0-9a-f]{40}', h):\n"
            "        return h\n"
            "    if not h.startswith('ref:'):\n"
            "        return None\n"
            "    ref_name = h.split(':', 1)[1].strip()\n"
            "    try:\n"
            "        v = open(os.path.join(g, ref_name)).read().strip()\n"
            "        if re.fullmatch('[0-9a-f]{40}', v):\n"
            "            return v\n"
            "    except OSError:\n"
            "        pass\n"
            "    try:\n"
            "        for ln in open(os.path.join(g, 'packed-refs')):\n"
            "            parts = ln.split()\n"
            "            if len(parts) == 2 and parts[1] == ref_name:\n"
            "                return parts[0]\n"
            "    except OSError:\n"
            "        pass\n"
            "    return None\n"
            "out = {}\n"
            "for p in " + json.dumps(paths) + ":\n"
            "    v = head(p)\n"
            "    if v:\n"
            "        out[p] = v\n"
            "print(json.dumps(out))\n")
    proc = subprocess.run(["docker", "run", "--rm", "--entrypoint", python_bin, ref, "-c", code],
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise BaselineError(f"读取镜像内 .git HEAD 失败: {ref}\n{proc.stderr.strip()}", code=1)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise BaselineError(f"读取镜像内 .git HEAD 无输出: {ref}", code=1)
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        raise BaselineError(f".git HEAD 探针输出不是 JSON: {ref}\n{e}", code=1)


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


# ---- Dockerfile 解析（ENV-SPEC §4.2：只识别列出的模式，其余一律报错） ----

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
        self.packages = {}    # pip 发行名 -> 索引发行包（§4.2 模式 6，版本 pin、无 commit）
        self.patch_dirs = {}  # 镜像内 patches 目录 -> assets 相对路径
        self.warnings = []

    def add_repo(self, name, **kw):
        if name in self.repos:
            raise BaselineError(f"同一 repo 在本层出现两次，无法判定: {name}")
        self.repos[name] = {"name": name, **kw}
        if kw.get("dir"):
            self.dir_repo[kw["dir"]] = name
        return self.repos[name]

    def add_package(self, dist, version, opts, seg):
        """§4.2 模式 6：pip 索引发行包。版本锚是 version，没有 commit，不可下放为挂载资产。"""
        if dist in self.packages:
            raise BaselineError(f"同一索引发行包在本层安装两次，无法判定: {dist}")
        index_url = None
        for k, o in enumerate(opts):
            if o in PIP_INDEX_OPTS and k + 1 < len(opts):
                index_url = opts[k + 1]
            else:
                m = re.fullmatch(r"(--index-url|-i)=(.+)", o)
                if m:
                    index_url = m.group(2)
        if not index_url:
            raise BaselineError(f"{dist}: pip install 从索引安装但未写出 --index-url/-i，"
                                f"无法记录来源索引（§4.2 模式 6）: {seg[:80]}")
        self.packages[dist] = {
            "name": dist, "version": version, "index_url": index_url,
            "install": "pip (index wheel" + (", --no-deps" if "--no-deps" in opts else "") + ")",
        }
        return self.packages[dist]

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
            raise BaselineError("本构建层出现 pip install -e：固化资产不允许 editable 安装（ENV-SPEC §2.1）")
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
            # 选项解析：带独立取值的选项要跳过它的取值，否则 `-i <url>` 的 url 被当成第二个目标
            targets, opts, skip = [], [], False
            for t in toks[i + 1:]:
                if skip:
                    opts.append(t)
                    skip = False
                elif t.startswith("-"):
                    opts.append(t)
                    skip = t in PIP_OPTS_WITH_VALUE
                else:
                    targets.append(t)
            if len(targets) != 1:
                raise BaselineError(f"pip install 的目标不唯一，拒绝猜测: {seg[:80]}")
            # §4.2 模式 6：pip install <pkg>==<pinned version>（索引发行包，版本 pin、无 commit）
            m_req = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)===?([A-Za-z0-9][A-Za-z0-9.+!_-]*)",
                                 targets[0])
            if m_req:
                self.add_package(m_req.group(1), m_req.group(2), opts, seg)
                continue
            d = cd_dir if targets[0] in (".", "./") else targets[0].rstrip("/")
            if d not in self.dir_repo:
                if not re.search(r"[/.]", targets[0]):
                    raise BaselineError(
                        f"pip install 从索引安装但未固定版本（§4.2 模式 6 要求 "
                        f"<pkg>==<version> 或 ===<version>），拒绝猜测: {seg[:80]}")
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


def _strip_local(version: str) -> str:
    """剥掉基座包版本里只表示构建形态、不表示版本的 local version 后缀（见 LOCAL_VERSION_STRIP）。"""
    for suffix in LOCAL_VERSION_STRIP:
        version = version.replace(suffix, "")
    return version


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
    report = image_python_report(recipe.from_ref, python_bin, BASE_PROVIDES_FILES)
    provides = {}
    # L1 版本事实：① 环境变量探针 ② 镜像内文件探针 ③ 包版本白名单。三者都是声明式表，
    # 接新芯片只加数据，不改代码（昇腾的 CANN 在环境变量里，昆仑芯的 XTDK/XRE/XCCL 在
    # <site-packages>/torch_xmlir/_versions.txt 里，不在任何环境变量里）。
    env_vals = {}
    for e in (base_info.get("Config") or {}).get("Env") or []:
        k, _, v = e.partition("=")
        env_vals[k] = v
    for var, (key, pattern) in BASE_PROVIDES_ENV.items():
        m = re.match(pattern, env_vals.get(var, "")) if var in env_vals else None
        if m:
            provides[key] = m.group(1)
    for path, text in (report.get("base_files") or {}).items():
        keys = next((p["keys"] for p in BASE_PROVIDES_FILES if p.get("path") == path), {})
        for line in text.splitlines():
            name, sep, val = line.partition(":")
            if sep and name.strip() in keys:
                provides[keys[name.strip()]] = val.strip()
    if report["python_version"]:
        provides["python"] = report["python_version"]
    for pkg in BASE_PROVIDES_PKGS:
        if report["versions"].get(pkg):
            provides[pkg] = _strip_local(report["versions"][pkg])
    repos = []
    in_image = {}        # dist -> 镜像内源码绝对路径
    for dist, e in sorted(report["editables"].items()):
        src = e.get("source", "")
        name = PKG_TO_REPO.get(dist, dist)   # pip 发行名 -> 仓库名，三张表才认得出层级与标签
        if not src.startswith("file:/"):
            warnings.append(f"{name}: 父镜像的 editable 安装源码不在镜像内（{src}），不记为固化资产")
            continue
        in_image[dist] = re.sub(r"^file://", "", src)
    # pip 元数据没记 commit 时，回退读镜像内源码目录的 .git HEAD。**只填空，不改写**：
    # 已有 commit 的条目一律以 pip 元数据为准，避免换一个取数口径就让既有基线漂移。
    # 这条回退把「commit: unavailable 由 image id 兜底」降级成了真正的版本锚（§3.1）。
    heads = {}
    missing = [in_image[d] for d, e in report["editables"].items()
               if d in in_image and not e.get("commit")]
    if missing:
        try:
            heads = image_git_heads(recipe.from_ref, python_bin, missing)
        except BaselineError as e:
            warnings.append(f"镜像内 .git HEAD 探针未能执行，取不到 commit 的条目仍记 "
                            f"unavailable: {e}")
    for dist, e in sorted(report["editables"].items()):
        if dist not in in_image:
            continue
        src_path, name = in_image[dist], PKG_TO_REPO.get(dist, dist)
        commit, commit_source = e.get("commit"), None
        if not commit and heads.get(src_path):
            commit, commit_source = heads[src_path], "in-image-git"
            warnings.append(f"{name}: pip 元数据无 commit，回退取镜像内 {src_path}/.git 的 "
                            f"HEAD={commit[:12]} 作为版本锚（commit_source: in-image-git）")
        provides.pop(dist, None)  # 能对应到 commit 的包只记在 baked_repos，不在 provides 重复
        entry = {"_name": name, "layer": LAYER.get(name, DEFAULT_LAYER), "from": "base_image",
                 "commit": commit,
                 "version": _strip_local(report["versions"].get(dist) or "") or None,
                 "source_path": src_path,
                 "install": "editable-in-image",
                 "info": repo_info(name, commit, repo_url=e.get("repo_url"))}
        if commit_source:
            entry["commit_source"] = commit_source
        repos.append(entry)
        warnings.append(f"{name}: 继承自父镜像的 editable 安装，源码位于镜像内 file://{src_path}，"
                        "记为固化资产（install: editable-in-image，ENV-SPEC §2.1）")
    return {"ref": recipe.from_ref, "digest": digest, "provides": provides}, repos


def _merge_baked_repos(entries, warnings):
    """按名字合并固化 repo，后出现者（本层）覆盖先出现者（父层/基座），保留原顺序。

    被覆盖的条目不是错误：父基座里 editable 装着同一个组件，本层又把它非 editable 固化一遍，
    是昆仑芯这条线的常态。但基线文件里只能留一条，且必须说清留下的是哪一条。"""
    merged = {}
    for e in entries:
        name = e["_name"]
        old = merged.get(name)
        if old is None:
            merged[name] = e
            continue
        e = dict(e)
        e["supersedes"] = f"{old.get('from', 'unknown')} (commit={_short(old.get('commit'))}," \
                          f" install={old.get('install')})"
        merged[name] = e
        warnings.append(f"{name}: 本层固化覆盖了 {old.get('from', 'unknown')} 继承来的同名资产，"
                        f"基线只记本层这一条（supersedes 字段留痕）")
    return list(merged.values())


def _baked_repo_groups(entries):
    """按 provenance 把固化 repo 分成两组，组内保持入参顺序（不重排）。

    返回 [("base", 厂商基座自带), ("recipe", 本组配方带入)]。划分依据是 from 字段：
    `base_image` 一组是被动继承的现状（本组配方未声明，改不了）；其余一组是本组配方带入
    （父镜像是本仓自建镜像时，从父基线继承来的条目 from 是父层的 Dockerfile.repro(...)，
    那仍属本组可控，不划到厂商侧）。先 base 后 recipe 与 export_image 的自然追加顺序一致
    （_external_base()/父基线的条目先、本层 recipe 条目后），故分组不会引起条目重排。"""
    base, ours = [], []
    for e in entries:
        (base if (e.get("from") or "") == "base_image" else ours).append(e)
    return [("base", base), ("recipe", ours)]


def _assert_inherited_commits(entries, img_labels, warnings):
    """继承资产（from: base_image）的 commit 与本组声明的 commit 标签交叉核对。

    对这类条目，标签不再是「本层固化资产的锚」（本组没固化它），而是「本组对基座内容的
    断言」——类似 lockfile 对不受控上游的 digest 断言：基座换了内容就必须炸，而不是静默
    跟随。标签不存在时不核对（绝大多数继承资产没有标签，例如昇腾 v3 的 vllm）；基座读不出
    commit 时也不拿标签值去填 commit（那是拿断言冒充观测），只告警。"""
    for e in entries:
        if (e.get("from") or "") != "base_image":
            continue
        key = COMMIT_LABEL.get(e.get("_name"))
        declared = img_labels.get(key) if key else None
        if not declared:
            continue
        if not e.get("commit"):
            warnings.append(f"{e['_name']}: 本组声明了标签 {key}={declared}，但基座 pip 元数据取不到 "
                            f"commit，无法核对（不以标签值冒充观测值，仍记 commit: unavailable）")
        elif e["commit"] != declared:
            raise BaselineError(
                f"{e['_name']}: 基座内容与本组声明不符（可能是厂商换了基座）：基座 pip 元数据读出 "
                f"commit={e['commit']}，本组标签 {key}={declared}。"
                f"核对基座后更新该标签，或改回在本层固化该资产", code=1)


def export_image(image_dir, demo=None, image_ref=None) -> int:
    image_dir = Path(image_dir).resolve()
    recipe = analyze_dockerfile(image_dir)
    ref = image_ref
    if not ref and (image_dir / "build.sh").is_file():
        tags = re.findall(r"-t\s+(\S+)", (image_dir / "build.sh").read_text(encoding="utf-8"))
        ref = tags[-1] if tags else None
        # build.sh 的 -t 必须是字面 tag：这里只做文本抽取，不展开 shell 变量。抽到
        # "$TAG" 之类就直接报错，不要把它当 ref 丢给 docker（否则报错出在 docker 那层，
        # 看起来像镜像不存在，实际是配方写法问题）。
        if ref and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*(:[A-Za-z0-9._-]+)?", ref):
            raise BaselineError(
                f"build.sh 的 -t 不是字面镜像 tag（抽到 {ref}），本工具不展开 shell 变量："
                f"请在 build.sh 的 -t 后写出完整 tag（参照 ascend-train-comm/v3/build.sh），"
                f"或用 --image-ref 显式指定")
    if not ref:
        raise BaselineError(f"无法确定镜像 ref（build.sh 里没有 -t，可用 --image-ref 指定）: {image_dir}")
    info = inspect_image(ref)
    img_labels = (info.get("Config") or {}).get("Labels") or {}
    warnings = list(recipe.warnings)

    # Dockerfile LABEL（ARG 展开后）与镜像标签交叉核对。版本锚标签与 commit 标签同等对待：
    # 无 commit 的固化资产全靠 version 标签做核对项（§4.2）。
    anchor_labels = set(COMMIT_LABEL.values()) | set(VERSION_LABEL.values())
    for key, val in recipe.labels.items():
        if key in anchor_labels:
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

    # 同名资产：本层固化覆盖父层/基座继承来的同名条目（就近者赢），覆盖关系留痕。
    # 不去重会写出重复 YAML 键，yaml.safe_load 静默取后者——基线文件必须无歧义。
    baked_repos = _merge_baked_repos(baked_repos, warnings)
    # 合并之后再核对继承资产：被本层覆盖掉的同名条目已经不在了，标签该归属谁没有歧义。
    _assert_inherited_commits(baked_repos, img_labels, warnings)

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

    # 索引发行包（§4.2 模式 6）：版本锚三方核对 —— ARG（已随 LABEL 核对过）== 镜像标签
    # == 镜像内实装 dist 版本。第三条边是 commit 情形做不到的（源码固化后可能已删），
    # 所以无 commit 的资产可核对性并不弱于有 commit 的资产。
    baked_packages = []
    if recipe.packages:
        python_bin = recipe.args.get("PYTHON_BIN")
        if not python_bin:
            raise BaselineError("Dockerfile 未声明 ARG PYTHON_BIN，无法读取镜像内实装版本"
                                "以核对索引发行包的版本锚（§4.2 模式 6）")
        installed = image_python_report(ref, python_bin)["versions"]
        wheel_tags = image_wheel_tags(ref, python_bin, recipe.packages.keys())
        for dist, pkg in recipe.packages.items():
            key = dist.replace("-", "_")
            label_key = VERSION_LABEL.get(dist)
            if not label_key:
                raise BaselineError(f"{dist}: 未登记版本标签（请在 baseline.py 的 VERSION_LABEL "
                                    f"里登记，§4.2 模式 6 要求版本锚可机器核对）")
            label_val = img_labels.get(label_key)
            if not label_val:
                raise BaselineError(f"{dist}: 镜像缺少 {label_key} 标签，无法确定版本锚", code=1)
            if label_val != pkg["version"]:
                raise BaselineError(f"{dist}: ARG 版本 {pkg['version']} 与镜像标签 "
                                    f"{label_key}={label_val} 不一致", code=1)
            actual = installed.get(key)
            if not actual:
                raise BaselineError(f"{dist}: 镜像内未安装该包（pip 元数据查无此项），"
                                    f"配方声称从 {pkg['index_url']} 安装", code=1)
            if actual != pkg["version"]:
                raise BaselineError(f"{dist}: 镜像标签/ARG 版本 {pkg['version']} 与镜像内实装版本 "
                                    f"{actual} 不一致", code=1)
            # 第四条边（§4.2 模式 6）：索引页 PEP 503 的 `#sha256=`。wheel 文件名由
            # 「dist 名 + 实装版本 + dist-info 的 WHEEL Tag」从镜像内事实完整推出，再在索引页
            # 上按解析匹配定位同一条目，取其指纹。**尽力而为**：构建机可能无外网，取不到
            # 只告警并省略 wheel 块，不让 export 失败（离线建得出镜像就该导得出基线）。
            wheel = None
            tags = wheel_tags.get(key) or wheel_tags.get(dist) or []
            if len(tags) > 1:
                warnings.append(f"{dist}: dist-info 的 WHEEL 有 {len(tags)} 条 Tag 行"
                                f"（压缩 tag 集 {tags}），不猜文件名，本项不记 wheel sha256")
            else:
                try:
                    wheel = resolve_index_wheel(pkg["index_url"], dist, actual,
                                                tag=tags[0] if tags else None)
                    wheel["source"] = "index-page-pep503"
                    wheel["fetched_at"] = now_iso()
                except BaselineError as e:
                    warnings.append(f"{dist}: 未能取到 wheel sha256，本项只有 version 一个锚"
                                    f"（上游以同一版本号重发时察觉不到）；原因: {e}")
            anchors = f"version (标签 {label_key} 与镜像内实装版本双向核对"
            anchors += "；wheel sha256 见 wheel 块，由 check --online 核对)" if wheel else \
                       "；本次未取到 wheel sha256，见 export 告警)"
            entry = {
                "name": dist, "layer": LAYER.get(dist, DEFAULT_LAYER),
                "version": pkg["version"], "index_url": pkg["index_url"],
                "install": pkg["install"], "pin": anchors,
            }
            if wheel:
                entry["wheel"] = wheel
            baked_packages.append(entry)

    # 完整包清单：assets/provenance/ 下名字含 pipfreeze 的快照。多份时按 PACKAGES_PREFER
    # 的标记顺序挑（各血统的命名惯例登记在那张表里），同一标记内取名字最大的一份；
    # 一个标记都不命中时取名字最大的并告警 —— 这是「仅供查阅」字段，不是 check 项。
    freezes = sorted(f for f in (image_dir / "assets" / "provenance").glob("*.txt")
                     if "pipfreeze" in f.name)
    packages = None
    if freezes:
        picked = []
        for marker in PACKAGES_PREFER:
            picked = [f for f in freezes if marker in f.name]
            if picked:
                break
        packages = rel_to_repo(max(picked or freezes, key=lambda f: f.name))
        if not picked and len(freezes) > 1:
            warnings.append(
                f"assets/provenance/ 下有 {len(freezes)} 份 pipfreeze 快照，没有一份命中 "
                f"PACKAGES_PREFER（{', '.join(PACKAGES_PREFER)}），取名字最大的一份 "
                f"{Path(packages).name}；要换别的请在 baseline.py 的 PACKAGES_PREFER 里登记标记")

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
              "  # 厂商基座：被动继承、按 digest 锁定、本组无法重建 —— 这是设计边界，不是缺陷（§3.1）。",
              f"  ref: {y(base_block.get('ref'))}",
              f"  digest: {y(base_block.get('digest'))}",
              f"  provides: {y(base_block.get('provides') or {})}"]
    # 分组输出（ENV-SPEC §3.1）：读者必须一眼看出这项资产是本组可控的带入，还是被动继承的
    # 现状。只有该组有条目时才出组头；全部条目同属一组时也要出那一组的组头。
    L += ["", "baked_repos: {}" if not baked_repos else "baked_repos:"]
    for group, group_entries in _baked_repo_groups(baked_repos):
        if not group_entries:
            continue
        L += BAKED_GROUP_HEADERS[group]
        for e in group_entries:
            e = dict(e)
            e_name = e.pop("_name")
            L.append(f"  {e_name}:")
            L.append(f"    layer: {e.pop('layer', DEFAULT_LAYER)}")
            L.append(f"    from: {y(e.pop('from', 'unknown'))}")
            # 取不到 commit 不是错误也不能静默成 null：显式 unavailable + 告警，由 image id 兜底
            commit_v = e.pop("commit", None)
            L.append(f"    commit: {y(commit_v) if commit_v else 'unavailable'}")
            if not commit_v:
                warnings.append(f"{e_name}: 取不到 commit，已记 commit: unavailable，"
                                f"版本事实由 image id 兜底（ENV-SPEC §3.1）")
            # commit 的取数口径：缺省（不写该字段）= pip 元数据；写出来的只有回退来源
            if e.get("commit_source"):
                L.append(f"    commit_source: {y(e.pop('commit_source'))}")
            for k in ("version", "patches", "source_path"):
                if e.get(k):
                    L.append(f"    {k}: {y(e.pop(k))}")
                else:
                    e.pop(k, None)   # 空值不泄漏到字段块末尾，字段顺序才稳定
            L.append(f"    install: {y(e.pop('install', None))}")
            info_v = e.pop("info", None)
            L.append(f"    info: {y(info_v) if info_v else 'unavailable'}")
            L += [f"    {k}: {y(v)}" for k, v in e.items() if v is not None]
    # 模式 4/6 只可能出现在本组配方里，父基线继承来的也是本组自建镜像的产物：恒为本组带入。
    if baked_packages:
        L += ["", "baked_packages:                                # 索引发行包：版本 pin，无 commit",
              "  # 恒为本组配方带入（§4.2 模式 6）：改 pin + 重新构建即可变更。",
              "  # wheel 块（若有）是索引页 PEP 503 的 `#sha256=`，由 check --online 核对；",
              "  # 它防的是「上游以同一版本号重发不同内容」，不证明镜像内字节即此指纹（§4.2）。"]
        for pkg in baked_packages:
            # 块式映射：wheel 是嵌套块，平铺进一行流式映射会长到不可读（index_url 已经很长）
            wheel = pkg.pop("wheel", None)
            items = list(pkg.items())
            L.append(f"  - {items[0][0]}: {y(items[0][1])}")
            L += [f"    {k}: {y(v)}" for k, v in items[1:]]
            if wheel:
                L.append(f"    wheel: {y(wheel)}")
    L += ["", "baked_resources: []" if not baked_resources else "baked_resources:"]
    if baked_resources:
        L.append("  # 恒为本组配方带入（§4.2 模式 4）：改配方 + 重新构建即可变更。")
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


def _image_python_bin(image_dir):
    """从配方取 ARG PYTHON_BIN（镜像内目标解释器）。取不到返回 None —— 它只给可选的
    附加核对用，不能让 check 本体失败（v1 那种不被 §4.2 识别的配方就取不到）。"""
    try:
        return analyze_dockerfile(Path(image_dir).resolve()).args.get("PYTHON_BIN")
    except BaselineError:
        return None


def _check_editable_git(image_dir, ref, baked, report):
    """附加核对：`install: editable-in-image` 的资产，镜像内 .git 的 HEAD 与基线 commit 比对。

    这些资产的 commit 只来自基座 pip 元数据，而镜像标签只是「本组对基座内容的断言」、
    标签不存在时不核对（ENV-SPEC §3.1）。镜像内源码目录若带 .git，就是一个**不依赖标签、
    镜像自证**的第二锚。取不到（无 .git / 无 PYTHON_BIN / 探针失败）一律 INFO，不 FAIL ——
    带 .git 不是对基座的要求。"""
    want = {name: item.get("source_path") for name, item in baked.items()
            if item.get("install") == "editable-in-image" and item.get("source_path")}
    if not want:
        return
    python_bin = _image_python_bin(image_dir)
    if not python_bin:
        report.append(("INFO", "editable-in-image", "配方未声明 ARG PYTHON_BIN，跳过镜像内 "
                                                    ".git HEAD 这项附加核对"))
        return
    try:
        heads = image_git_heads(ref, python_bin, want.values())
    except BaselineError as e:
        report.append(("INFO", "editable-in-image", f"镜像内 .git HEAD 探针未能执行，跳过该项"
                                                    f"附加核对: {e}"))
        return
    for name, path in sorted(want.items()):
        head, commit = heads.get(path), str(baked[name].get("commit") or "")
        if not head:
            report.append(("INFO", name, f"镜像内 {path} 无 .git，该资产只有 pip 元数据这一个 "
                                         f"commit 来源"))
        elif not SHA40.fullmatch(commit):
            report.append(("INFO", name, f"基线记 commit: {commit or 'unavailable'}，但镜像内 "
                                         f"{path}/.git HEAD={head[:12]} —— 可作为第二版本锚，"
                                         f"重新 export 后即可纳入核对"))
        elif head != commit:
            report.append(("FAIL", name, f"镜像内 {path}/.git HEAD={head[:12]} 与基线 commit "
                                         f"{commit[:12]} 不一致"))
        else:
            report.append(("PASS", name, f"{path}/.git HEAD={head[:12]}（镜像自证，不依赖标签）"))


def _check_wheel_sha256(pkg, name, online, report):
    """附加核对：索引页 PEP 503 的 `#sha256=` 与基线记的 wheel.sha256 比对。

    **必须 opt-in（--online）**：check 在 up.sh --strict 的启动路径上，默认联网会破坏
    离线/气隙使用、而且可能挂住。语义边界要说清 —— 它发现的是「索引当前内容 ≠ 基线当初
    记的值」，**不证明镜像内装的字节就是这个指纹**（要证明就得留着整个 wheel）。"""
    wheel = pkg.get("wheel") or {}
    want = wheel.get("sha256")
    if not want:
        report.append(("INFO", name, "基线未记 wheel sha256（export 时未取到，或基线由旧版"
                                     "工具生成），该项不核对；version 标签仍是硬核对项"))
        return
    if not online:
        report.append(("INFO", name, f"wheel sha256 未核对（需 check --online）: "
                                     f"{wheel.get('filename')} {want[:12]}…"))
        return
    try:
        got = resolve_index_wheel(pkg.get("index_url"), name, pkg.get("version"))
    except BaselineError as e:
        # --online 是显式要求核对，取不到就是「无法下判定」，静默放过会让这个开关失去意义
        report.append(("FAIL", name, f"--online 要求核对 wheel sha256，但未能取到索引页条目: {e}"))
        return
    if got["sha256"] != want:
        report.append(("FAIL", name, f"索引页 sha256 与基线不一致 —— 上游可能以同一版本号重发了"
                                     f"不同内容，须人工裁定: 基线 {want} 索引 {got['sha256']}"
                                     f"（{got['filename']}）"))
    else:
        report.append(("PASS", name, f"wheel sha256 与索引页一致: {got['filename']} {want[:12]}…"))


def check_image(image_dir, online=False) -> int:
    data = load_yaml(Path(image_dir).resolve() / "image.baseline.yaml")
    img = data.get("image") or {}
    line, labels = _check_image_id(img.get("ref"), img.get("id"))
    report = [line]
    baked = (data.get("baked_repos") or {}) if labels is not None else {}
    for name, item in baked.items():
        key = COMMIT_LABEL.get(name)
        if key not in labels:
            # 标签是本组构建层打的，只有「本层固化」的资产才该有：
            #  · from: base_image —— commit 取自基座的 pip 元数据，本来就没有标签可核对，
            #    报 INFO（由 image id 兜底）。昇腾 v3 的 vllm 即属此类。
            #  · 本层固化却没有标签 —— 标签丢了，必须 FAIL，不能以「跳过」放过。
            if item.get("from") == "base_image":
                anchor = f"commit={str(item.get('commit'))[:12]}" if item.get("commit") else "无 commit"
                where = f"无 {key} 标签可核对" if key else "未登记 commit 标签（COMMIT_LABEL）"
                report.append(("INFO", name, f"{anchor} 取自基座 pip 元数据，{where}，"
                                             f"由 image id 兜底"))
            elif SHA40.fullmatch(str(item.get("commit") or "")):
                report.append(("FAIL", name, f"基线记了 commit {str(item['commit'])[:12]}，"
                                             f"但镜像缺少标签 {key}，无法核对"))
            else:
                report.append(("INFO", name, "无 commit 锚（commit: unavailable），"
                                             "由 image id 兜底（ENV-SPEC §3.1）"))
        elif labels[key] != item.get("commit"):
            report.append(("FAIL", name, f"标签 {key} 不一致: 基线 {item.get('commit')} 镜像 {labels[key]}"))
        else:
            report.append(("PASS", name, f"{key}={labels[key][:12]}"))
    # 索引发行包：核对项是 version 标签（§4.2 模式 6）。缺标签即 FAIL —— 它是该资产
    # 唯一的版本锚，跳过就等于零校验。
    for pkg in (data.get("baked_packages") or []) if labels is not None else []:
        name = pkg.get("name")
        key = VERSION_LABEL.get(name)
        if not key:
            report.append(("FAIL", name, "未登记版本标签（baseline.py 的 VERSION_LABEL），无法核对"))
        elif key not in labels:
            report.append(("FAIL", name, f"镜像缺少版本标签 {key}，该资产无其他版本锚，无法核对"))
        elif labels[key] != pkg.get("version"):
            report.append(("FAIL", name, f"标签 {key} 不一致: 基线 {pkg.get('version')} "
                                         f"镜像 {labels[key]}"))
        else:
            report.append(("PASS", name, f"{key}={labels[key]}"))
        _check_wheel_sha256(pkg, name, online, report)
    if labels is not None:
        _check_editable_git(image_dir, img.get("ref"), baked, report)
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
        raise BaselineError(f"{rel_to_repo(compose)} 缺少顶层 x-flagrt 声明（ENV-SPEC §3.2）")
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


def _on_remote(host: Path, commit: str) -> bool:
    """commit 是否已在某个远端跟踪分支上（依据本地已 fetch 的远端信息）。"""
    out, _ = run_git("branch", "-r", "--contains", commit, cwd=host)
    return bool(out)


def export_env(env_dir, require_clean=False) -> int:
    env_dir, compose, image_baseline_rel, declared = _env(env_dir)
    image_baseline = load_yaml(env_dir / image_baseline_rel)
    baked = image_baseline.get("baked_repos") or {}
    binds = compose_binds(compose) if declared else []
    mounted, dirty = {}, []
    baked_pkgs = image_baseline.get("baked_packages") or []
    for name, spec in declared.items():
        spec = spec or {}
        # 索引发行包不可下放：没有源码 checkout 可挂，挂载也不会改变已装好的 wheel（§2.3）
        if any(p.get("name") == name for p in baked_pkgs):
            raise BaselineError(
                f"{name} 是镜像基线里的索引发行包（baked_packages），不是可挂载的代码仓，"
                f"不能作 mounted_repos 的键（ENV-SPEC §2.3「不可下放的固化资产」）")
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
        entry["pushed"] = _on_remote(host, head)
        info = repo_info(name, head)
        if info:
            entry["info"] = info
        mounted[name] = entry
        if is_dirty:
            dirty.append(name)
    unpushed = [n for n, e in mounted.items() if not e["pushed"]]
    if require_clean and (dirty or unpushed):
        if dirty:
            print(f"[FAIL] --require-clean: 以下挂载 repo 工作区不干净，拒绝导出: {', '.join(dirty)}",
                  file=sys.stderr)
        if unpushed:
            print(f"[FAIL] --require-clean: 以下挂载 repo 的 commit 不在任何远端分支上，他人无法 load，"
                  f"拒绝导出: {', '.join(unpushed)}", file=sys.stderr)
        return 1
    for name in dirty:
        print(f"[WARN] {name}: 工作区不干净，已记录 dirty: true")
    for name in unpushed:
        print(f"[WARN] {name}: commit 不在任何远端分支上（未 push 或远端信息未 fetch），他人无法 load")

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
        if key == "mounted_repos":
            L.append("# " + "─" * 30 + " 环境挂载（env） " + "─" * 30)
        L.append(f"{key}:{' {}' if not items else ''}{comment}")
        for name, item in items.items():
            L.append(f"  {name}:")
            L += [f"    {k}: {'unavailable' if v is None else y(v)}" for k, v in item.items()]
        L.append("")
        if key == "baked_repos" and baked_pkgs:
            L.append("baked_packages:            # 原样抄录自镜像基线，只读；不可下放为挂载资产")
            L += [f"  - {y(p)}" for p in baked_pkgs]
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
        if not _on_remote(host, head):
            report.append(("WARN", name, "当前 commit 不在任何远端分支上，他人无法按基线 load（不阻断）"))
        if item.get("overrides_baked"):
            report.append(("INFO", name, f"覆盖镜像内固化版本 {str(item.get('baked_commit'))[:12]}，"
                                         f"生效方式 {item.get('activation')}"))
        if item.get("baked_patches"):
            report.append(("WARN", name, "挂载副本不含镜像内固化的补丁: "
                           + ", ".join(p.get("file", "?") for p in item["baked_patches"])))
    if base.get("baked_repos"):
        report.append(("INFO", "baked", "固化资产不与宿主比对，由 image id 约束（ENV-SPEC §2）"))
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
        print("[ABORT] 以下挂载 repo 工作区不干净，整体中止，未做任何切换（ENV-SPEC §4.3）: "
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
        pkgs = data.get("baked_packages") or []
        if pkgs:
            print("\n固化索引发行包（baked_packages，版本 pin、无 commit、不可下放为挂载资产）:")
            print(_table([[p.get("layer"), p.get("name"), p.get("version"), p.get("install"),
                           p.get("index_url")] for p in pkgs],
                         ["层级", "包", "version", "安装方式", "来源索引"]))
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
        items += [[p.get("layer"), p.get("name"), p.get("version"), p.get("install")]
                  for p in data.get("baked_packages") or []]
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
                                "规范见 dev/ENV-SPEC.md")
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
                            help="运行环境场景：挂载 repo 不干净或 commit 未推送时拒绝导出")
        if action == "check":
            sp.add_argument("--online", action="store_true",
                            help="镜像场景：额外联网核对 baked_packages 的 wheel sha256"
                                 "（PEP 503 索引页的 #sha256=）。默认不联网 —— check 在 "
                                 "up.sh --strict 的启动路径上。需绕代理时先 export no_proxy='*'")
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
            if args.online and not args.image:
                print("用法错误: --online 只用于镜像场景（--image <image-dir>）", file=sys.stderr)
                return 2
            return (check_image(args.image, online=args.online) if args.image
                    else check_env(args.env_dir))
        return show(args.image or args.env_dir, image=bool(args.image))
    except BaselineError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
