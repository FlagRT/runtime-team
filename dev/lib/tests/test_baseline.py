#!/usr/bin/env python3
"""dev/lib/baseline.py 的单元测试：覆盖 dev/ENV-SPEC.md §4.2 的全部 Dockerfile 判定模式
（含报错分支）与 §4.3 的运行环境安全规则。

运行方式（本机 python3 没有 pytest，用 unittest；有 pytest 时也可直接 pytest）：
  python3 -m unittest discover -s dev/lib/tests -v
  python3 dev/lib/tests/test_baseline.py

docker 调用（image inspect / docker run 读 pip 元数据）在单测里全部 mock，
不需要本机有镜像；git 操作用临时仓库真跑。
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

LIB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LIB))

import baseline  # noqa: E402

DF_HEAD = """ARG BASE_IMAGE=ext/base:1
FROM ${BASE_IMAGE}
ARG PYTHON_BIN=/usr/bin/python3
"""

BASE_REPORT = {
    "python_version": "3.11.15",
    "versions": {"torch": "2.10.0+cpu", "torch_npu": "2.10.0", "vllm": "0.20.2+empty"},
    "editables": {},
}


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=str(cwd), check=True,
                   capture_output=True, text=True)


def make_repo(path: Path, files=("a.txt",)) -> str:
    path.mkdir(parents=True, exist_ok=True)
    git("init", "-q", "-b", "main", cwd=path)
    git("config", "user.email", "t@example.com", cwd=path)
    git("config", "user.name", "t", cwd=path)
    for f in files:
        (path / f).write_text("x\n", encoding="utf-8")
    git("add", "-A", cwd=path)
    git("commit", "-qm", "init", cwd=path)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(path),
                          capture_output=True, text=True).stdout.strip()


def fake_inspect(labels=None, image_id="sha256:" + "a" * 64, digest="sha256:" + "b" * 64,
                 env=None):
    def _inner(ref):
        return {
            "Id": image_id,
            "RepoDigests": [f"{ref.split(':')[0]}@{digest}"] if digest else [],
            "Config": {
                "Labels": dict(labels or {}),
                "Env": list(env or ["ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0"]),
            },
        }
    return _inner


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="baseline-test-"))
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(self.tmp)]))
        self.env_patch = mock.patch.dict(os.environ, {
            "FLAGRT_WORKSPACE_ROOT": str(self.tmp / "ws"),
            "FLAGRT_BASELINE_CACHE": str(self.tmp / "cache"),
        })
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        (self.tmp / "ws").mkdir(parents=True, exist_ok=True)
        # 三个「会真的出网/起容器」的探针默认全部封死：假 ref 上 docker run 会去 pull、
        # curl 会真连索引页，单测里两者都不许发生（实测漏掉会把测试挂死而不是报错）。
        # 要测它们自身行为的用例在自己的 with 块里覆盖这些默认值。
        for name, kw in (("image_wheel_tags", {"return_value": {}}),
                         ("image_git_heads", {"return_value": {}}),
                         ("resolve_index_wheel",
                          {"side_effect": baseline.BaselineError("单测默认不联网", code=1)})):
            p = mock.patch.object(baseline, name, **kw)
            p.start()
            self.addCleanup(p.stop)

    def image_dir(self, dockerfile, name="img", version="v3", assets=None, tag=None):
        d = self.tmp / "dev" / "images" / name / version
        (d / "assets").mkdir(parents=True, exist_ok=True)
        (d / "Dockerfile.repro").write_text(dockerfile, encoding="utf-8")
        for rel, content in (assets or {}).items():
            p = d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        if tag:
            (d / "build.sh").write_text(f"docker build -t {tag} .\n", encoding="utf-8")
        return d


# --------------------------------------------------------------------------
# §4.2 Dockerfile 判定规则
# --------------------------------------------------------------------------

class TestDockerfilePatterns(TempCase):
    def test_pattern1_copy_src_plus_pip_install(self):
        d = self.image_dir(DF_HEAD + """COPY src/FlagGems/ /opt/src/FlagGems/
RUN ${PYTHON_BIN} -m pip install --no-deps /opt/src/FlagGems
""")
        r = baseline.analyze_dockerfile(d)
        self.assertEqual(r.repos["FlagGems"]["source"], "copy")
        self.assertEqual(r.repos["FlagGems"]["install"], "pip (non-editable)")

    def test_pattern1_use_flag_recorded_in_install(self):
        d = self.image_dir(DF_HEAD + """COPY src/FlagCX/ /opt/src/FlagCX/
RUN cd /opt/src/FlagCX && USE_ASCEND=1 ${PYTHON_BIN} -m pip install .
""")
        r = baseline.analyze_dockerfile(d)
        self.assertEqual(r.repos["FlagCX"]["install"],
                         "pip (non-editable, USE_ASCEND=1)")

    def test_pattern1_copy_src_without_install_errors(self):
        d = self.image_dir(DF_HEAD + "COPY src/FlagGems/ /opt/src/FlagGems/\n")
        with self.assertRaises(baseline.BaselineError) as cm:
            baseline.analyze_dockerfile(d)
        self.assertIn("未找到对应的 pip install", str(cm.exception))

    def test_pattern2_git_fetch_arg(self):
        d = self.image_dir(DF_HEAD + """ARG FLAGTREE_COMMIT=%s
RUN mkdir -p /root/FlagTree && cd /root/FlagTree && git init -q \\
    && git remote add origin https://github.com/flagos-ai/FlagTree.git \\
    && git fetch --depth 1 origin ${FLAGTREE_COMMIT} \\
    && git checkout -q FETCH_HEAD -b triton_v3.5.x
RUN cd /root/FlagTree && ${PYTHON_BIN} -m pip install .
""" % ("1" * 40))
        r = baseline.analyze_dockerfile(d)
        repo = r.repos["FlagTree"]
        self.assertEqual(repo["source"], "fetch")
        self.assertEqual(repo["arg"], "FLAGTREE_COMMIT")
        self.assertEqual(repo["arg_value"], "1" * 40)
        self.assertEqual(repo["branch_hint"], "triton_v3.5.x")
        self.assertEqual(repo["install"], "pip (non-editable)")

    def test_pattern2_unpinned_fetch_errors(self):
        d = self.image_dir(DF_HEAD + """RUN cd /root/x && git remote add origin https://e/x.git \\
    && git fetch --depth 1 origin main
""")
        with self.assertRaises(baseline.BaselineError) as cm:
            baseline.analyze_dockerfile(d)
        self.assertIn("拒绝猜测", str(cm.exception))

    def test_pattern3_external_base_editable_in_image(self):
        d = self.image_dir(DF_HEAD, tag="my/img:1")
        report = dict(BASE_REPORT)
        report["editables"] = {"vllm": {
            "source": "file:///vllm-workspace/vllm", "commit": "c" * 40,
            "repo_url": "https://github.com/vllm-project/vllm.git"}}
        with mock.patch.object(baseline, "inspect_image", fake_inspect()), \
             mock.patch.object(baseline, "image_python_report", return_value=report), \
             mock.patch.object(baseline, "repo_info", return_value={"branch": "main"}):
            self.assertEqual(baseline.export_image(d), 0)
        data = baseline.load_yaml(d / "image.baseline.yaml")
        self.assertEqual(data["base_image"]["ref"], "ext/base:1")
        self.assertEqual(data["base_image"]["digest"], "sha256:" + "b" * 64)
        self.assertEqual(data["base_image"]["provides"]["cann"], "9.0.0")
        # 能对应到 commit 的包只出现一次：在 baked_repos，带版本；provides 不重复
        self.assertNotIn("vllm", data["base_image"]["provides"])
        vllm = data["baked_repos"]["vllm"]
        self.assertEqual(vllm["version"], "0.20.2")
        self.assertEqual(vllm["install"], "editable-in-image")
        self.assertEqual(vllm["from"], "base_image")
        self.assertEqual(vllm["commit"], "c" * 40)
        self.assertEqual(vllm["source_path"], "/vllm-workspace/vllm")

    def test_pattern3_parent_baseline_is_inlined(self):
        parent = self.image_dir(DF_HEAD, name="parent")
        (parent / "image.baseline.yaml").write_text(json.dumps({
            "schema": baseline.SCHEMA, "kind": "image",
            "image": {"ref": "flagrt/parent:1", "id": "sha256:" + "a" * 64,
                      "recipe": ["dev/images/parent/v3/Dockerfile.repro"]},
            "base_image": {"ref": "ext/base:1", "digest": "sha256:" + "b" * 64,
                           "provides": {"python": "3.11.15"}},
            "baked_repos": {"FlagGems": {"layer": "L2", "from": "parent/v3 ...",
                                         "commit": "f" * 40,
                                         "install": "pip (non-editable)"}},
            "baked_resources": [{"name": "llvm-toolchain", "layer": "L2",
                                 "url": "http://x/y.tar.gz", "sha256": "0" * 64}],
        }), encoding="utf-8")
        child = self.image_dir("""ARG PARENT_IMAGE=flagrt/parent:1
FROM ${PARENT_IMAGE}
ARG PYTHON_BIN=/usr/bin/python3
ARG FLAGCX_COMMIT=%s
COPY src/FlagCX/ /opt/src/FlagCX/
RUN cd /opt/src/FlagCX && USE_ASCEND=1 ${PYTHON_BIN} -m pip install .
LABEL io.flagrt.flagcx.commit="${FLAGCX_COMMIT}"
""" % ("e" * 40), name="child", tag="flagrt/child:1")
        labels = {"io.flagrt.flagcx.commit": "e" * 40}
        with mock.patch.object(baseline, "REPO_ROOT", self.tmp), \
             mock.patch.object(baseline, "inspect_image", fake_inspect(labels)), \
             mock.patch.object(baseline, "repo_info", return_value=None):
            self.assertEqual(baseline.export_image(child, demo="x/demo/"), 0)
        data = baseline.load_yaml(child / "image.baseline.yaml")
        self.assertEqual(list(data["baked_repos"]), ["FlagGems", "FlagCX"])
        self.assertEqual(data["baked_repos"]["FlagGems"]["commit"], "f" * 40)
        self.assertEqual(data["base_image"]["ref"], "ext/base:1")
        self.assertEqual(len(data["image"]["recipe"]), 2)
        self.assertEqual(data["image"]["demo"], "x/demo/")
        self.assertEqual(data["baked_resources"][0]["name"], "llvm-toolchain")
        self.assertEqual(data["baked_repos"]["FlagCX"]["info"], "unavailable")

    def test_pattern3_parent_id_mismatch_errors(self):
        parent = self.image_dir(DF_HEAD, name="parent")
        (parent / "image.baseline.yaml").write_text(json.dumps({
            "image": {"ref": "flagrt/parent:1", "id": "sha256:" + "9" * 64, "recipe": []},
            "baked_repos": {}, "baked_resources": [],
        }), encoding="utf-8")
        child = self.image_dir("""ARG PARENT_IMAGE=flagrt/parent:1
FROM ${PARENT_IMAGE}
ARG PYTHON_BIN=/usr/bin/python3
""", name="child", tag="flagrt/child:1")
        with mock.patch.object(baseline, "REPO_ROOT", self.tmp), \
             mock.patch.object(baseline, "inspect_image", fake_inspect()):
            with self.assertRaises(baseline.BaselineError) as cm:
                baseline.export_image(child)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("父镜像 id 与其基线不一致", str(cm.exception))

    def test_pattern4_curl_url_and_assets(self):
        d = self.image_dir(DF_HEAD + """ARG LLVM_TOOLCHAIN_URL=http://example/llvm.tar.gz
RUN mkdir -p /x && cd /x && curl -fL -o llvm.tar.gz "${LLVM_TOOLCHAIN_URL}" && tar xzf llvm.tar.gz
COPY assets/verify_runtime.py /opt/verify_runtime.py
""", assets={"assets/verify_runtime.py": "print(1)\n"}, tag="my/img:1")
        with mock.patch.object(baseline, "inspect_image", fake_inspect()), \
             mock.patch.object(baseline, "image_python_report", return_value=dict(BASE_REPORT)), \
             mock.patch.object(baseline, "cached_download_sha256", return_value="7" * 64):
            self.assertEqual(baseline.export_image(d), 0)
        res = {r["name"]: r for r in
               baseline.load_yaml(d / "image.baseline.yaml")["baked_resources"]}
        self.assertEqual(res["llvm-toolchain"]["sha256"], "7" * 64)
        self.assertEqual(res["llvm-toolchain"]["url"], "http://example/llvm.tar.gz")
        self.assertEqual(res["verify_runtime.py"]["sha256"],
                         baseline.sha256_file(d / "assets" / "verify_runtime.py"))

    def test_pattern4_curl_without_url_arg_errors(self):
        d = self.image_dir(DF_HEAD + 'RUN curl -fL -o x.tar.gz http://example/x.tar.gz\n')
        with self.assertRaises(baseline.BaselineError) as cm:
            baseline.analyze_dockerfile(d)
        self.assertIn("curl", str(cm.exception))

    def test_patches_under_assets_are_attached_to_repo(self):
        d = self.image_dir("""ARG BASE_IMAGE=ext/base:1
FROM ${BASE_IMAGE}
ARG PYTHON_BIN=/usr/bin/python3
ARG VLLM_PLUGIN_FL_REF=%s
LABEL io.flagrt.vllm_plugin_fl.ref="${VLLM_PLUGIN_FL_REF}"
COPY assets/patches/vllm-plugin-FL/ /opt/flagrt/patches/vllm-plugin-FL/
RUN mkdir -p /root/p && cd /root/p && git init -q \\
    && git remote add origin https://github.com/flagos-ai/vllm-plugin-FL.git \\
    && git fetch --depth 1 origin ${VLLM_PLUGIN_FL_REF} \\
    && git checkout -q FETCH_HEAD \\
    && git apply /opt/flagrt/patches/vllm-plugin-FL/*.patch \\
    && ${PYTHON_BIN} -m pip install .
""" % ("8" * 40), assets={"assets/patches/vllm-plugin-FL/0001-x.patch": "diff\n"},
            tag="my/img:1")
        labels = {"io.flagrt.vllm_plugin_fl.ref": "8" * 40}
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels)), \
             mock.patch.object(baseline, "image_python_report", return_value=dict(BASE_REPORT)), \
             mock.patch.object(baseline, "repo_info", return_value=None):
            self.assertEqual(baseline.export_image(d), 0)
        repo = baseline.load_yaml(d / "image.baseline.yaml")["baked_repos"]["vllm-plugin-FL"]
        self.assertEqual(repo["patches"][0]["file"], "0001-x.patch")
        self.assertEqual(repo["patches"][0]["sha256"],
                         baseline.sha256_file(d / "assets/patches/vllm-plugin-FL/0001-x.patch"))
        self.assertIn("补丁", repo["from"])

    def test_pattern5_pip_install_editable_errors(self):
        d = self.image_dir(DF_HEAD + """COPY src/FlagGems/ /opt/src/FlagGems/
RUN ${PYTHON_BIN} -m pip install -e /opt/src/FlagGems
""")
        with self.assertRaises(baseline.BaselineError) as cm:
            baseline.analyze_dockerfile(d)
        self.assertIn("editable", str(cm.exception))

    def test_unrecognised_copy_errors(self):
        d = self.image_dir(DF_HEAD + "COPY vendor/foo/ /opt/foo/\n")
        with self.assertRaises(baseline.BaselineError) as cm:
            baseline.analyze_dockerfile(d)
        self.assertIn("§4.2", str(cm.exception))

    def test_add_instruction_errors(self):
        d = self.image_dir(DF_HEAD + "ADD http://example/x.tar.gz /opt/\n")
        with self.assertRaises(baseline.BaselineError):
            baseline.analyze_dockerfile(d)

    def test_wget_errors(self):
        d = self.image_dir(DF_HEAD + "RUN wget http://example/x.tar.gz\n")
        with self.assertRaises(baseline.BaselineError):
            baseline.analyze_dockerfile(d)

    def test_pip_install_from_index_without_index_url_errors(self):
        # §4.2 模式 6 要求写出来源索引：版本 pin 住了，但「从哪个索引装的」也是版本事实
        d = self.image_dir(DF_HEAD + "RUN ${PYTHON_BIN} -m pip install flagtree==0.4.0\n")
        with self.assertRaises(baseline.BaselineError) as cm:
            baseline.analyze_dockerfile(d)
        self.assertIn("--index-url", str(cm.exception))

    def test_pip_install_from_index_unpinned_errors(self):
        d = self.image_dir(DF_HEAD + "RUN ${PYTHON_BIN} -m pip install flagtree -i http://idx/s\n")
        with self.assertRaises(baseline.BaselineError) as cm:
            baseline.analyze_dockerfile(d)
        self.assertIn("未固定版本", str(cm.exception))

    def test_pip_install_local_path_not_registered_still_errors(self):
        # 带路径特征的目标仍走原来的报错分支，不被模式 6 吞掉
        d = self.image_dir(DF_HEAD + "RUN ${PYTHON_BIN} -m pip install /opt/somewhere\n")
        with self.assertRaises(baseline.BaselineError) as cm:
            baseline.analyze_dockerfile(d)
        self.assertIn("未登记为固化 repo", str(cm.exception))

    def test_unregistered_repo_name_errors(self):
        d = self.image_dir(DF_HEAD + """COPY src/Unknown-Repo/ /opt/src/Unknown-Repo/
RUN ${PYTHON_BIN} -m pip install /opt/src/Unknown-Repo
""", tag="my/img:1")
        with mock.patch.object(baseline, "inspect_image", fake_inspect()), \
             mock.patch.object(baseline, "image_python_report", return_value=dict(BASE_REPORT)):
            with self.assertRaises(baseline.BaselineError) as cm:
                baseline.export_image(d)
        self.assertIn("COMMIT_LABEL", str(cm.exception))

    def test_arg_label_mismatch_errors(self):
        d = self.image_dir(DF_HEAD + """ARG FLAG_GEMS_COMMIT=%s
LABEL io.flagrt.flag_gems.commit="${FLAG_GEMS_COMMIT}"
COPY src/FlagGems/ /opt/src/FlagGems/
RUN ${PYTHON_BIN} -m pip install /opt/src/FlagGems
""" % ("1" * 40), tag="my/img:1")
        labels = {"io.flagrt.flag_gems.commit": "2" * 40}
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels)), \
             mock.patch.object(baseline, "image_python_report", return_value=dict(BASE_REPORT)):
            with self.assertRaises(baseline.BaselineError) as cm:
                baseline.export_image(d)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("交叉核对失败", str(cm.exception))

    def test_missing_commit_label_errors(self):
        d = self.image_dir(DF_HEAD + """COPY src/FlagGems/ /opt/src/FlagGems/
RUN ${PYTHON_BIN} -m pip install /opt/src/FlagGems
""", tag="my/img:1")
        with mock.patch.object(baseline, "inspect_image", fake_inspect({})), \
             mock.patch.object(baseline, "image_python_report", return_value=dict(BASE_REPORT)):
            with self.assertRaises(baseline.BaselineError) as cm:
                baseline.export_image(d)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("镜像缺少", str(cm.exception))


class TestImageCheck(TempCase):
    def _export(self):
        d = self.image_dir(DF_HEAD + """ARG FLAG_GEMS_COMMIT=%s
LABEL io.flagrt.flag_gems.commit="${FLAG_GEMS_COMMIT}"
COPY src/FlagGems/ /opt/src/FlagGems/
RUN ${PYTHON_BIN} -m pip install /opt/src/FlagGems
""" % ("3" * 40), tag="my/img:1")
        labels = {"io.flagrt.flag_gems.commit": "3" * 40}
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels)), \
             mock.patch.object(baseline, "image_python_report", return_value=dict(BASE_REPORT)), \
             mock.patch.object(baseline, "repo_info", return_value=None):
            baseline.export_image(d)
        return d, labels

    def test_check_image_pass_and_fail(self):
        d, labels = self._export()
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels)):
            self.assertEqual(baseline.check_image(d), 0)
        with mock.patch.object(baseline, "inspect_image",
                               fake_inspect(labels, image_id="sha256:" + "f" * 64)):
            self.assertEqual(baseline.check_image(d), 1)
        with mock.patch.object(baseline, "inspect_image",
                               fake_inspect({"io.flagrt.flag_gems.commit": "4" * 40})):
            self.assertEqual(baseline.check_image(d), 1)

    def test_show_image_is_read_only(self):
        d, _ = self._export()
        before = (d / "image.baseline.yaml").read_text(encoding="utf-8")
        self.assertEqual(baseline.show(d, image=True), 0)
        self.assertEqual((d / "image.baseline.yaml").read_text(encoding="utf-8"), before)

    def test_list_lists_every_image_baseline(self):
        d, _ = self._export()
        with mock.patch.object(baseline, "REPO_ROOT", self.tmp):
            self.assertEqual(baseline.list_baselines(), 0)


# --------------------------------------------------------------------------
# §4.3 运行环境场景
# --------------------------------------------------------------------------

class TestEnvScenario(TempCase):
    def setUp(self):
        super().setUp()
        self.ws = self.tmp / "ws"
        self.commit = make_repo(self.ws / "vllm-plugin-FL")
        self.image_baseline = self.tmp / "image.baseline.yaml"
        self.image_baseline.write_text(json.dumps({
            "schema": baseline.SCHEMA, "kind": "image",
            "image": {"ref": "flagrt/img:1", "id": "sha256:" + "a" * 64, "recipe": []},
            "baked_repos": {"vllm-plugin-FL": {"layer": "L3", "from": "img/v3 ...",
                                               "commit": "9" * 40,
                                               "install": "pip (non-editable)"}},
            "baked_resources": [],
        }), encoding="utf-8")
        self.env = self.tmp / "demo"
        self.env.mkdir()
        (self.env / "docker-compose.yml").write_text(f"""services:
  dev:
    image: flagrt/img:1
    volumes:
      - ${{WORKSPACE_ROOT}}/vllm-plugin-FL:/workspace/vllm-plugin-FL
x-flagrt:
  image_baseline: ../image.baseline.yaml
  mounted_repos:
    vllm-plugin-FL: {{activation: pip-editable}}
""", encoding="utf-8")
        # docker compose config 的解析结果（真实调用见 TestComposeBinds）
        self.binds = [((self.ws / "vllm-plugin-FL").resolve(), Path("/workspace/vllm-plugin-FL"))]
        p = mock.patch.object(baseline, "compose_binds", side_effect=lambda _c: self.binds)
        p.start()
        self.addCleanup(p.stop)

    def _export(self, **kw):
        with mock.patch.object(baseline, "repo_info", return_value={"branch": "main"}):
            return baseline.export_env(self.env, **kw)

    def test_export_records_mounted_and_copies_baked(self):
        self.assertEqual(self._export(), 0)
        data = baseline.load_yaml(self.env / "env.baseline.yaml")
        self.assertEqual(data["kind"], "container")
        self.assertEqual(data["image"]["id"], "sha256:" + "a" * 64)
        self.assertEqual(data["baked_repos"]["vllm-plugin-FL"]["commit"], "9" * 40)
        m = data["mounted_repos"]["vllm-plugin-FL"]
        self.assertEqual(m["commit"], self.commit)
        self.assertFalse(m["dirty"])
        self.assertTrue(m["overrides_baked"])
        self.assertEqual(m["container_path"], "/workspace/vllm-plugin-FL")
        self.assertEqual(m["activation"], "pip-editable")

    def test_export_dirty_warns_and_require_clean_refuses(self):
        (self.ws / "vllm-plugin-FL" / "a.txt").write_text("dirty\n", encoding="utf-8")
        self.assertEqual(self._export(), 0)
        data = baseline.load_yaml(self.env / "env.baseline.yaml")
        self.assertTrue(data["mounted_repos"]["vllm-plugin-FL"]["dirty"])
        self.assertEqual(self._export(require_clean=True), 1)

    def test_export_separator_and_unpushed(self):
        # 本地新建的 repo 没有远端：pushed=false，--require-clean 拒绝
        self.assertEqual(self._export(), 0)
        text = (self.env / "env.baseline.yaml").read_text(encoding="utf-8")
        self.assertIn("环境挂载（env）", text)
        self.assertLess(text.index("baked_repos:"), text.index("环境挂载（env）"))
        self.assertLess(text.index("环境挂载（env）"), text.index("mounted_repos:"))
        data = baseline.load_yaml(self.env / "env.baseline.yaml")
        self.assertFalse(data["mounted_repos"]["vllm-plugin-FL"]["pushed"])
        self.assertEqual(self._export(require_clean=True), 1)

    def test_pushed_commit_passes_require_clean(self):
        repo = self.ws / "vllm-plugin-FL"
        remote = self.tmp / "remote.git"
        git("init", "-q", "--bare", str(remote), cwd=self.tmp)
        git("remote", "add", "origin", str(remote), cwd=repo)
        git("push", "-q", "origin", "HEAD:refs/heads/dev", cwd=repo)
        git("fetch", "-q", "origin", cwd=repo)
        self.assertEqual(self._export(require_clean=True), 0)
        data = baseline.load_yaml(self.env / "env.baseline.yaml")
        self.assertTrue(data["mounted_repos"]["vllm-plugin-FL"]["pushed"])

    def test_check_env_commit_mismatch_and_last_check(self):
        self._export()
        self.assertEqual(self._check(), 0)
        last = json.loads((self.env / ".last-check.json").read_text(encoding="utf-8"))
        self.assertEqual(last["result"], "PASS")
        repo = self.ws / "vllm-plugin-FL"
        (repo / "b.txt").write_text("y\n", encoding="utf-8")
        git("add", "-A", cwd=repo)
        git("commit", "-qm", "second", cwd=repo)
        self.assertEqual(self._check(), 1)
        last = json.loads((self.env / ".last-check.json").read_text(encoding="utf-8"))
        self.assertEqual(last["result"], "FAIL")

    def test_check_env_dirty_is_warning_only(self):
        self._export()
        (self.ws / "vllm-plugin-FL" / "a.txt").write_text("dirty\n", encoding="utf-8")
        self.assertEqual(self._check(), 0)

    def _check(self):
        with mock.patch.object(baseline, "inspect_image", fake_inspect()):
            return baseline.check_env(self.env)

    def test_load_aborts_on_dirty_without_switching(self):
        repo = self.ws / "vllm-plugin-FL"
        self._export()
        (repo / "b.txt").write_text("y\n", encoding="utf-8")
        git("add", "-A", cwd=repo)
        git("commit", "-qm", "second", cwd=repo)
        second = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo),
                                capture_output=True, text=True).stdout.strip()
        (repo / "a.txt").write_text("dirty\n", encoding="utf-8")
        self.assertEqual(baseline.load_env(self.env), 3)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo),
                              capture_output=True, text=True).stdout.strip()
        self.assertEqual(head, second, "load 被中止时不得做任何切换")

    def test_load_switches_to_detached_head(self):
        repo = self.ws / "vllm-plugin-FL"
        self._export()
        (repo / "b.txt").write_text("y\n", encoding="utf-8")
        git("add", "-A", cwd=repo)
        git("commit", "-qm", "second", cwd=repo)
        self.assertEqual(baseline.load_env(self.env), 0)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo),
                              capture_output=True, text=True).stdout.strip()
        self.assertEqual(head, self.commit)
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo),
                                capture_output=True, text=True).stdout.strip()
        self.assertEqual(branch, "HEAD", "切换结果应为 detached HEAD")

    def test_load_fetches_missing_commit(self):
        origin = self.tmp / "origin"
        make_repo(origin)
        clone = self.ws / "cloned"
        subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True,
                       capture_output=True)
        (origin / "c.txt").write_text("z\n", encoding="utf-8")
        git("add", "-A", cwd=origin)
        git("commit", "-qm", "third", cwd=origin)
        want = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(origin),
                              capture_output=True, text=True).stdout.strip()
        missing = subprocess.run(["git", "cat-file", "-e", want + "^{commit}"],
                                 cwd=str(clone), capture_output=True)
        self.assertNotEqual(missing.returncode, 0, "前置条件：本机确实缺少该 commit")
        (self.env / "env.baseline.yaml").write_text(json.dumps({
            "schema": baseline.SCHEMA, "kind": "container",
            "image": {"ref": "flagrt/img:1", "id": "sha256:" + "a" * 64},
            "baked_repos": {},
            "mounted_repos": {"cloned": {"host_path": "cloned", "commit": want,
                                         "activation": "pythonpath", "dirty": False}},
        }), encoding="utf-8")
        self.assertEqual(baseline.load_env(self.env), 0)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(clone),
                              capture_output=True, text=True).stdout.strip()
        self.assertEqual(head, want)

    # ---- 挂载来源必须就是被核对的宿主目录（WORKSPACE_ROOT 分叉） ----
    def test_check_env_mount_source_mismatch_fails(self):
        self._export()
        other = make_repo(self.tmp / "other" / "vllm-plugin-FL")
        self.assertTrue(other)
        self.binds = [((self.tmp / "other" / "vllm-plugin-FL").resolve(),
                       Path("/workspace/vllm-plugin-FL"))]
        self.assertEqual(self._check(), 1)
        last = json.loads((self.env / ".last-check.json").read_text(encoding="utf-8"))
        self.assertTrue(any(r["status"] == "FAIL" and "挂载来源" in r["detail"]
                            for r in last["report"]))

    def test_check_env_parent_mount_covers_repo(self):
        self.binds = [(self.ws.resolve(), Path("/workspace"))]
        self._export()
        data = baseline.load_yaml(self.env / "env.baseline.yaml")
        self.assertEqual(data["mounted_repos"]["vllm-plugin-FL"]["container_path"],
                         "/workspace/vllm-plugin-FL")
        self.assertEqual(self._check(), 0)

    def test_check_env_unmounted_repo_fails(self):
        self._export()
        self.binds = []
        self.assertEqual(self._check(), 1)

    def test_compose_config_failure_is_exit_2(self):
        self._export()
        with mock.patch.object(baseline, "compose_binds",
                               side_effect=baseline.BaselineError("docker compose config 失败")), \
             mock.patch.object(baseline, "inspect_image", fake_inspect()):
            self.assertEqual(baseline.main(["check", str(self.env)]), 2)

    # ---- 覆盖固化 repo 时补丁可见 ----
    def test_baked_patches_copied_and_check_warns(self):
        data = json.loads(self.image_baseline.read_text(encoding="utf-8"))
        data["baked_repos"]["vllm-plugin-FL"]["patches"] = [
            {"file": "0001-x.patch", "sha256": "5" * 64}]
        self.image_baseline.write_text(json.dumps(data), encoding="utf-8")
        self._export()
        m = baseline.load_yaml(self.env / "env.baseline.yaml")["mounted_repos"]["vllm-plugin-FL"]
        self.assertEqual(m["baked_patches"], [{"file": "0001-x.patch", "sha256": "5" * 64}])
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(self._check(), 0, "补丁缺失只告警、不算不一致")
        self.assertIn("[WARN] vllm-plugin-FL: 挂载副本不含镜像内固化的补丁: 0001-x.patch",
                      out.getvalue())

    def test_no_baked_patches_field_without_patches(self):
        self._export()
        m = baseline.load_yaml(self.env / "env.baseline.yaml")["mounted_repos"]["vllm-plugin-FL"]
        self.assertNotIn("baked_patches", m)

    # ---- load 必须验证检出完整 ----
    def test_load_fails_on_incomplete_checkout(self):
        repo = self.ws / "vllm-plugin-FL"
        self._export()
        (repo / "b.txt").write_text("y\n", encoding="utf-8")
        git("add", "-A", cwd=repo)
        git("commit", "-qm", "second", cwd=repo)
        real = baseline.run_git

        def broken_checkout(*args, cwd, timeout=60):
            out = real(*args, cwd=cwd, timeout=timeout)
            if args[0] == "checkout":  # 模拟部分克隆：checkout 返回 0 但跟踪文件缺失
                (Path(cwd) / "a.txt").unlink()
            return out

        with mock.patch.object(baseline, "run_git", side_effect=broken_checkout), \
             mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(baseline.load_env(self.env), 1)
        self.assertIn("vllm-plugin-FL", err.getvalue())
        self.assertIn("检出不完整", err.getvalue())

    def test_show_env(self):
        self._export()
        self.assertEqual(baseline.show(self.env, image=False), 0)

    def test_compose_without_x_flagrt_is_format_error(self):
        (self.env / "docker-compose.yml").write_text("services: {dev: {image: x}}\n",
                                                     encoding="utf-8")
        rc = baseline.main(["export", str(self.env)])
        self.assertEqual(rc, 2)

    def test_malformed_compose_yaml_is_format_error(self):
        (self.env / "docker-compose.yml").write_text("x-flagrt: [\n", encoding="utf-8")
        self.assertEqual(baseline.main(["check", str(self.env)]), 2)

    def test_mounted_repos_come_only_from_x_flagrt(self):
        """仓根整体挂载不等于声明挂载（ENV-SPEC §2.1）。"""
        (self.env / "docker-compose.yml").write_text(f"""services:
  dev:
    image: flagrt/img:1
    volumes:
      - ${{WORKSPACE_ROOT}}:/workspace
      - ${{WORKSPACE_ROOT}}/vllm-plugin-FL:/workspace/vllm-plugin-FL
x-flagrt:
  image_baseline: ../image.baseline.yaml
  mounted_repos: {{}}
""", encoding="utf-8")
        self.assertEqual(self._export(), 0)
        data = baseline.load_yaml(self.env / "env.baseline.yaml")
        self.assertEqual(data["mounted_repos"], {})


class TestComposeBinds(TempCase):
    def _fake_run(self, rc=0):
        seen = {}

        def run(cmd, **kw):
            seen["cmd"], seen["env"] = cmd, kw.get("env")
            out = json.dumps({"services": {"dev": {"volumes": [
                {"type": "bind", "source": "/srv/ws/vllm-plugin-FL",
                 "target": "/workspace/vllm-plugin-FL"},
                {"type": "volume", "source": "data", "target": "/data"}]}}})
            return subprocess.CompletedProcess(cmd, rc, out if rc == 0 else "", "boom")
        return run, seen

    def test_parses_bind_mounts_with_workspace_root_default(self):
        run, seen = self._fake_run()
        env = {k: v for k, v in os.environ.items() if k != "WORKSPACE_ROOT"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(baseline.subprocess, "run", side_effect=run):
            binds = baseline.compose_binds(Path("/x/docker-compose.yml"))
        self.assertEqual(binds, [(Path("/srv/ws/vllm-plugin-FL").resolve(),
                                  Path("/workspace/vllm-plugin-FL"))])
        self.assertEqual(seen["cmd"][:4], ["docker", "compose", "-f", "/x/docker-compose.yml"])
        self.assertEqual(seen["env"]["WORKSPACE_ROOT"], str(self.tmp / "ws"))

    def test_failure_is_format_error(self):
        run, _ = self._fake_run(rc=1)
        with mock.patch.object(baseline.subprocess, "run", side_effect=run):
            with self.assertRaises(baseline.BaselineError) as cm:
                baseline.compose_binds(Path("/x/docker-compose.yml"))
        self.assertEqual(cm.exception.code, 2)


class TestUpSh(TempCase):
    """up.sh 给 compose 的 WORKSPACE_ROOT 与 baseline.py 用的 FLAGRT_WORKSPACE_ROOT 同源。"""

    def _run_up(self, extra_env):
        lib = self.tmp / "dev" / "lib"  # up.sh 以 lib/../.. 为仓根
        lib.mkdir(parents=True)
        (lib / "up.sh").write_text((LIB / "up.sh").read_text(encoding="utf-8"), encoding="utf-8")
        (lib / "baseline.py").write_text(
            "import os\nprint('CHECK', os.environ.get('FLAGRT_WORKSPACE_ROOT'),"
            " os.environ.get('WORKSPACE_ROOT'))\n", encoding="utf-8")
        bindir = self.tmp / "bin"
        bindir.mkdir()
        (bindir / "docker").write_text('#!/bin/sh\necho "DOCKER $WORKSPACE_ROOT"\n',
                                       encoding="utf-8")
        (bindir / "docker").chmod(0o755)
        envdir = self.tmp / "envdir"
        envdir.mkdir()
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", **extra_env)
        env.pop("WORKSPACE_ROOT", None)
        if "FLAGRT_WORKSPACE_ROOT" not in extra_env:
            env.pop("FLAGRT_WORKSPACE_ROOT", None)
        return subprocess.run(["bash", str(lib / "up.sh"), str(envdir)], env=env,
                              capture_output=True, text=True).stdout

    def test_flagrt_workspace_root_drives_compose(self):
        out = self._run_up({"FLAGRT_WORKSPACE_ROOT": "/srv/other"})
        self.assertIn("CHECK /srv/other /srv/other", out)
        self.assertIn("DOCKER /srv/other", out)

    def test_default_is_repo_root(self):
        out = self._run_up({})
        root = str(self.tmp.resolve())
        self.assertIn(f"CHECK {root} {root}", out)
        self.assertIn(f"DOCKER {root}", out)


class TestInfoBlock(TempCase):
    def test_info_from_host_checkout(self):
        commit = make_repo(self.tmp / "ws" / "FlagGems")
        info = baseline.repo_info("FlagGems", commit)
        self.assertEqual(info["branch"], "main")
        self.assertIn("20", info["committed_at"])

    def test_info_unavailable_without_checkout_or_url(self):
        self.assertIsNone(baseline.repo_info("FlagGems", "d" * 40))

    def test_info_branch_hint_fallback(self):
        info = baseline.repo_info("FlagGems", "d" * 40, branch_hint="triton_v3.5.x")
        self.assertEqual(info, {"branch": "triton_v3.5.x"})


# --------------------------------------------------------------------------
# §4.2 模式 6（pip 索引发行包）与无 commit 资产的版本锚 —— 昆仑芯接入带出来的规则
# --------------------------------------------------------------------------

KL_DF = """ARG BASE_IMAGE=ext/kbase:1
FROM ${BASE_IMAGE}
ARG PYTHON_BIN=/opt/conda/envs/py310/bin/python3
ARG FLAGTREE_VERSION=0.6.1+xpu3.6
ARG FLAGTREE_INDEX_URL=https://idx.example/simple
LABEL io.flagrt.flagtree.version="${FLAGTREE_VERSION}"
RUN ${PYTHON_BIN} -m pip install --no-deps "flagtree===${FLAGTREE_VERSION}" \\
      --index-url=${FLAGTREE_INDEX_URL}
"""


class TestIndexPackagePattern(TempCase):
    """§4.2 模式 6：pip install <pkg>===<version>，版本锚取代 commit 锚。"""

    def _report(self, installed_flagtree="0.6.1+xpu3.6", **kw):
        r = {"python_version": "3.10.18",
             "versions": {"torch": "2.9.0+cu129", **kw.pop("versions", {})},
             "editables": {}, "base_files": {}}
        if installed_flagtree:
            r["versions"]["flagtree"] = installed_flagtree
        r.update(kw)
        return r

    def test_recipe_records_package_with_version_and_index(self):
        r = baseline.analyze_dockerfile(self.image_dir(KL_DF))
        pkg = r.packages["flagtree"]
        self.assertEqual(pkg["version"], "0.6.1+xpu3.6")
        self.assertEqual(pkg["index_url"], "https://idx.example/simple")
        self.assertEqual(pkg["install"], "pip (index wheel, --no-deps)")
        self.assertEqual(r.repos, {})          # 不是 repo，不进 baked_repos

    def test_index_url_as_separate_token_is_not_a_second_target(self):
        """G10：`-i <url>` 的取值不能被当成第二个安装目标。"""
        d = self.image_dir(KL_DF.replace('--index-url=${FLAGTREE_INDEX_URL}',
                                         '-i ${FLAGTREE_INDEX_URL}'))
        self.assertEqual(baseline.analyze_dockerfile(d).packages["flagtree"]["index_url"],
                         "https://idx.example/simple")

    def test_export_writes_baked_packages_block(self):
        d = self.image_dir(KL_DF, tag="flagrt/kl:2")
        labels = {"io.flagrt.flagtree.version": "0.6.1+xpu3.6"}
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels, env=["PATH=/usr/bin"])), \
             mock.patch.object(baseline, "image_python_report", return_value=self._report()):
            self.assertEqual(baseline.export_image(d), 0)
        data = baseline.load_yaml(d / "image.baseline.yaml")
        self.assertEqual(data["baked_repos"], {})
        pkg = data["baked_packages"][0]
        self.assertEqual(pkg["name"], "flagtree")
        self.assertEqual(pkg["version"], "0.6.1+xpu3.6")
        self.assertEqual(pkg["layer"], "L2")           # LAYER 表里登记过
        self.assertEqual(pkg["index_url"], "https://idx.example/simple")
        self.assertIn("io.flagrt.flagtree.version", pkg["pin"])
        # 本用例走的是「取不到 wheel sha256」的降级路径（TempCase 默认不联网）：
        # 不写 wheel 块、pin 里说明未取到，但 export 仍成功 —— 离线建得出镜像就该导得出基线。
        self.assertNotIn("wheel", pkg)
        self.assertIn("未取到 wheel sha256", pkg["pin"])

    def test_export_fails_when_image_label_version_differs(self):
        """G14：版本锚标签与 ARG 不一致要报错（此前版本类 LABEL 完全不参与核对）。"""
        d = self.image_dir(KL_DF, tag="flagrt/kl:2")
        labels = {"io.flagrt.flagtree.version": "0.7.0rc3+xpu3.6"}
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels, env=["PATH=/x"])), \
             mock.patch.object(baseline, "image_python_report", return_value=self._report()):
            with self.assertRaises(baseline.BaselineError) as cm:
                baseline.export_image(d)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("交叉核对失败", str(cm.exception))

    def test_export_fails_when_installed_version_differs(self):
        """第三条边：镜像内实装 dist 版本与标签/ARG 不一致要报错。"""
        d = self.image_dir(KL_DF, tag="flagrt/kl:2")
        labels = {"io.flagrt.flagtree.version": "0.6.1+xpu3.6"}
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels, env=["PATH=/x"])), \
             mock.patch.object(baseline, "image_python_report",
                               return_value=self._report(installed_flagtree="0.5.0+xpu3.6")):
            with self.assertRaises(baseline.BaselineError) as cm:
                baseline.export_image(d)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("实装版本", str(cm.exception))

    def test_export_fails_when_package_absent_from_image(self):
        d = self.image_dir(KL_DF, tag="flagrt/kl:2")
        labels = {"io.flagrt.flagtree.version": "0.6.1+xpu3.6"}
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels, env=["PATH=/x"])), \
             mock.patch.object(baseline, "image_python_report",
                               return_value=self._report(installed_flagtree=None)):
            with self.assertRaises(baseline.BaselineError) as cm:
                baseline.export_image(d)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("未安装", str(cm.exception))

    def test_unregistered_package_points_at_where_to_register(self):
        d = self.image_dir(KL_DF.replace("flagtree", "mystery"), tag="flagrt/kl:2")
        with mock.patch.object(baseline, "inspect_image", fake_inspect({}, env=["PATH=/x"])), \
             mock.patch.object(baseline, "image_python_report", return_value=self._report()):
            with self.assertRaises(baseline.BaselineError) as cm:
                baseline.export_image(d)
        self.assertIn("VERSION_LABEL", str(cm.exception))

    def test_check_image_verifies_version_label(self):
        d = self.image_dir(KL_DF, tag="flagrt/kl:2")
        labels = {"io.flagrt.flagtree.version": "0.6.1+xpu3.6"}
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels, env=["PATH=/x"])), \
             mock.patch.object(baseline, "image_python_report", return_value=self._report()):
            self.assertEqual(baseline.export_image(d), 0)
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels, env=["PATH=/x"])):
            self.assertEqual(baseline.check_image(d), 0)
        # 标签漂了：check 必须 FAIL，而不是「无 commit 标签，跳过」
        with mock.patch.object(baseline, "inspect_image",
                               fake_inspect({"io.flagrt.flagtree.version": "9.9.9"},
                                            env=["PATH=/x"])):
            self.assertEqual(baseline.check_image(d), 1)
        # 标签整个丢了：同样 FAIL —— 它是该资产唯一的版本锚
        with mock.patch.object(baseline, "inspect_image", fake_inspect({}, env=["PATH=/x"])):
            self.assertEqual(baseline.check_image(d), 1)

    def test_check_image_accepts_base_image_commit_without_label(self):
        """基座继承来的 commit 取自 pip 元数据，本来就没有标签可核对 -> INFO，不是 FAIL。

        昇腾 v3 的 vllm 条目正是这个形状（from: base_image + commit + 无标签）；把它当
        FAIL 会让昇腾侧 check 从 PASS 变 FAIL。"""
        d = self.image_dir(DF_HEAD, name="img3")
        (d / "image.baseline.yaml").write_text(json.dumps({
            "schema": baseline.SCHEMA, "kind": "image",
            "image": {"ref": "flagrt/img3:1", "id": "sha256:" + "a" * 64},
            "baked_repos": {"vllm": {"layer": "L3", "from": "base_image", "commit": "c" * 40,
                                     "install": "editable-in-image"}},
        }), encoding="utf-8")
        with mock.patch.object(baseline, "inspect_image", fake_inspect({})):
            self.assertEqual(baseline.check_image(d), 0)

    def test_check_image_fails_when_commit_label_lost(self):
        """G13：基线记了 commit 但镜像没有该标签时，不能再以 INFO「跳过」放过。"""
        d = self.image_dir(DF_HEAD, name="img2")
        (d / "image.baseline.yaml").write_text(json.dumps({
            "schema": baseline.SCHEMA, "kind": "image",
            "image": {"ref": "flagrt/img2:1", "id": "sha256:" + "a" * 64},
            "baked_repos": {"FlagGems": {"layer": "L2", "from": "img2/v3 Dockerfile.repro(...)",
                                         "commit": "3" * 40,
                                         "install": "pip (non-editable)"}},
        }), encoding="utf-8")
        with mock.patch.object(baseline, "inspect_image", fake_inspect({})):
            self.assertEqual(baseline.check_image(d), 1)

    def test_package_cannot_be_a_mounted_repo_key(self):
        """G17：索引发行包不可下放为挂载资产。"""
        img = self.tmp / "image.baseline.yaml"
        img.write_text(json.dumps({
            "schema": baseline.SCHEMA, "kind": "image",
            "image": {"ref": "flagrt/kl:2", "id": "sha256:" + "a" * 64},
            "baked_repos": {},
            "baked_packages": [{"name": "flagtree", "layer": "L2", "version": "0.6.1+xpu3.6"}],
        }), encoding="utf-8")
        env = self.tmp / "demo"
        env.mkdir()
        (env / "docker-compose.yml").write_text("""services:
  dev: {image: 'flagrt/kl:2'}
x-flagrt:
  image_baseline: ../image.baseline.yaml
  mounted_repos:
    flagtree: {activation: pip-editable}
""", encoding="utf-8")
        with mock.patch.object(baseline, "compose_binds", return_value=[]):
            with self.assertRaises(baseline.BaselineError) as cm:
                baseline.export_env(env)
        self.assertIn("不可下放", str(cm.exception))

    def test_env_export_copies_baked_packages(self):
        img = self.tmp / "image.baseline.yaml"
        img.write_text(json.dumps({
            "schema": baseline.SCHEMA, "kind": "image",
            "image": {"ref": "flagrt/kl:2", "id": "sha256:" + "a" * 64},
            "baked_repos": {},
            "baked_packages": [{"name": "flagtree", "layer": "L2", "version": "0.6.1+xpu3.6"}],
        }), encoding="utf-8")
        env = self.tmp / "demo"
        env.mkdir()
        (env / "docker-compose.yml").write_text("""services:
  dev: {image: 'flagrt/kl:2'}
x-flagrt:
  image_baseline: ../image.baseline.yaml
  mounted_repos: {}
""", encoding="utf-8")
        with mock.patch.object(baseline, "compose_binds", return_value=[]):
            self.assertEqual(baseline.export_env(env), 0)
        data = baseline.load_yaml(env / "env.baseline.yaml")
        self.assertEqual(data["baked_packages"][0]["version"], "0.6.1+xpu3.6")
        self.assertEqual(data["mounted_repos"], {})


class TestBaseImageFacts(TempCase):
    """父基座事实的取数：声明式探针表 + pip 发行名归一化（原先全是昇腾专属硬编码）。"""

    def _export(self, report, labels=None, env=None, df=None):
        d = self.image_dir(df or DF_HEAD, tag="my/img:1")
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels or {}, env=env)), \
             mock.patch.object(baseline, "image_python_report", return_value=report), \
             mock.patch.object(baseline, "repo_info", return_value=None):
            self.assertEqual(baseline.export_image(d), 0)
        return baseline.load_yaml(d / "image.baseline.yaml")

    def test_file_probe_supplies_l1_versions(self):
        """G6：昆仑芯的 L1 版本事实在镜像内文件里，不在任何环境变量里。"""
        data = self._export({
            "python_version": "3.10.18", "versions": {"torch": "2.9.0+cu129"}, "editables": {},
            "base_files": {"torch_xmlir/_versions.txt":
                           "XTDK: 3.6.0.1\nXRE: 5.13.0.0\nXCCL: 20260209_9\nOTHER: x\n"},
        }, env=["PATH=/usr/bin"])
        provides = data["base_image"]["provides"]
        self.assertEqual(provides["xtdk"], "3.6.0.1")
        self.assertEqual(provides["xre"], "5.13.0.0")
        self.assertEqual(provides["xccl"], "20260209_9")
        self.assertNotIn("cann", provides)      # 昆仑芯没有 CANN
        self.assertNotIn("other", provides)     # 未登记的键不收录

    def test_env_probe_still_supplies_cann(self):
        """昇腾路径不回归：CANN 仍从 ASCEND_TOOLKIT_HOME 取。"""
        data = self._export({"python_version": "3.11.15",
                             "versions": {"torch": "2.10.0+cpu", "torch_npu": "2.10.0"},
                             "editables": {}, "base_files": {}})
        self.assertEqual(data["base_image"]["provides"]["cann"], "9.0.0")
        self.assertEqual(data["base_image"]["provides"]["torch_npu"], "2.10.0")

    def test_provides_covers_more_packages(self):
        """G7：provides 的包白名单不再只有 torch/torch_npu/vllm。"""
        data = self._export({
            "python_version": "3.10.18", "editables": {}, "base_files": {},
            "versions": {"torch": "2.9.0+cu129", "torchvision": "0.24.0+cu129",
                         "torchaudio": "2.9.0+cu129", "xmlir": "1.0"},
        }, env=["PATH=/usr/bin"])
        provides = data["base_image"]["provides"]
        self.assertEqual(provides["torchvision"], "0.24.0+cu129")
        self.assertEqual(provides["torchaudio"], "2.9.0+cu129")
        self.assertEqual(provides["xmlir"], "1.0")

    def test_editable_dist_name_is_normalised_to_repo_name(self):
        """G5：pip 发行名 flag_gems 要归一成仓库名 FlagGems，层级表才认得出 L2。"""
        data = self._export({
            "python_version": "3.10.18", "versions": {"flag_gems": "4.2.1rc0"},
            "base_files": {},
            "editables": {"flag_gems": {"source": "file:///env/FlagGems", "commit": "7" * 40}},
        }, env=["PATH=/usr/bin"])
        self.assertIn("FlagGems", data["baked_repos"])
        self.assertNotIn("flag_gems", data["baked_repos"])
        self.assertEqual(data["baked_repos"]["FlagGems"]["layer"], "L2")
        self.assertEqual(data["baked_repos"]["FlagGems"]["commit"], "7" * 40)
        self.assertEqual(data["baked_repos"]["FlagGems"]["source_path"], "/env/FlagGems")

    def test_editable_without_commit_records_unavailable_not_null(self):
        """G2：取不到 commit 要显式 unavailable，不能静默写 null。"""
        data = self._export({
            "python_version": "3.10.18", "versions": {"vllm_fl": "0.1.0"}, "base_files": {},
            "editables": {"vllm_fl": {"source": "file:///env/xvllm-plugin-FL", "commit": None}},
        }, env=["PATH=/usr/bin"])
        entry = data["baked_repos"]["vllm_fl"]
        self.assertEqual(entry["commit"], "unavailable")
        self.assertEqual(entry["install"], "editable-in-image")
        self.assertNotIn("version: null", (self.tmp / "dev/images/img/v3/image.baseline.yaml")
                         .read_text(encoding="utf-8"))

    def test_subdirectory_fragment_keeps_commit(self):
        """G3：`#egg=flagcx&subdirectory=plugin/torch` 此前整行不匹配，commit 被吞掉。"""
        def _run(*args):
            if args[:2] == ("-m", "pip") and args[2] == "inspect":
                return json.dumps({"installed": [{"metadata": {"name": "flagcx",
                                                               "version": "0.1"},
                                                  "direct_url": {"url": "file:///env/FlagCX",
                                                                 "dir_info": {"editable": True}}}],
                                   "environment": {"python_full_version": "3.10.18"}})
            return ("-e git+https://github.com/flagos-ai/FlagCX.git@" + "3" * 40
                    + "#egg=flagcx&subdirectory=plugin/torch\n")

        with mock.patch.object(baseline.subprocess, "run") as m:
            m.side_effect = lambda cmd, **kw: mock.Mock(
                returncode=0, stdout=_run(*cmd[cmd.index("ext/base:1") + 1:]), stderr="")
            report = baseline.image_python_report("ext/base:1", "/usr/bin/python3")
        self.assertEqual(report["editables"]["flagcx"]["commit"], "3" * 40)
        self.assertEqual(report["editables"]["flagcx"]["package_subdir"], "plugin/torch")

    def test_this_layer_supersedes_inherited_same_name_asset(self):
        """G4：父基座 editable 装了 FlagGems，本层又 COPY 固化同名资产 -> 只留一条 + 留痕。"""
        df = (DF_HEAD + """ARG FLAG_GEMS_COMMIT=%s
LABEL io.flagrt.flag_gems.commit="${FLAG_GEMS_COMMIT}"
COPY src/FlagGems/ /opt/src/FlagGems/
RUN ${PYTHON_BIN} -m pip install --no-deps /opt/src/FlagGems
""" % ("4" * 40))
        data = self._export({
            "python_version": "3.10.18", "versions": {"flag_gems": "4.2.1rc0"},
            "base_files": {},
            "editables": {"flag_gems": {"source": "file:///env/FlagGems", "commit": "7" * 40}},
        }, labels={"io.flagrt.flag_gems.commit": "4" * 40}, env=["PATH=/usr/bin"], df=df)
        entries = [n for n in data["baked_repos"]]
        self.assertEqual(entries.count("FlagGems"), 1)
        entry = data["baked_repos"]["FlagGems"]
        self.assertEqual(entry["commit"], "4" * 40)            # 本层的 commit 赢
        self.assertEqual(entry["install"], "pip (non-editable)")
        self.assertIn("base_image", entry["supersedes"])       # 被覆盖的那条留痕
        self.assertIn("7" * 12, entry["supersedes"])


# --------------------------------------------------------------------------
# §3.1 provenance 分组（分割线）与继承资产的 commit 断言
# --------------------------------------------------------------------------

BASE_HEADER = "厂商基座自带"
RECIPE_HEADER = "本组配方带入"
GEMS_DF = (DF_HEAD + 'ARG FLAG_GEMS_COMMIT=%s\n'
           'LABEL io.flagrt.flag_gems.commit="${FLAG_GEMS_COMMIT}"\n')
GEMS_COPY = """COPY src/FlagGems/ /opt/src/FlagGems/
RUN ${PYTHON_BIN} -m pip install --no-deps /opt/src/FlagGems
"""


def base_report(versions=None, editables=None):
    return {"python_version": "3.10.18", "versions": dict(versions or {}),
            "editables": dict(editables or {}), "base_files": {}}


class TestBakedReposGrouping(TempCase):
    """ENV-SPEC §3.1：baked_repos 按 provenance 分两组输出，组内保持原顺序（不重排）。"""

    def _export(self, df, report, labels=None):
        d = self.image_dir(df, tag="my/img:1")
        with mock.patch.object(baseline, "inspect_image",
                               fake_inspect(labels or {}, env=["PATH=/usr/bin"])), \
             mock.patch.object(baseline, "image_python_report", return_value=report), \
             mock.patch.object(baseline, "repo_info", return_value=None):
            self.assertEqual(baseline.export_image(d), 0)
        return d, (d / "image.baseline.yaml").read_text(encoding="utf-8")

    def test_both_groups_get_headers_vendor_side_first(self):
        d, text = self._export(
            GEMS_DF % ("4" * 40) + GEMS_COPY,
            base_report({"vllm": "0.20.2"},
                        {"vllm": {"source": "file:///vllm-workspace/vllm", "commit": "c" * 40}}),
            labels={"io.flagrt.flag_gems.commit": "4" * 40})
        for marker in (BASE_HEADER, "\n  vllm:", RECIPE_HEADER, "\n  FlagGems:"):
            self.assertIn(marker, text)
        self.assertLess(text.index(BASE_HEADER), text.index("\n  vllm:"))
        self.assertLess(text.index("\n  vllm:"), text.index(RECIPE_HEADER))
        self.assertLess(text.index(RECIPE_HEADER), text.index("\n  FlagGems:"))
        # 分组只是分组，条目本身不变
        self.assertEqual(list(baseline.load_yaml(d / "image.baseline.yaml")["baked_repos"]),
                         ["vllm", "FlagGems"])

    def test_only_recipe_group_still_prints_its_header(self):
        _d, text = self._export(GEMS_DF % ("4" * 40) + GEMS_COPY, base_report(),
                                labels={"io.flagrt.flag_gems.commit": "4" * 40})
        self.assertIn(RECIPE_HEADER, text)
        self.assertNotIn(BASE_HEADER, text)

    def test_only_base_group_still_prints_its_header(self):
        _d, text = self._export(DF_HEAD, base_report(
            {"vllm": "0.20.2"},
            {"vllm": {"source": "file:///vllm-workspace/vllm", "commit": "c" * 40}}))
        self.assertIn(BASE_HEADER, text)
        self.assertNotIn(RECIPE_HEADER, text)

    def test_empty_baked_repos_prints_no_header(self):
        _d, text = self._export(DF_HEAD, base_report())
        self.assertIn("baked_repos: {}", text)
        self.assertNotIn(BASE_HEADER, text)
        self.assertNotIn(RECIPE_HEADER, text)

    def test_parent_recipe_entries_are_not_pushed_to_vendor_side(self):
        """昇腾形状：父镜像是本仓自建镜像，从父基线继承来的条目 from 是父层的
        Dockerfile.repro(...)，仍属本组可控，不能被误划到厂商侧。"""
        parent = self.image_dir(DF_HEAD, name="parent")
        (parent / "image.baseline.yaml").write_text(json.dumps({
            "schema": baseline.SCHEMA, "kind": "image",
            "image": {"ref": "flagrt/parent:1", "id": "sha256:" + "a" * 64,
                      "recipe": ["dev/images/parent/v3/Dockerfile.repro"]},
            "base_image": {"ref": "ext/base:1", "digest": "sha256:" + "b" * 64,
                           "provides": {"python": "3.11.15"}},
            "baked_repos": {
                "vllm": {"layer": "L3", "from": "base_image", "commit": "c" * 40,
                         "install": "editable-in-image"},
                "FlagTree": {"layer": "L2",
                             "from": "parent/v3 Dockerfile.repro(构建期 git fetch)",
                             "commit": "1" * 40, "install": "pip (non-editable)"},
                "FlagGems": {"layer": "L2",
                             "from": "parent/v3 Dockerfile.repro(COPY 构建上下文)",
                             "commit": "2" * 40, "install": "pip (non-editable)"}},
            "baked_resources": [],
        }), encoding="utf-8")
        child = self.image_dir("""ARG PARENT_IMAGE=flagrt/parent:1
FROM ${PARENT_IMAGE}
ARG PYTHON_BIN=/usr/bin/python3
ARG FLAGCX_COMMIT=%s
LABEL io.flagrt.flagcx.commit="${FLAGCX_COMMIT}"
COPY src/FlagCX/ /opt/src/FlagCX/
RUN cd /opt/src/FlagCX && USE_ASCEND=1 ${PYTHON_BIN} -m pip install .
""" % ("e" * 40), name="child", tag="flagrt/child:1")
        with mock.patch.object(baseline, "REPO_ROOT", self.tmp), \
             mock.patch.object(baseline, "inspect_image",
                               fake_inspect({"io.flagrt.flagcx.commit": "e" * 40})), \
             mock.patch.object(baseline, "repo_info", return_value=None):
            self.assertEqual(baseline.export_image(child), 0)
        out = child / "image.baseline.yaml"
        text = out.read_text(encoding="utf-8")
        # 只有 vllm 在厂商侧；父层配方的 FlagTree/FlagGems 与本层的 FlagCX 都在本组侧
        self.assertLess(text.index(BASE_HEADER), text.index("\n  vllm:"))
        self.assertLess(text.index("\n  vllm:"), text.index(RECIPE_HEADER))
        for name in ("\n  FlagTree:", "\n  FlagGems:", "\n  FlagCX:"):
            self.assertLess(text.index(RECIPE_HEADER), text.index(name))
        self.assertEqual(list(baseline.load_yaml(out)["baked_repos"]),
                         ["vllm", "FlagTree", "FlagGems", "FlagCX"])

    def test_grouping_does_not_reorder_shipped_ascend_baseline(self):
        """昇腾回归：对仓内 ascend-* 基线的条目集合，分组后顺序与现有文件逐项一致。"""
        for rel in ("dev/images/ascend-operator-runtime/v3/image.baseline.yaml",
                    "dev/images/ascend-train-comm/v3/image.baseline.yaml"):
            path = baseline.REPO_ROOT / rel
            if not path.is_file():
                self.skipTest(f"{rel} 不在本机")
            entries = [{"_name": n, **i} for n, i
                       in (baseline.load_yaml(path).get("baked_repos") or {}).items()]
            grouped = [e["_name"] for _g, es in baseline._baked_repo_groups(entries) for e in es]
            self.assertEqual(grouped, [e["_name"] for e in entries], rel)


class TestInheritedCommitAssertion(TempCase):
    """§4.2：对 from: base_image 的条目，本组的 commit 标签是「对基座内容的断言」。"""

    def _export(self, df, report, labels=None):
        """导出成功的路径；返回 (镜像目录, 导出过程打印的内容)。"""
        d = self.image_dir(df, tag="my/img:1")
        with mock.patch.object(baseline, "inspect_image",
                               fake_inspect(labels or {}, env=["PATH=/usr/bin"])), \
             mock.patch.object(baseline, "image_python_report", return_value=report), \
             mock.patch.object(baseline, "repo_info", return_value=None), \
             mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(baseline.export_image(d), 0)
            printed = out.getvalue()
        return d, printed

    def _gems(self, commit):
        return base_report({"flag_gems": "4.2.1rc0"},
                           {"flag_gems": {"source": "file:///env/FlagGems", "commit": commit}})

    def test_label_matching_base_metadata_passes(self):
        d, _out = self._export(GEMS_DF % ("7" * 40), self._gems("7" * 40),
                               labels={"io.flagrt.flag_gems.commit": "7" * 40})
        entry = baseline.load_yaml(d / "image.baseline.yaml")["baked_repos"]["FlagGems"]
        self.assertEqual(entry["from"], "base_image")
        self.assertEqual(entry["commit"], "7" * 40)
        self.assertEqual(entry["install"], "editable-in-image")

    def test_label_mismatch_is_export_error(self):
        d = self.image_dir(GEMS_DF % ("4" * 40), tag="my/img:1")
        with mock.patch.object(baseline, "inspect_image",
                               fake_inspect({"io.flagrt.flag_gems.commit": "4" * 40},
                                            env=["PATH=/usr/bin"])), \
             mock.patch.object(baseline, "image_python_report",
                               return_value=self._gems("7" * 40)), \
             mock.patch.object(baseline, "repo_info", return_value=None):
            with self.assertRaises(baseline.BaselineError) as cm:
                baseline.export_image(d)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("基座内容与本组声明不符", str(cm.exception))
        self.assertIn("7" * 40, str(cm.exception))   # 两个值都要打出来
        self.assertIn("4" * 40, str(cm.exception))

    def test_missing_label_is_not_an_error(self):
        """绝大多数继承资产没有标签（昇腾 v3 的 vllm 即如此）：不核对、不报错。"""
        d, _out = self._export(DF_HEAD, self._gems("7" * 40))
        entry = baseline.load_yaml(d / "image.baseline.yaml")["baked_repos"]["FlagGems"]
        self.assertEqual(entry["commit"], "7" * 40)

    def test_base_commit_unavailable_warns_and_keeps_unavailable(self):
        """基座取不到 commit 时不拿标签值冒充观测值，只追加一条 warning。"""
        d, out = self._export(GEMS_DF % ("7" * 40), self._gems(None),
                              labels={"io.flagrt.flag_gems.commit": "7" * 40})
        entry = baseline.load_yaml(d / "image.baseline.yaml")["baked_repos"]["FlagGems"]
        self.assertEqual(entry["commit"], "unavailable")
        self.assertIn("无法核对", out)
        self.assertIn("io.flagrt.flag_gems.commit", out)

    def test_superseded_base_entry_is_not_asserted_against_this_layer_label(self):
        """本层固化覆盖同名基座资产时，标签属于本层资产，不能再拿去核对基座的 commit。"""
        d = self.image_dir(GEMS_DF % ("4" * 40) + GEMS_COPY, tag="my/img:1")
        with mock.patch.object(baseline, "inspect_image",
                               fake_inspect({"io.flagrt.flag_gems.commit": "4" * 40},
                                            env=["PATH=/usr/bin"])), \
             mock.patch.object(baseline, "image_python_report",
                               return_value=self._gems("7" * 40)), \
             mock.patch.object(baseline, "repo_info", return_value=None):
            self.assertEqual(baseline.export_image(d), 0)
        entry = baseline.load_yaml(d / "image.baseline.yaml")["baked_repos"]["FlagGems"]
        self.assertEqual(entry["commit"], "4" * 40)
        self.assertIn("base_image", entry["supersedes"])


class TestPackagesField(TempCase):
    def _mk(self, names):
        d = self.image_dir(DF_HEAD, tag="my/img:1")
        for n in names:
            (d / "assets" / "provenance").mkdir(parents=True, exist_ok=True)
            (d / "assets" / "provenance" / n).write_text("x==1\n", encoding="utf-8")
        with mock.patch.object(baseline, "inspect_image", fake_inspect()), \
             mock.patch.object(baseline, "image_python_report",
                               return_value={"python_version": "3.11.15", "versions": {},
                                             "editables": {}, "base_files": {}}):
            self.assertEqual(baseline.export_image(d), 0)
        return baseline.load_yaml(d / "image.baseline.yaml").get("packages")

    def test_ascend_naming_still_selects_the_npu_snapshot(self):
        got = self._mk(["pipfreeze-comm-v3.txt", "pipfreeze-comm-v3-round4.txt",
                        "pipfreeze-comm-v3-round4-npu11.txt"])
        self.assertTrue(got.endswith("pipfreeze-comm-v3-round4-npu11.txt"), got)

    def test_kunlun_naming_is_found(self):
        got = self._mk(["repro-pipfreeze-v2.txt"])
        self.assertTrue(got.endswith("repro-pipfreeze-v2.txt"), got)


class TestUsage(TempCase):
    def test_both_scenarios_is_usage_error(self):
        self.assertEqual(baseline.main(["check", "--image", "a", "b"]), 2)

    def test_no_scenario_is_usage_error(self):
        self.assertEqual(baseline.main(["check"]), 2)

    def test_online_flag_is_image_scope_only(self):
        self.assertEqual(baseline.main(["check", "--online", str(self.tmp)]), 2)


# --------------------------------------------------------------------------
# 模式 6 的第四条边：索引发行包的 wheel sha256（PEP 503 索引页 `#sha256=`）
# --------------------------------------------------------------------------

# 索引页片段，形状照抄真实的 resource.flagos.net/.../simple/flagtree/：
# 同一基础版本号下挂着多个 local version，还有 0.6.1a1+xpu3.6 这种近邻干扰项。
IDX_PAGE = """<html><body>
<a href="../../packages/flagtree/0.6.1/flagtree-0.6.1-cp312-cp312-linux_x86_64.whl#sha256={d}">x</a>
<a href="../../packages/flagtree/0.6.1a1%2Bxpu3.6/flagtree-0.6.1a1+xpu3.6-cp310-cp310-linux_x86_64.whl#sha256={a}">x</a>
<a href="../../packages/flagtree/0.6.1%2Bmthreads3.6/flagtree-0.6.1+mthreads3.6-cp310-cp310-linux_x86_64.whl#sha256={m}">x</a>
<a href="../../packages/flagtree/0.6.1%2Bxpu3.6/flagtree-0.6.1+xpu3.6-cp310-cp310-linux_x86_64.whl#sha256={t}">x</a>
<a href="../../packages/flagtree/0.7.0%2Bxpu3.6/flagtree-0.7.0+xpu3.6-cp310-cp310-linux_x86_64.whl#sha256={n}">x</a>
</body></html>""".format(t="1" * 64, a="2" * 64, m="3" * 64, d="4" * 64, n="5" * 64)

TARGET_WHL = "flagtree-0.6.1+xpu3.6-cp310-cp310-linux_x86_64.whl"


def fake_curl(page, rc=0, stderr=""):
    """只拦 index_wheel_entries 里那次 curl，其余 subprocess.run 原样放过。"""
    real = subprocess.run

    def _inner(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "curl":
            return subprocess.CompletedProcess(cmd, rc, page, stderr)
        return real(cmd, *a, **kw)
    return _inner


class TestIndexWheelSha256(TempCase):
    """坑都是 2026-10-12 在真索引页上实测到的，逐条钉死，见
    kunlun-operator-runtime/v2/lock.yaml known_issues: flagtree-wheel-sha256-in-check-chain。"""

    def _resolve(self, page=IDX_PAGE, **kw):
        # TempCase 默认把 resolve_index_wheel 封死了，这里要测的正是它本体，故直接调
        # 导入时留存的真函数（_REAL_RESOLVE），只把它内部那次 curl 换成假页面。
        with mock.patch.object(baseline.subprocess, "run", fake_curl(page)):
            return _REAL_RESOLVE("https://idx.example/simple", "flagtree", "0.6.1+xpu3.6", **kw)

    def test_entries_parse_pep503_fragment_and_unescape_href(self):
        with mock.patch.object(baseline.subprocess, "run", fake_curl(IDX_PAGE)):
            entries = dict(baseline.index_wheel_entries("https://idx.example/simple", "flagtree"))
        self.assertEqual(entries[TARGET_WHL], "1" * 64)
        self.assertEqual(len(entries), 5)

    def test_curl_failure_hints_at_no_proxy(self):
        with mock.patch.object(baseline.subprocess, "run", fake_curl("", rc=7, stderr="refused")):
            with self.assertRaises(baseline.BaselineError) as cm:
                baseline.index_wheel_entries("https://idx.example/simple", "flagtree")
        self.assertIn("no_proxy", str(cm.exception))

    def test_matches_by_pep440_not_by_substring(self):
        """子串 `0.6.1` 在真索引页上命中 20 条，`0.6.1a1+xpu3.6` 是近邻干扰项。"""
        got = self._resolve()
        self.assertEqual(got["filename"], TARGET_WHL)
        self.assertEqual(got["sha256"], "1" * 64)      # 不是 a1 预发布那条的 "2"*64

    def test_tag_cross_check_passes_on_unique_hit(self):
        got = self._resolve(tag="cp310-cp310-linux_x86_64")
        self.assertEqual(got["filename"], TARGET_WHL)

    def test_tag_mismatch_on_unique_hit_errors(self):
        """唯一命中但 tag 对不上 —— 装进镜像的 wheel 与索引现在的不是同一个，必须报错。"""
        with self.assertRaises(baseline.BaselineError) as cm:
            self._resolve(tag="cp311-cp311-linux_aarch64")
        self.assertIn("不是同一个", str(cm.exception))

    def test_version_absent_from_index_errors(self):
        with mock.patch.object(baseline.subprocess, "run", fake_curl(IDX_PAGE)):
            with self.assertRaises(baseline.BaselineError) as cm:
                _REAL_RESOLVE("https://idx.example/simple", "flagtree", "9.9.9+xpu3.6")
        self.assertIn("没有", str(cm.exception))

    def test_ambiguous_same_version_needs_tag(self):
        """同版本号多 tag：无 tag 时必须报错而不是静默挑一个。"""
        extra = IDX_PAGE.replace("</body>",
                                 '<a href="../flagtree-0.6.1+xpu3.6-cp311-cp311-linux_x86_64.whl'
                                 '#sha256=%s">x</a></body>' % ("6" * 64))
        with mock.patch.object(baseline.subprocess, "run", fake_curl(extra)):
            with self.assertRaises(baseline.BaselineError) as cm:
                _REAL_RESOLVE("https://idx.example/simple", "flagtree", "0.6.1+xpu3.6")
            self.assertIn("无 tag 可消歧", str(cm.exception))
            got = _REAL_RESOLVE("https://idx.example/simple", "flagtree", "0.6.1+xpu3.6",
                                tag="cp311-cp311-linux_x86_64")
        self.assertEqual(got["sha256"], "6" * 64)

    def test_missing_sha256_fragment_errors_not_null(self):
        page = IDX_PAGE.replace("#sha256=" + "1" * 64, "")
        with mock.patch.object(baseline.subprocess, "run", fake_curl(page)):
            with self.assertRaises(baseline.BaselineError) as cm:
                _REAL_RESOLVE("https://idx.example/simple", "flagtree", "0.6.1+xpu3.6")
        self.assertIn("没有 #sha256=", str(cm.exception))


class TestWheelSha256InBaseline(TempCase):
    def _report(self, **kw):
        r = {"python_version": "3.10.18",
             "versions": {"torch": "2.9.0+cu129", "flagtree": "0.6.1+xpu3.6"},
             "editables": {}, "base_files": {}}
        r.update(kw)
        return r

    def _export(self, wheel=None, tags=("cp310-cp310-linux_x86_64",)):
        d = self.image_dir(KL_DF, tag="flagrt/kl:2")
        labels = {"io.flagrt.flagtree.version": "0.6.1+xpu3.6"}
        ctx = [mock.patch.object(baseline, "inspect_image",
                                 fake_inspect(labels, env=["PATH=/x"])),
               mock.patch.object(baseline, "image_python_report", return_value=self._report()),
               mock.patch.object(baseline, "image_wheel_tags",
                                 return_value={"flagtree": list(tags)})]
        if wheel is not None:
            ctx.append(mock.patch.object(baseline, "resolve_index_wheel", return_value=dict(wheel)))
        with ctx[0], ctx[1], ctx[2]:
            if wheel is not None:
                with ctx[3]:
                    rc = baseline.export_image(d)
            else:
                rc = baseline.export_image(d)
        self.assertEqual(rc, 0)
        return d, baseline.load_yaml(d / "image.baseline.yaml")

    def test_export_records_wheel_block(self):
        d, data = self._export(wheel={"filename": TARGET_WHL, "sha256": "1" * 64})
        pkg = data["baked_packages"][0]
        self.assertEqual(pkg["wheel"]["filename"], TARGET_WHL)
        self.assertEqual(pkg["wheel"]["sha256"], "1" * 64)
        self.assertEqual(pkg["wheel"]["source"], "index-page-pep503")
        self.assertIn("fetched_at", pkg["wheel"])
        self.assertIn("wheel sha256", pkg["pin"])

    def test_export_degrades_with_warning_when_offline(self):
        """构建机无外网时 export 必须仍然成功，只是少一个锚（不是报错）。"""
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            d, data = self._export(wheel=None)
        self.assertNotIn("wheel", data["baked_packages"][0])
        self.assertIn("未能取到 wheel sha256", out.getvalue())

    def test_export_refuses_to_guess_on_compressed_tag_set(self):
        """dist-info 有多条 Tag 行时不猜文件名。"""
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            d, data = self._export(wheel={"filename": "x", "sha256": "1" * 64},
                                   tags=("cp310-cp310-linux_x86_64", "cp311-cp311-linux_x86_64"))
        self.assertNotIn("wheel", data["baked_packages"][0])
        self.assertIn("不猜文件名", out.getvalue())

    def _baseline_with_wheel(self, wheel=True):
        d = self.image_dir(KL_DF, name="kl", tag="flagrt/kl:2")
        pkg = {"name": "flagtree", "layer": "L2", "version": "0.6.1+xpu3.6",
               "index_url": "https://idx.example/simple",
               "install": "pip (index wheel, --no-deps)", "pin": "version (...)"}
        if wheel:
            pkg["wheel"] = {"filename": TARGET_WHL, "sha256": "1" * 64,
                            "source": "index-page-pep503"}
        (d / "image.baseline.yaml").write_text(json.dumps({
            "schema": baseline.SCHEMA, "kind": "image",
            "image": {"ref": "flagrt/kl:2", "id": "sha256:" + "a" * 64},
            "baked_repos": {}, "baked_packages": [pkg],
        }), encoding="utf-8")
        return d

    def test_check_offline_reports_info_and_stays_rc0(self):
        """默认不联网：check 在 up.sh --strict 启动路径上，不能因为没网就 FAIL。"""
        d = self._baseline_with_wheel()
        labels = {"io.flagrt.flagtree.version": "0.6.1+xpu3.6"}
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels, env=["PATH=/x"])), \
             mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(baseline.check_image(d), 0)
        self.assertIn("需 check --online", out.getvalue())

    def test_check_online_passes_when_index_unchanged(self):
        d = self._baseline_with_wheel()
        labels = {"io.flagrt.flagtree.version": "0.6.1+xpu3.6"}
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels, env=["PATH=/x"])), \
             mock.patch.object(baseline, "resolve_index_wheel",
                               return_value={"filename": TARGET_WHL, "sha256": "1" * 64}):
            self.assertEqual(baseline.check_image(d, online=True), 0)

    def test_check_online_detects_index_republish(self):
        """同一版本号、不同内容 —— 这正是本条核对要防的事，必须 FAIL。"""
        d = self._baseline_with_wheel()
        labels = {"io.flagrt.flagtree.version": "0.6.1+xpu3.6"}
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels, env=["PATH=/x"])), \
             mock.patch.object(baseline, "resolve_index_wheel",
                               return_value={"filename": TARGET_WHL, "sha256": "9" * 64}), \
             mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(baseline.check_image(d, online=True), 1)
        self.assertIn("重发", out.getvalue())

    def test_check_online_fetch_failure_is_fail_not_silent_pass(self):
        """--online 是显式要求核对，取不到就不能静默放过（否则这个开关没有意义）。"""
        d = self._baseline_with_wheel()
        labels = {"io.flagrt.flagtree.version": "0.6.1+xpu3.6"}
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels, env=["PATH=/x"])):
            self.assertEqual(baseline.check_image(d, online=True), 1)

    def test_check_old_baseline_without_wheel_block_still_passes(self):
        """向后兼容：旧版工具生成的基线没有 wheel 块，不能因此 FAIL。"""
        d = self._baseline_with_wheel(wheel=False)
        labels = {"io.flagrt.flagtree.version": "0.6.1+xpu3.6"}
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels, env=["PATH=/x"])), \
             mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(baseline.check_image(d, online=True), 0)
        self.assertIn("基线未记 wheel sha256", out.getvalue())


class TestEditableInImageGitAnchor(TempCase):
    """`editable-in-image` 资产的第二版本锚：镜像内 .git 的 HEAD（不依赖镜像标签）。"""

    SRC = "/env/FlagGems"

    def _dir(self, commit="7" * 40):
        d = self.image_dir(DF_HEAD, name="kg", tag="flagrt/kg:1")
        (d / "image.baseline.yaml").write_text(json.dumps({
            "schema": baseline.SCHEMA, "kind": "image",
            "image": {"ref": "flagrt/kg:1", "id": "sha256:" + "a" * 64},
            "baked_repos": {"FlagGems": {"layer": "L2", "from": "base_image",
                                         "commit": commit, "source_path": self.SRC,
                                         "install": "editable-in-image"}},
        }), encoding="utf-8")
        return d

    def _check(self, d, heads, labels=None):
        with mock.patch.object(baseline, "inspect_image", fake_inspect(labels or {})), \
             mock.patch.object(baseline, "image_git_heads", return_value=heads), \
             mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            rc = baseline.check_image(d)
        return rc, out.getvalue()

    def test_git_head_matching_baseline_commit_passes(self):
        rc, out = self._check(self._dir(), {self.SRC: "7" * 40})
        self.assertEqual(rc, 0)
        self.assertIn("镜像自证", out)

    def test_git_head_drift_fails(self):
        rc, out = self._check(self._dir(), {self.SRC: "8" * 40})
        self.assertEqual(rc, 1)
        self.assertIn("不一致", out)

    def test_absent_git_is_info_not_fail(self):
        """带 .git 不是对厂商基座的要求，取不到只能是 INFO。"""
        rc, out = self._check(self._dir(), {})
        self.assertEqual(rc, 0)
        self.assertIn("无 .git", out)

    def test_unavailable_commit_with_git_head_reports_candidate(self):
        rc, out = self._check(self._dir(commit="unavailable"), {self.SRC: "7" * 40})
        self.assertEqual(rc, 0)
        self.assertIn("第二版本锚", out)

    def _export_with_editable(self, pip_commit, git_head):
        d = self.image_dir(DF_HEAD, name="ke", tag="flagrt/ke:1")
        report = dict(BASE_REPORT)
        report["base_files"] = {}
        report["editables"] = {"flag_gems": {"source": "file://" + self.SRC,
                                             "commit": pip_commit}}
        report["versions"] = dict(report["versions"], flag_gems="4.2.1rc0")
        heads = {self.SRC: git_head} if git_head else {}
        with mock.patch.object(baseline, "inspect_image", fake_inspect()), \
             mock.patch.object(baseline, "image_python_report", return_value=report), \
             mock.patch.object(baseline, "image_git_heads", return_value=heads), \
             mock.patch.object(baseline, "repo_info", return_value=None), \
             mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(baseline.export_image(d), 0)
        return baseline.load_yaml(d / "image.baseline.yaml")["baked_repos"]["FlagGems"], \
            out.getvalue()

    def test_export_fills_unavailable_commit_from_in_image_git(self):
        """pip 元数据没记 commit 时回退读 .git，把 unavailable 降级成真正的版本锚。"""
        repo, out = self._export_with_editable(pip_commit=None, git_head="7" * 40)
        self.assertEqual(repo["commit"], "7" * 40)
        self.assertEqual(repo["commit_source"], "in-image-git")
        self.assertIn("回退取镜像内", out)

    def test_export_never_overwrites_pip_metadata_commit(self):
        """只填空、不改写：否则换一个取数口径就会让既有基线漂移。"""
        repo, _ = self._export_with_editable(pip_commit="c" * 40, git_head="7" * 40)
        self.assertEqual(repo["commit"], "c" * 40)
        self.assertNotIn("commit_source", repo)

    def test_export_keeps_unavailable_when_no_git(self):
        repo, _ = self._export_with_editable(pip_commit=None, git_head=None)
        self.assertEqual(repo["commit"], "unavailable")
        self.assertNotIn("commit_source", repo)

    def test_probe_failure_does_not_break_check(self):
        d = self._dir()
        with mock.patch.object(baseline, "inspect_image", fake_inspect({})), \
             mock.patch.object(baseline, "image_git_heads",
                               side_effect=baseline.BaselineError("探针炸了", code=1)), \
             mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(baseline.check_image(d), 0)
        self.assertIn("跳过该项", out.getvalue())


_REAL_RESOLVE = baseline.resolve_index_wheel

if __name__ == "__main__":
    unittest.main(verbosity=2)
