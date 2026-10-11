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

    def test_pip_install_from_index_errors(self):
        d = self.image_dir(DF_HEAD + "RUN ${PYTHON_BIN} -m pip install flagtree==0.4.0\n")
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


class TestUsage(TempCase):
    def test_both_scenarios_is_usage_error(self):
        self.assertEqual(baseline.main(["check", "--image", "a", "b"]), 2)

    def test_no_scenario_is_usage_error(self):
        self.assertEqual(baseline.main(["check"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
