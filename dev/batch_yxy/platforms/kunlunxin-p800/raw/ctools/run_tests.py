#!/usr/bin/env python
"""容器内测试 runner：预注入 tomllib shim（py3.10 环境）后跑 pytest。"""
import sys

import tomli

sys.modules["tomllib"] = tomli
import pytest

args = sys.argv[1:] or ["tests/", "-q"]
sys.exit(pytest.main(args))
