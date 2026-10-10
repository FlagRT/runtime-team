# L1 修复前后对照（平头哥 PPU · 78 项职责响应审计）
# 修复点：prototype/runtime/backends/base.py::_register_owned_stream（id 键表改用**强持有**）

## 修复前（弱引用持有器）
  pre_run1.log                 OK 71 / FAIL 1 / SKIP 6
  pre_run2.log                 OK 71 / FAIL 1 / SKIP 6
  pre_run3.log                 OK 71 / FAIL 1 / SKIP 6

## 修复后（强持有）
  post5_run1.log               OK 72 / FAIL 0 / SKIP 6 | [OK] L1
  post5_run2.log               OK 72 / FAIL 0 / SKIP 6 | [OK] L1
  post5_run3.log               OK 72 / FAIL 0 / SKIP 6 | [OK] L1
  post5_run4.log               OK 72 / FAIL 0 / SKIP 6 | [OK] L1
  post5_run5.log               OK 72 / FAIL 0 / SKIP 6 | [OK] L1
  post_run1.log                OK 72 / FAIL 0 / SKIP 6 | [OK] L1
  post_run2.log                OK 72 / FAIL 0 / SKIP 6 | [OK] L1
  post_run3.log                OK 72 / FAIL 0 / SKIP 6 | [OK] L1

## 四家离线回归（含 id 复用非空转判据；与修复前逐字相同）
  ascend 90/0/1 · kunlun 108/0/1 · cambricon 97/0/0 · ppu 112/0/1 ；对称性 7/0
