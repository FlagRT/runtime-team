#!/usr/bin/env python3
"""backend_flagcx.cpp initComm：status_==1 时也重设设备（覆盖所有 collective）"""
P = "/workspace/FlagCX/plugin/torch/flagcx/src/backend_flagcx.cpp"
s = open(P).read()

old = """  } else {
    if (dev.is_cuda() || dev.is_privateuseone()) {
      if (deviceId_ != dev.index()) {
        throw std::runtime_error(
            "flagcx communicator was initialized with different device");
      }
    }
  }
}"""
new = """  } else {
    if (dev.is_cuda() || dev.is_privateuseone()) {
      if (deviceId_ != dev.index()) {
        throw std::runtime_error(
            "flagcx communicator was initialized with different device");
      }
      // Kistich: HCCL collectives require the current ACL device to match the
      // communicator's device. torch_fl operations (model load, allocator)
      // may switch the current device away; re-assert it on every collective.
      C10D_FLAGCX_CHECK(devHandle_->setDevice(deviceId_), std::nullopt);
    }
  }
}"""
assert s.count(old) == 1, "initComm else branch not found"
s = s.replace(old, new, 1)
open(P, "w").write(s)
print("OK: initComm else 分支已加 setDevice（覆盖所有 collective）")
