#!/usr/bin/env python3
"""backend_flagcx.cpp allgather 加 debug 打印（output 大小 + datatype）"""
P = "/workspace/FlagCX/plugin/torch/flagcx/src/backend_flagcx.cpp"
s = open(P).read()

old = """  auto device = inputTensor.device();
  auto flagcxDataType = getFlagcxDataType(inputTensor.scalar_type());
  auto stream = getStreamByIndex(0);
  check_device(inputTensor.device(), outputTensorsTmp[0].device());"""
new = """  auto device = inputTensor.device();
  auto flagcxDataType = getFlagcxDataType(inputTensor.scalar_type());
  auto stream = getStreamByIndex(0);
  check_device(inputTensor.device(), outputTensorsTmp[0].device());
  fprintf(stderr, "[AGDBG] input.numel=%lld dtype=%d scalar=%d outTensors=%zu\\n",
          (long long)inputTensor.numel(), (int)flagcxDataType,
          (int)inputTensor.scalar_type(), outputTensorsTmp.size());"""
assert s.count(old) == 1, "allgather head not found"
s = s.replace(old, new, 1)

old2 = """    at::Tensor outputFlattened = newLikeFlat(outputTensorsTmp);"""
new2 = """    at::Tensor outputFlattened = newLikeFlat(outputTensorsTmp);
    fprintf(stderr, "[AGDBG] outputFlattened.numel=%lld\\n",
            (long long)outputFlattened.numel());"""
assert s.count(old2) == 1, "newLikeFlat site not found"
s = s.replace(old2, new2, 1)

open(P, "w").write(s)
print("OK: AGDBG 已加")
