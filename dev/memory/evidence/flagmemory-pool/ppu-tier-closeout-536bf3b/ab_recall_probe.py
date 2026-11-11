#!/usr/bin/env python3
# ab_recall_probe.py -- 触发覆写→召回→load DMA（README 剧本形态）
import json, time, urllib.request, threading

URL = "http://127.0.0.1:18351/v1/chat/completions"
PREFIX = ("我们团队研发一套异构计算运行时软件，核心组件包括设备抽象层、"
          "任务调度引擎、显存池管理器与监控诊断模块，目标是让大模型推理"
          "与训练在多种国产加速器上高效运行。")
RECALL = [PREFIX + f"请围绕上述背景，用约五十字介绍其中「{t}」的职责。" for t in
          ("任务调度引擎", "显存池管理器", "设备抽象层", "监控诊断模块")]
FILLER = ("请逐条解释以下术语并各写一个例子：张量并行、流水线并行、数据并行、"
          "专家并行、序列并行、上下文并行、零冗余优化器、梯度检查点、激活值卸载、"
          "优化器状态卸载、混合精度训练、动态损失缩放、梯度累积、学习率预热。") * 4


def fire(prompt, max_tokens=48, timeout=900):
    body = json.dumps({"model": "tiny", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}), timeout=timeout)
    d = json.loads(r.read())
    return round(time.time() - t0, 3), d["choices"][0]["message"]["content"]


print("== stage1: recall prompts (store) ==", flush=True)
for p in RECALL:
    dt, out = fire(p, max_tokens=48)
    print(f"  {dt}s {out[:40]}", flush=True)
time.sleep(3)

print("== stage2: filler burst (overwrite GPU pool) ==", flush=True)
def burst(i):
    try:
        dt, out = fire(f"第{i}组{i}{i}。{FILLER}", max_tokens=48)
        print(f"  B{i}: {dt}s ok", flush=True)
    except Exception as e:
        print(f"  B{i}: FAIL {repr(e)[:100]}", flush=True)
ts = [threading.Thread(target=burst, args=(i,)) for i in range(4)]
for t in ts: t.start()
for t in ts: t.join()
time.sleep(3)

print("== stage3: replay recall prompts (expect load DMA) ==", flush=True)
for p in RECALL:
    dt, out = fire(p, max_tokens=48)
    print(f"  {dt}s {out[:40]}", flush=True)
print("done", flush=True)
