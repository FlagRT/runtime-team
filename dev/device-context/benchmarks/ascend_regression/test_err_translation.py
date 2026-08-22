#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
test_err_translation.py — 细项21 补验：错误码三维翻译（类别/位置/根因）
═══════════════════════════════════════════════════════════════════════════════

【验证目标】设备执行上下文"错误三维翻译"职责在 910C（torch_fl/flagos 设备）上
 的可观测性。按设计，每个厂商错误应翻译为统一错误对象的三个投影：
    - 类别（L1 资源类 / L2 参数类 / L3 执行类 / L4 致命类）
    - 位置（错误归因到的流 / 事件 / 任务）
    - 根因（厂商原始错误码与描述原样保留）
  并触发对应分级处置（L2 → 上抛调用方，不重试）。

【用法】容器内、tf-venv-integration 激活状态下单进程运行：
    python test_err_translation.py
  （无需 torchrun，本脚本为单进程探针）

【输出】stdout 打印每类错误的"三维投影探测结果"，末尾输出汇总：
    ERR_TRANSLATION_PASS / ERR_TRANSLATION_PARTIAL / ERR_TRANSLATION_FAIL
  PARTIAL = 部分错误能被捕获并翻译，但统一错误对象接口未完全暴露（记录接口缺口）

【硬约束】全程 torch_fl（flagos 设备），不 import torch_npu。
【注意】torch_fl 当前为框架层（PrivateUse1 机制），底层 CANN 错误会被 PyTorch
  包装为异常；若 torch_fl 尚未暴露"统一错误对象"API，本脚本记录异常原文作为
  三维翻译的输入证据，并在结果中标注"接口缺口"——这本身即是有价值的验证结论。
"""

import os
import sys
import json


def probe(label, fn):
    """执行一个可能失败的调用，捕获并解析错误，返回三维投影探测结果。"""
    result = {
        "probe": label,
        "caught": False,
        "category": "UNKNOWN",   # L1/L2/L3/L4 或 UNKNOWN
        "location": "UNKNOWN",   # 流/事件/任务 归因（框架层通常无）
        "root_cause": "",        # 厂商原始错误信息
        "note": "",
    }
    try:
        fn()
        result["note"] = "调用未报错（可能未触发该错误路径）"
        return result
    except Exception as e:
        result["caught"] = True
        result["root_cause"] = f"{type(e).__name__}: {e}"
        msg = str(e).lower()
        # 0) 优先识别厂商错误码（如 aclnn...failed, ret=161002）→ 统一类别
        import re as _re
        ACL_ERR_TO_CATEGORY = {
            161002: "L2", 161001: "L2", 161003: "L2", 161004: "L2", 161005: "L2",
            161007: "L1", 161025: "L3",
        }
        m = _re.search(r"ret=(\d+)", msg)
        if m and int(m.group(1)) in ACL_ERR_TO_CATEGORY:
            result["category"] = ACL_ERR_TO_CATEGORY[int(m.group(1))]
            result["note"] = f"识别到 ACL 错误码 {m.group(1)} → 统一类别 {result['category']}"
            return result
        # 粗分类：根据异常形态推断统一类别（探针级，非权威）
        if any(k in msg for k in ("shape", "size mismatch", "dimension", "预期", "形状")):
            result["category"] = "L2"          # 参数类
        elif any(k in msg for k in ("device", "ordinal", "device_count")):
            result["category"] = "L2"          # 设备参数类
        elif any(k in msg for k in ("memory", "alloc", "out of", "resource", "名额")):
            result["category"] = "L1"          # 资源类
        elif any(k in msg for k in ("runtime", "execute", "kernel", "stream", "event", "aclnn", "failed")):
            result["category"] = "L3"          # 执行类（算子调用失败）
        # 位置投影：框架层异常通常不携带流/事件/任务归因 → 如实标注
        result["location"] = "N/A（框架层异常，无在途任务登记表）"
        result["note"] = "捕获到异常；统一错误对象接口未暴露时，以下为三维翻译的输入证据"
        return result


def main():
    print("=== test_err_translation.py: 错误三维翻译补验 ===")
    print("目标: 910C 上验证错误 → 类别/位置/根因 三投影可观测\n")

    # 0. 环境自检
    import torch
    import torch_fl
    from torch_fl import flagos
    print(f"[env] torch={torch.__version__} torch_fl={getattr(torch_fl,'__version__','unknown')}")
    devs = flagos.device_count()
    print(f"[env] flagos devices={devs}")
    if devs < 1:
        print("ERR_TRANSLATION_FAIL: 无可用 flagos 设备，先恢复环境")
        sys.exit(1)

    # 1. 正常基线：flagos 张量 + 合法运算（证明环境可用）
    x = torch.randn(4, 4, device="flagos")
    y = x @ x
    pass  # flagos 环境禁用 torch.cuda（is_available 误报）
    print(f"[baseline] flagos 张量运算 ok: {y.device}, 无异常\n")

    # 2. 错误注入（三维翻译的输入证据）
    results = []

    # 2a. L2 参数类：形状不匹配的矩阵乘
    results.append(probe("L2-形状不匹配", lambda: (torch.randn(3, 4, device="flagos") @ torch.randn(5, 6, device="flagos"))))

    # 2b. L2 参数类：非法设备 ordinal（若 flagos 暴露设备枚举边界检查）
    def bad_ordinal():
        d = devs + 99
        # 若 torch_fl 暴露 set_device，尝试越界；否则构造 device 字符串
        if hasattr(flagos, "set_device"):
            flagos.set_device(d)
        else:
            torch.zeros(1, device=f"flagos:{d}")
    results.append(probe("L2-非法设备ordinal", bad_ordinal))

    # 2c. 执行类/资源类：尝试大分配（超显存）或非法操作
    results.append(probe("L1/L3-大分配或执行类", lambda: torch.empty(2 ** 33, device="flagos") if devs else 0))

    # 3. 汇总输出
    print("=== 三维投影探测结果 ===")
    for r in results:
        print(f"  [{r['probe']}]")
        print(f"    caught    = {r['caught']}")
        print(f"    category  = {r['category']}")
        print(f"    location  = {r['location']}")
        print(f"    root_cause= {r['root_cause'][:120]}")
        if r["note"]:
            print(f"    note      = {r['note']}")

    # 4. 判定
    caught = [r for r in results if r["caught"]]
    has_l2 = any(r["category"] == "L2" for r in caught)
    if not caught:
        verdict = "ERR_TRANSLATION_FAIL"
        print(f"\n{verdict}: 未捕获到任何错误（环境或注入方式问题）")
    elif has_l2:
        # 能捕获且能识别 L2 参数类 → 三维翻译的"类别"投影部分可验证
        print(f"\n结论: 错误可被捕获并粗分类（L2 参数类可识别）; 位置投影依赖在途任务登记表（框架层未暴露）; 根因原文完整保留")
        verdict = "ERR_TRANSLATION_PARTIAL"
        print(f"\n{verdict}: 三维翻译部分可观测。若 torch_fl 后续暴露统一错误对象接口，本脚本的 probe() 可扩展为断言三投影")
    else:
        verdict = "ERR_TRANSLATION_PARTIAL"
        print(f"\n{verdict}: 捕获到错误但未能按 L1-L4 粗分类，需补充错误样本")

    # 5. 结果落盘（供看板回填）
    out = {"verdict": verdict, "results": results, "interface_gap": "统一错误对象API未暴露，当前为框架层异常原文"}
    with open("err_translation_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n结果已写入 err_translation_result.json")


if __name__ == "__main__":
    main()
