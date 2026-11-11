"""Diagnostic RMSNorm family ablations, never change the default allowlist."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import traceback

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from qwen_embedding_baseline import initialize_device, last_token_indices, tolerance
from qwen_rms_shadow import compare
from qwen_scoped_adapter import QwenScopedGems


def family(name):
    return "qk" if name.endswith((".q_norm", ".k_norm")) else "hidden"


def run(args, result):
    device = initialize_device(torch, "npu:0", args.runtime_root, result)
    tol = tolerance("float16")
    result["tolerance"] = dict(atol=tol, rtol=tol)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                                     dtype=torch.float16, attn_implementation="eager").to(device).eval()
    if model.config.model_type != "qwen3":
        raise ValueError("Only Qwen3 is supported")
    targets = [(n, m) for n, m in model.named_modules() if type(m) is Qwen3RMSNorm]
    result["families"] = {f: [n for n, _ in targets if family(n) == f] for f in ("qk", "hidden")}
    with torch.inference_mode():
        for texts in [["如何申请退款"], ["如何申请退款", "退款流程是怎样的", "今天天气怎么样"]]:
            encoded = tokenizer(texts, padding=True, return_tensors="pt").to(device)
            indices = torch.tensor(last_token_indices(encoded["attention_mask"].cpu().tolist()), device=device)

            def forward():
                hidden = model(**encoded, use_cache=False).last_hidden_state
                pooled = hidden[torch.arange(len(texts), device=device), indices]
                vector = torch.nn.functional.normalize(pooled.float(), dim=-1)
                torch.npu.synchronize(device)
                return hidden.cpu(), vector.cpu()

            native_h, native_v = forward()
            for selected in ("qk", "hidden"):
                with ExitStack() as stack:
                    scopes = [stack.enter_context(QwenScopedGems(m, allowed_ops=("rms_norm",)))
                              for n, m in targets if family(n) == selected]
                    actual_h, actual_v = forward()
                completed = sum(sum(v for k, v in s.counts.items() if k[1] == "flaggems" and k[3] == "completed")
                                for s in scopes)
                if completed != len(scopes) or not scopes:
                    raise AssertionError("Expected exactly one target call per selected module")
                assert all("forward" not in m.__dict__ for _, m in targets)
                case = dict(batch=len(texts), selected=selected, target_completed=completed,
                            hidden=compare(actual_h, native_h, tol), embedding=compare(actual_v, native_v, tol))
                case["within_tolerance"] = all(case[k]["finite"] and not case[k]["outside_tolerance"]
                                               for k in ("hidden", "embedding"))
                result["cases"].append(case)
                print(json.dumps(case), flush=True)
            restored_h, restored_v = forward()
            torch.testing.assert_close(restored_h, native_h, atol=0, rtol=0)
            torch.testing.assert_close(restored_v, native_v, atol=0, rtol=0)
    result["status"] = "completed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = dict(status="started", cases=[], dtype="float16",
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="diagnostic role ablation, not production admission or accuracy acceptance")
    with Path(args.out).open("x") as handle:
        try:
            run(args, result)
        except Exception as exc:
            result.update(status="error", error_type=type(exc).__name__, error=str(exc))
            traceback.print_exc()
        finally:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
