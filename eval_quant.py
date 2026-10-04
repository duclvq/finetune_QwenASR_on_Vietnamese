"""Weight-only quantization sweep for the full fine-tuned Qwen3-ASR (LLM linears quantized, audio encoder kept bf16).

q2/q3/q4/q8 use HQQ (real packed weights, calibration-free). HQQ has no 6-bit packing, so q6 (and any
unsupported width) falls back to group-wise round-to-nearest fake quantization (same accuracy characteristics,
but weights stay bf16 so memory is NOT reduced). Output: OUT/<name>_predictions.jsonl + OUT/quant_summary.json.

Usage: python eval_quant.py --configs q8 q6 q4 q3 q2 [--max_samples 400]
"""
import argparse
import json
import os
import re
import time

import jiwer
import torch
from qwen_asr import Qwen3ASRModel
from transformers import HqqConfig

from eval_wer import normalize
from train_qwen_asr import DATA, PreparedAudio

MODEL = "outputs/full-1.7b/final"
LLM_RE = re.compile(r"thinker\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$")


def fake_quant_(model, nbits, group):
    """In-place RTN asymmetric group-wise quantize->dequantize of the LLM linears."""
    qmax = 2 ** nbits - 1
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and LLM_RE.search(name):
            w = mod.weight.data.float()
            o, i = w.shape
            g = w.reshape(o, i // group, group)
            lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
            scale = ((hi - lo) / qmax).clamp_min(1e-8)
            zero = (-lo / scale).round()
            q = (g / scale + zero).round().clamp(0, qmax)
            mod.weight.data = ((q - zero) * scale).reshape(o, i).to(mod.weight.dtype)


def load(cfg):
    bits, group = cfg["bits"], cfg["group"]
    kw = dict(dtype=torch.bfloat16, device_map={"": 0}, max_inference_batch_size=8, max_new_tokens=300)
    if bits == 16:
        return Qwen3ASRModel.from_pretrained(MODEL, **kw), "bf16"
    if bits in (1, 2, 3, 4, 8):
        q = HqqConfig(nbits=bits, group_size=group, skip_modules=["lm_head", "audio_tower"])
        w = Qwen3ASRModel.from_pretrained(MODEL, quantization_config=q, **kw)
        # HQQ's packed params make model.dtype report float32, so the wrapper feeds fp32 features to a bf16 encoder
        type(w.model).dtype = property(lambda self: torch.bfloat16)
        from hqq.core.quantize import HQQLinear  # its scales/zeros are fp32: run the matmul in the activation dtype
        HQQLinear.matmul = lambda self, x, transpose=True: torch.matmul(
            x, self.dequantize().to(x.dtype).t() if transpose else self.dequantize().to(x.dtype))
        for m_ in w.model.modules():
            if type(m_).__name__ == "HQQLinear" and getattr(m_, "bias", None) is not None:
                m_.bias.data = m_.bias.data.to(torch.bfloat16)
        for mod in w.model.modules():  # modules HQQ skipped come back fp32; HQQLinear manages its own dtypes
            if type(mod).__name__ != "HQQLinear":
                for prm in mod.parameters(recurse=False):
                    if prm.dtype == torch.float32:
                        prm.data = prm.data.to(torch.bfloat16)
        return w, "hqq"
    w = Qwen3ASRModel.from_pretrained(MODEL, **kw)
    fake_quant_(w.model, bits, group)
    return w, "fake-rtn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["q8", "q6", "q4", "q3", "q2"])
    ap.add_argument("--max_samples", type=int, default=400)
    ap.add_argument("--out", default="outputs/eval_quant")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ds = PreparedAudio(os.path.join(DATA, "test"), max_samples=args.max_samples, seed=7)
    refs = [m["text"] for m in ds.meta]
    path = os.path.join(args.out, "quant_summary.json")
    summary = json.load(open(path)) if os.path.exists(path) else {}

    for name in args.configs:
        m = re.fullmatch(r"q(\d+)(?:g(\d+))?", name) or re.fullmatch(r"(bf16)", name)
        cfg = {"bits": 16, "group": 0} if name == "bf16" else {"bits": int(m.group(1)), "group": int(m.group(2) or 64)}
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        w, method = load(cfg)
        mem_load = torch.cuda.memory_allocated() / 2**30
        w.model.eval()
        hyps, t1 = [], time.time()
        for s in range(0, len(ds), args.batch_size):
            audios = [(ds[i]["audio"], 16000) for i in range(s, min(s + args.batch_size, len(ds)))]
            hyps += [r.text.strip() for r in w.transcribe(audio=audios, language="Vietnamese")]
        infer = time.time() - t1
        pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps) if normalize(r)]
        res = {"method": method, "bits": cfg["bits"], "group": cfg["group"], "clips": len(pairs),
               "wer": round(100 * jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs]), 2),
               "cer": round(100 * jiwer.cer([p[0] for p in pairs], [p[1] for p in pairs]), 2),
               "gpu_gb_after_load": round(mem_load, 2), "gpu_gb_peak": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "load_s": round(t1 - t0, 1), "infer_s": round(infer, 1),
               "empty_hyps": sum(not h for h in hyps)}
        summary[name] = res
        print(name, json.dumps(res), flush=True)
        with open(os.path.join(args.out, f"{name}_predictions.jsonl"), "w", encoding="utf-8") as f:
            for mm, h in zip(ds.meta, hyps):
                f.write(json.dumps({"filename": mm["filename"], "ref": mm["text"], "hyp": h}, ensure_ascii=False) + "\n")
        json.dump(summary, open(path, "w"), indent=2)
        del w
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
