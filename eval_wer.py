"""Evaluate Qwen3-ASR base vs LoRA vs QLoRA adapters on the prepared ViMD test split.

Writes OUT/<name>_predictions.jsonl (every clip) and OUT/summary.json (WER/CER overall + per region).
QLoRA adapters are merged into the bf16 base for evaluation.

Usage: python eval_wer.py [--max_samples N] [--out outputs/eval_test]
"""
import argparse
import json
import os
import re
import string
import time

import jiwer
import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel

from train_qwen_asr import DATA, LANG, PreparedAudio

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}“”‘’…–—]")


def normalize(text):
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", text.lower())).strip()


def transcribe(model_path, adapter, ds, bs, lang=LANG):
    w = Qwen3ASRModel.from_pretrained(model_path, dtype=torch.bfloat16, device_map={"": 0},
                                      max_inference_batch_size=bs, max_new_tokens=400)
    if adapter:
        w.model = PeftModel.from_pretrained(w.model, adapter).merge_and_unload()
    w.model.eval()
    hyps, t0 = [], time.time()
    for s in range(0, len(ds), bs):
        audios = [(ds[i]["audio"], 16000) for i in range(s, min(s + bs, len(ds)))]
        hyps += [r.text.strip() for r in w.transcribe(audio=audios, language=lang)]
        print(f"{adapter or 'base'}: {len(hyps)}/{len(ds)} ({time.time()-t0:.0f}s)", flush=True)
    del w
    torch.cuda.empty_cache()
    return hyps, time.time() - t0


def score(refs, hyps):
    pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps) if normalize(r)]
    r, h = [p[0] for p in pairs], [p[1] for p in pairs]
    return {"wer": round(100 * jiwer.wer(r, h), 2), "cer": round(100 * jiwer.cer(r, h), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Qwen3-ASR-1.7B")
    ap.add_argument("--runs", nargs="*", default=["lora-1.7b", "qlora-1.7b"])
    ap.add_argument("--full", nargs="*", default=[], help="full-FT runs; evaluates outputs/<run>/final")
    ap.add_argument("--skip_base", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="only evaluate these system names")
    ap.add_argument("--out", default="outputs/eval_test")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--data", default=DATA, help="prepared data dir with test/")
    ap.add_argument("--lang", default=LANG)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ds = PreparedAudio(os.path.join(args.data, args.split), max_samples=args.max_samples)

    systems = {} if args.skip_base else {"base": (args.model_path, None)}
    systems.update({r: (args.model_path, f"outputs/{r}/adapter") for r in args.runs if os.path.isdir(f"outputs/{r}/adapter")})
    systems.update({r: (f"outputs/{r}/final", None) for r in args.full if os.path.isdir(f"outputs/{r}/final")})
    if args.only:
        systems = {k: v for k, v in systems.items() if k in args.only}
    preds, summary = {}, {}
    for name, (mpath, adapter) in systems.items():
        preds[name], secs = transcribe(mpath, adapter, ds, args.batch_size, args.lang)
        with open(os.path.join(args.out, f"{name}_predictions.jsonl"), "w", encoding="utf-8") as f:
            for m, h in zip(ds.meta, preds[name]):
                f.write(json.dumps({"filename": m.get("filename", m.get("source")), "region": m.get("region"), "ref": m["text"], "hyp": h,
                                    **score([m["text"]], [h])}, ensure_ascii=False) + "\n")
        summary[name] = {"infer_seconds": round(secs, 1)}
        for region in ["All", "North", "Central", "South"]:
            idx = [i for i, m in enumerate(ds.meta) if region == "All" or m.get("region") == region]
            if idx:
                summary[name][region] = {"clips": len(idx), **score([ds.meta[i]["text"] for i in idx],
                                                                    [preds[name][i] for i in idx])}
        print(name, json.dumps(summary[name]), flush=True)
    path = os.path.join(args.out, "summary.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    json.dump({**old, **summary}, open(path, "w"), indent=2)  # merge, so earlier systems are kept


if __name__ == "__main__":
    main()
