"""Evaluate Qwen3-ASR base vs LoRA adapters on Revolab/ASR-Benchmark-Public (820 Malay clips, 12 domains).

Scoring follows the benchmark: references and hypotheses go through MalayTextNormalizer + Malay number spelling
(score_malay.NORMS["num"]); each row is scored against both `text` and `normalized_text` and the lower-WER reference
wins. Corpus WER/CER overall and per category -> OUT/summary.json; every clip -> OUT/<name>_predictions.jsonl.

Usage: USE_TF=0 python eval_revolab.py --model_path <Qwen3-ASR-1.7B dir> --runs lora-1.7b-malay50h [--out outputs/eval_revolab]
"""
import argparse
import glob
import io
import json
import os

import jiwer
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from eval_wer import transcribe
from score_malay import NORMS

NORM = NORMS["num"]


class RevolabAudio:
    def __init__(self, parquet):
        t = pq.read_table(parquet)
        self.meta = t.drop(["audio"]).to_pylist()
        self.audio = t.column("audio").to_pylist()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        a, sr = sf.read(io.BytesIO(self.audio[i]["bytes"]), dtype="float32", always_2d=True)
        a = a.mean(axis=1)
        if sr != 16000:
            import librosa
            a = librosa.resample(a, orig_sr=sr, target_sr=16000)
        return {"audio": a, "text": self.meta[i]["text"]}


def best_ref(m, hyp):
    """(normalized ref, normalized hyp) using whichever reference gives fewer errors (ties -> dataset normalized)."""
    h = NORM(hyp) or "x"
    cands = [NORM(m["normalized_text"]), NORM(m["text"])]
    cands = [c for c in cands if c]
    errs = [jiwer.process_words(c, h) for c in cands]
    k = int(np.argmin([o.substitutions + o.deletions + o.insertions for o in errs]))
    return cands[k], h


def corpus(pairs):
    r, h = [p[0] for p in pairs], [p[1] for p in pairs]
    return {"clips": len(pairs), "wer": round(100 * jiwer.wer(r, h), 2), "cer": round(100 * jiwer.cer(r, h), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--skip_base", action="store_true")
    ap.add_argument("--lang", default="Malay")
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--out", default="outputs/eval_revolab")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()
    parquet = args.parquet or glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--Revolab--ASR-Benchmark-Public/snapshots/*/data/*.parquet"))[0]
    ds = RevolabAudio(parquet)
    os.makedirs(args.out, exist_ok=True)

    systems = {} if args.skip_base else {"base": None}
    systems.update({r: f"outputs/{r}/adapter" for r in args.runs if os.path.isdir(f"outputs/{r}/adapter")})
    summary = {}
    for name, adapter in systems.items():
        pred_path = os.path.join(args.out, f"{name}_predictions.jsonl")
        if os.path.exists(pred_path):  # re-score without re-transcribing
            hyps = [json.loads(l)["hyp"] for l in open(pred_path, encoding="utf-8")]
            secs = None
        else:
            hyps, secs = transcribe(args.model_path, adapter, ds, args.batch_size, args.lang)
        pairs = [best_ref(m, h) for m, h in zip(ds.meta, hyps)]
        with open(pred_path, "w", encoding="utf-8") as f:
            for m, h, (rn, hn) in zip(ds.meta, hyps, pairs):
                f.write(json.dumps({"id": m["id"], "category": m["category"], "ref": m["text"],
                                    "ref_norm": m["normalized_text"], "hyp": h, "ref_scored": rn, "hyp_scored": hn,
                                    "wer": round(100 * jiwer.wer(rn, hn), 2)}, ensure_ascii=False) + "\n")
        summary[name] = {"infer_seconds": secs and round(secs, 1), "All": corpus(pairs)}
        for cat in sorted({m["category"] for m in ds.meta}):
            summary[name][cat] = corpus([p for m, p in zip(ds.meta, pairs) if m["category"] == cat])
        print(name, json.dumps(summary[name]["All"]), flush=True)
    path = os.path.join(args.out, "summary.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    json.dump({**old, **summary}, open(path, "w"), indent=2)


if __name__ == "__main__":
    main()
