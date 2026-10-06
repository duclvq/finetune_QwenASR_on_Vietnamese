"""Score Malay predictions (eval_wer.py output) under three normalizations.

  basic : lowercase + strip punctuation (what eval_wer.py reports)
  malay : MalayTextNormalizer from revolab-asr-benchmark (spelling variants, particles, some numbers)
  num   : malay + every remaining digit run spelled out in Malay (the references never contain digits)

Usage: python score_malay.py --dir outputs/eval_malay50h [--systems base lora-1.7b-malay50h]
"""
import argparse
import json
import os
import re
import sys

import jiwer

from eval_wer import normalize

sys.path.insert(0, os.environ.get("REVOLAB_DIR", "E:/revolab-asr-benchmark"))
from asr_benchmark.normalizer.malay import MalayTextNormalizer  # noqa: E402

ONES = ["kosong", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "lapan", "sembilan"]
SCALES = [(10**9, "bilion"), (10**6, "juta"), (1000, "ribu"), (100, "ratus")]


def ms_number(n):
    """Cardinal Malay reading: 11 sebelas, 25 dua puluh lima, 100 seratus, 1000 seribu, 2024 dua ribu dua puluh empat."""
    if n < 10:
        return ONES[n]
    if n < 20:
        return {10: "sepuluh", 11: "sebelas"}.get(n, f"{ONES[n - 10]} belas")
    if n < 100:
        return f"{ONES[n // 10]} puluh" + (f" {ONES[n % 10]}" if n % 10 else "")
    for value, word in SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            out = ("se" + word) if head == 1 and value <= 1000 else f"{ms_number(head)} {word}"
            return out + (f" {ms_number(rest)}" if rest else "")


def spell_numbers(text):
    return re.sub(r"\d+", lambda m: ms_number(int(m.group())) if len(m.group()) <= 12 else m.group(), text)


_MALAY = MalayTextNormalizer()
NORMS = {"basic": normalize,
         "malay": _MALAY,
         "num": lambda t: _MALAY(spell_numbers(_MALAY(t)))}


def score(rows, norm):
    pairs = [(norm(x["ref"]), norm(x["hyp"]) or "x") for x in rows if norm(x["ref"])]
    r, h = zip(*pairs)
    o = jiwer.process_words(list(r), list(h))
    n = sum(len(x.split()) for x in r)
    return {"wer": round(100 * o.wer, 2), "cer": round(100 * jiwer.cer(list(r), list(h)), 2),
            "sub": round(100 * o.substitutions / n, 2), "del": round(100 * o.deletions / n, 2),
            "ins": round(100 * o.insertions / n, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/eval_malay50h")
    ap.add_argument("--systems", nargs="*", default=None)
    args = ap.parse_args()
    systems = args.systems or sorted(f[:-len("_predictions.jsonl")] for f in os.listdir(args.dir)
                                     if f.endswith("_predictions.jsonl"))
    out = {}
    for name in systems:
        rows = [json.loads(l) for l in open(os.path.join(args.dir, f"{name}_predictions.jsonl"), encoding="utf-8")]
        out[name] = {"clips": len(rows), **{k: score(rows, f) for k, f in NORMS.items()}}
        print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(os.path.join(args.dir, "malay_scores.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
