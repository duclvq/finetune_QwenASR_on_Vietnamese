"""Re-score test predictions with Vietnamese number normalization (vi_numbers.py).

Per system: raw WER, WER with every number rewritten to its default spoken reading, and WER where each number takes the
reading closest to the reference ('best', optimistic). Usage: python rescore_numbers.py [--out results/number_norm_summary.json]
"""
import argparse
import json
import re

import jiwer

from eval_wer import normalize
from vi_numbers import NUM_RE, normalize_numbers

SYSTEMS = {"base": "base", "lora": "lora-1.7b", "qlora": "qlora-1.7b", "full": "full-1.7b"}


def n_err(ref, hyp):
    o = jiwer.process_words(normalize(ref), normalize(hyp) or "x")
    return o.substitutions + o.deletions + o.insertions


def wer(rows, key):
    r = [normalize(x["ref"]) for x in rows]
    h = [normalize(x[key]) or "x" for x in rows]
    return round(100 * jiwer.wer(r, h), 2), round(100 * jiwer.cer(r, h), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/eval_test")
    ap.add_argument("--out", default="results/number_norm_summary.json")
    args = ap.parse_args()
    out = {}
    for name, fname in SYSTEMS.items():
        rows = [json.loads(l) for l in open(f"{args.dir}/{fname}_predictions.jsonl", encoding="utf-8")]
        n_nums = 0
        for x in rows:
            x["hyp_default"] = normalize_numbers(x["hyp"])
            x["hyp_best"] = normalize_numbers(x["hyp"], ref=x["ref"], errors=n_err)
            n_nums += len(NUM_RE.findall(x["hyp"]))
        digit_rows = [x for x in rows if NUM_RE.search(x["hyp"])]
        res = {"clips_with_numbers": len(digit_rows), "numbers": n_nums}
        for key in ("hyp", "hyp_default", "hyp_best"):
            w, c = wer(rows, key)
            res[key] = {"wer": w, "cer": c}
            for region in ("North", "Central"):
                res[key][region] = wer([x for x in rows if x["region"] == region], key)[0]
            if digit_rows:
                res[key]["wer_on_number_clips"] = wer(digit_rows, key)[0]
            res[key]["wer_on_other_clips"] = wer([x for x in rows if x not in digit_rows], key)[0]
        out[name] = res
        print(name, json.dumps(res, ensure_ascii=False), flush=True)
        with open(f"{args.dir}/{fname}_predictions_numnorm.jsonl", "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps({k: x[k] for k in ("filename", "region", "ref", "hyp", "hyp_default", "hyp_best")}, ensure_ascii=False) + "\n")
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
