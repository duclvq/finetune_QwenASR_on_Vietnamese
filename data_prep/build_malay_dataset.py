"""Build Malay training data from a local copy of mesolitica/malaya-speech-malay-stt.

The source has a single train split (~1.6M mp3 clips, ~2600 hours, lowercase
text without punctuation). Rows are shuffled once with a fixed seed: the first
--n_test go to test, the next --n_valid to valid, and train takes a random
subset sized to roughly --max_hours (0 = everything). Output format matches
build_vimd_dataset.py (OUT/<split>/audio.bin int16 16kHz + meta.jsonl), so
train_qwen_asr.py reads it directly.

Each shard is decoded by a worker into OUT/_tmp/<split>/<shard>.{bin,jsonl},
so an interrupted run resumes from the shards already done.

Usage: python build_malay_dataset.py --src D:/hf_datasets/malaya-speech-malay-stt/data
           --out D:/data/malay_prepared [--max_hours 500] [--workers 12]
"""
import argparse
import glob
import io
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SR = 16000
SPLITS = ["train", "valid", "test"]


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def decode(mp3_bytes):
    audio, sr = sf.read(io.BytesIO(mp3_bytes), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    return audio


def estimate_mean_sec(path, n=300):
    rows = pq.ParquetFile(path).read_row_group(0, columns=["filename"]).slice(0, n).column("filename").to_pylist()
    return float(np.mean([len(decode(r["bytes"])) / SR for r in rows]))


def process_shard(path, picks, tmp_root, min_sec, max_sec):
    """picks: {split: sorted row indices}. Writes one tmp bin/jsonl pair per split."""
    name = os.path.basename(path).rsplit(".", 1)[0]
    todo = {s: idx for s, idx in picks.items()
            if len(idx) and not os.path.exists(os.path.join(tmp_root, s, name + ".jsonl"))}
    if not todo:
        return name, "cached"
    table = pq.read_table(path)
    stats = {}
    for split, idx in todo.items():
        sub = table.take(idx).to_pylist()
        base = os.path.join(tmp_root, split, name)
        offset, kept = 0, 0
        with open(base + ".bin", "wb") as fa, open(base + ".jsonl.part", "w", encoding="utf-8") as fm:
            for row_i, rec in zip(idx, sub):
                text = clean_text(rec["Y"])
                if not text:
                    continue
                try:
                    audio = decode(rec["filename"]["bytes"])
                except Exception:
                    continue
                if not (min_sec * SR <= len(audio) <= max_sec * SR):
                    continue
                pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                fa.write(pcm.tobytes())
                meta = {"text": text, "source": f"{name}:{row_i}", "offset": offset, "length": len(pcm)}
                fm.write(json.dumps(meta, ensure_ascii=False) + "\n")
                offset += len(pcm)
                kept += 1
        os.replace(base + ".jsonl.part", base + ".jsonl")
        stats[split] = f"{kept}/{len(idx)}"
    return name, " ".join(f"{s} {v}" for s, v in stats.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/hf_datasets/malaya-speech-malay-stt/data")
    ap.add_argument("--out", default="D:/data/malay_prepared")
    ap.add_argument("--max_hours", type=float, default=500.0, help="approx train hours; 0 = all")
    ap.add_argument("--n_valid", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--min_sec", type=float, default=0.5)
    ap.add_argument("--max_sec", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shards", type=int, default=0, help="use only the first N shards (smoke test)")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.src, "*.parquet")))[:args.shards or None]
    counts = [pq.ParquetFile(p).metadata.num_rows for p in shards]
    total = sum(counts)
    starts = np.cumsum([0] + counts[:-1])

    # Global shuffle -> split assignment. rank[i] = position of row i in the shuffled order.
    order = np.random.default_rng(args.seed).permutation(total)
    rank = np.empty(total, dtype=np.int64)
    rank[order] = np.arange(total)
    n_train = total - args.n_test - args.n_valid
    if args.max_hours > 0:
        mean_sec = estimate_mean_sec(shards[0])
        n_train = min(n_train, int(args.max_hours * 3600 / mean_sec))
        print(f"mean clip {mean_sec:.2f}s -> {n_train} train clips for ~{args.max_hours:.0f}h", flush=True)
    bounds = {"test": (0, args.n_test),
              "valid": (args.n_test, args.n_test + args.n_valid),
              "train": (args.n_test + args.n_valid, args.n_test + args.n_valid + n_train)}

    tmp_root = os.path.join(args.out, "_tmp")
    for s in SPLITS:
        os.makedirs(os.path.join(tmp_root, s), exist_ok=True)
    picks = []
    for p, st, n in zip(shards, starts, counts):
        r = rank[st:st + n]
        picks.append({s: np.flatnonzero((r >= lo) & (r < hi)) for s, (lo, hi) in bounds.items()})

    with ProcessPoolExecutor(args.workers) as ex:
        jobs = ex.map(process_shard, shards, picks, [tmp_root] * len(shards),
                      [args.min_sec] * len(shards), [args.max_sec] * len(shards))
        for i, (name, status) in enumerate(jobs, 1):
            print(f"[{i}/{len(shards)}] {name}: {status}", flush=True)

    # Concatenate per-shard pieces in shard order, rebasing offsets.
    for split in SPLITS:
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir, exist_ok=True)
        offset, kept = 0, 0
        with open(os.path.join(out_dir, "audio.bin.part"), "wb") as fa, \
                open(os.path.join(out_dir, "meta.jsonl.part"), "w", encoding="utf-8") as fm:
            for p in shards:
                base = os.path.join(tmp_root, split, os.path.basename(p).rsplit(".", 1)[0])
                if not os.path.exists(base + ".jsonl"):
                    continue
                with open(base + ".bin", "rb") as fb:
                    shutil.copyfileobj(fb, fa, 16 << 20)
                with open(base + ".jsonl", encoding="utf-8") as fj:
                    for line in fj:
                        meta = json.loads(line)
                        meta["offset"] += offset
                        fm.write(json.dumps(meta, ensure_ascii=False) + "\n")
                        kept += 1
                offset += os.path.getsize(base + ".bin") // 2
        os.replace(os.path.join(out_dir, "audio.bin.part"), os.path.join(out_dir, "audio.bin"))
        os.replace(os.path.join(out_dir, "meta.jsonl.part"), os.path.join(out_dir, "meta.jsonl"))
        print(f"{split}: {kept} clips, {offset / SR / 3600:.1f} hours", flush=True)
    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
