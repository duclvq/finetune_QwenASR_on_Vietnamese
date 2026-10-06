"""Add code-switched (Manglish) and English speech to the cleaned Malay set, against English forgetting.

A LoRA trained on Malay-only labels started forcing English speech into Malay words (Revolab telephony and
short-inputs regressed). This builds OUT = <base clean train> + Synth-Manglish + cs_pilot (real podcast CS clips)
+ an English slice of FLEURS en_us (tagged "lang": "English", so the trainer uses `language English`).
Extra transcripts are lowercased and stripped of punctuation to match the Malay labels.
A held-out code-switch test split (OUT/cs_test) is carved from Synth-Manglish + cs_pilot.
valid/test are copied from the base set unchanged.

Usage: python build_malay_mix.py --base D:/data/malay_50h_clean --out D:/data/malay_50h_mix
"""
import argparse
import glob
import io
import json
import os
import re
import shutil

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SR = 16000
HF = os.path.expanduser("~/.cache/huggingface/hub")


def clean_text(t):
    t = re.sub(r"[^\w\s'%-]", " ", t.lower().replace("’", "'"))
    return re.sub(r"\s+", " ", t.replace("_", " ")).strip()


def to_pcm(raw_bytes):
    a, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != SR:
        a = librosa.resample(a, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    return (np.clip(a, -1, 1) * 32767).astype(np.int16)


def parquet_clips(pattern, text_col, lang=None, source=""):
    for f in sorted(glob.glob(pattern)):
        t = pq.read_table(f, columns=["audio", text_col])
        for k, (a, txt) in enumerate(zip(t.column("audio").to_pylist(), t.column(text_col).to_pylist())):
            text = clean_text(txt or "")
            if text:
                yield to_pcm(a["bytes"]), {"text": text, "source": f"{source}:{os.path.basename(f)}:{k}",
                                            **({"lang": lang} if lang else {})}


def memmap_clips(split_dir, source):
    meta = [json.loads(l) for l in open(os.path.join(split_dir, "meta.jsonl"), encoding="utf-8")]
    audio = np.memmap(os.path.join(split_dir, "audio.bin"), dtype=np.int16, mode="r")
    for k, m in enumerate(meta):
        text = clean_text(m["text"])
        if text:
            yield np.array(audio[m["offset"]: m["offset"] + m["length"]]), {"text": text, "source": f"{source}:{k}"}


def write_split(out_dir, clips):
    os.makedirs(out_dir, exist_ok=True)
    offset = n = 0
    with open(os.path.join(out_dir, "audio.bin"), "wb") as fa, \
            open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as fm:
        for pcm, meta in clips:
            fa.write(pcm.tobytes())
            fm.write(json.dumps({**meta, "offset": offset, "length": len(pcm)}, ensure_ascii=False) + "\n")
            offset += len(pcm)
            n += 1
    return n, offset / SR / 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="D:/data/malay_50h_clean")
    ap.add_argument("--out", default="D:/data/malay_50h_mix")
    ap.add_argument("--cs_pilot", default="D:/cs_mining/cs_pilot/train")
    ap.add_argument("--fleurs_en", default="D:/data/fleurs_en/parquet-data/en_us/train-*.parquet")
    ap.add_argument("--en_hours", type=float, default=5.0)
    ap.add_argument("--cs_test_frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cs = list(parquet_clips(f"{HF}/datasets--emhaihsan--Synth-Manglish/snapshots/*/parquet/*.parquet", "text",
                            source="synth-manglish")) + list(memmap_clips(args.cs_pilot, "cs_pilot"))
    is_test = rng.random(len(cs)) < args.cs_test_frac
    en, en_samples = [], 0
    for pcm, meta in parquet_clips(args.fleurs_en, "transcription", lang="English", source="fleurs_en"):
        if en_samples >= args.en_hours * 3600 * SR:
            break
        en.append((pcm, meta))
        en_samples += len(pcm)

    base_meta = [json.loads(l) for l in open(os.path.join(args.base, "train", "meta.jsonl"), encoding="utf-8")]
    base_audio = np.memmap(os.path.join(args.base, "train", "audio.bin"), dtype=np.int16, mode="r")
    extra = [c for c, t in zip(cs, is_test) if not t] + en
    order = rng.permutation(len(base_meta) + len(extra))  # interleave so batches mix sources

    def train_clips():
        for i in order:
            if i < len(base_meta):
                m = base_meta[i]
                yield np.array(base_audio[m["offset"]: m["offset"] + m["length"]]), \
                    {k: v for k, v in m.items() if k not in ("offset", "length")}
            else:
                yield extra[i - len(base_meta)]

    n, h = write_split(os.path.join(args.out, "train"), train_clips())
    print(f"train: {n} clips, {h:.1f}h (base {len(base_meta)}, cs {len(cs) - is_test.sum()}, en {len(en)} "
          f"= {en_samples / SR / 3600:.1f}h)", flush=True)
    n, h = write_split(os.path.join(args.out, "cs_test"), (c for c, t in zip(cs, is_test) if t))
    print(f"cs_test: {n} clips, {h:.2f}h", flush=True)
    for split in ["valid", "test"]:
        shutil.copytree(os.path.join(args.base, split), os.path.join(args.out, split), dirs_exist_ok=True)


if __name__ == "__main__":
    main()
