"""Build a label-cleaned Malay train split of the same size as the original subset.

The source transcripts are semi-supervised (Google STT) and some only cover part of the audio. Silero VAD measures
the speech time of each clip; clips whose transcript is far too short for that much speech
(chars per speech-second < --min_cps) or that have no detected speech are dropped, and replaced by clips from the
larger pool (--pool, e.g. the 500h build) that pass the same check, until --hours is reached again.
valid/test are copied unchanged (with their vad.jsonl) so results stay comparable.

Usage: python clean_malay_subset.py --src D:/data/malay_50h --pool D:/data/malay_prepared --out D:/data/malay_50h_clean
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

SR = 16000


def load(split_dir):
    meta = [json.loads(l) for l in open(os.path.join(split_dir, "meta.jsonl"), encoding="utf-8")]
    return meta, np.memmap(os.path.join(split_dir, "audio.bin"), dtype=np.int16, mode="r")


def speech_seconds(vad, pcm):
    ts = get_speech_timestamps(torch.from_numpy(pcm.astype(np.float32) / 32768), vad, sampling_rate=SR)
    return sum(t["end"] - t["start"] for t in ts) / SR


def keep(text, speech_s, min_cps, min_speech):
    return speech_s >= min_speech and len(text) / speech_s >= min_cps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/data/malay_50h")
    ap.add_argument("--pool", default="D:/data/malay_prepared")
    ap.add_argument("--out", default="D:/data/malay_50h_clean")
    ap.add_argument("--hours", type=float, default=50.0)
    ap.add_argument("--min_cps", type=float, default=6.0)
    ap.add_argument("--min_speech", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(4)
    vad = load_silero_vad()

    meta, audio = load(os.path.join(args.src, "train"))
    vad_rows = [json.loads(l)["speech_s"] for l in open(os.path.join(args.src, "train", "vad.jsonl"))]
    rows = [(m, audio, sp) for m, sp in zip(meta, vad_rows) if keep(m["text"], sp, args.min_cps, args.min_speech)]
    total = sum(m["length"] for m, _, _ in rows)
    print(f"kept {len(rows)}/{len(meta)} clips ({total / SR / 3600:.1f}h) from {args.src}", flush=True)

    used = {m["source"] for m in meta}
    pmeta, paudio = load(os.path.join(args.pool, "train"))
    added = checked = 0
    for i in np.random.default_rng(args.seed).permutation(len(pmeta)):
        if total >= args.hours * 3600 * SR:
            break
        m = pmeta[i]
        if m["source"] in used:
            continue
        checked += 1
        sp = speech_seconds(vad, paudio[m["offset"]: m["offset"] + m["length"]])
        if keep(m["text"], sp, args.min_cps, args.min_speech):
            rows.append((m, paudio, sp))
            total += m["length"]
            added += 1
    print(f"topped up {added} clips from pool ({checked} checked)", flush=True)

    os.makedirs(os.path.join(args.out, "train"), exist_ok=True)
    offset = 0
    with open(os.path.join(args.out, "train", "audio.bin"), "wb") as fa, \
            open(os.path.join(args.out, "train", "meta.jsonl"), "w", encoding="utf-8") as fm, \
            open(os.path.join(args.out, "train", "vad.jsonl"), "w") as fv:
        for k, (m, src_audio, sp) in enumerate(rows):
            fa.write(src_audio[m["offset"]: m["offset"] + m["length"]].tobytes())
            fm.write(json.dumps({**m, "offset": offset}, ensure_ascii=False) + "\n")
            fv.write(json.dumps({"i": k, "speech_s": round(sp, 2)}) + "\n")
            offset += m["length"]
    for split in ["valid", "test"]:
        shutil.copytree(os.path.join(args.src, split), os.path.join(args.out, split), dirs_exist_ok=True)
    print(f"train: {len(rows)} clips, {offset / SR / 3600:.1f}h -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
