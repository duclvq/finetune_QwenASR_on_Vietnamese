"""Take a random N-hour train subset of a prepared Malay build; valid/test are copied unchanged.

Usage: python subset_malay.py --src D:/data/malay_prepared --out D:/data/malay_50h --hours 50
"""
import argparse
import json
import os
import shutil

import numpy as np

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/data/malay_prepared")
    ap.add_argument("--out", default="D:/data/malay_50h")
    ap.add_argument("--hours", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = [json.loads(l) for l in open(f"{args.src}/train/meta.jsonl", encoding="utf-8")]
    pick, total = [], 0
    for i in np.random.default_rng(args.seed).permutation(len(meta)):
        if total >= args.hours * 3600 * SR:
            break
        pick.append(i)
        total += meta[i]["length"]
    pick.sort()

    audio = np.memmap(f"{args.src}/train/audio.bin", dtype=np.int16, mode="r")
    os.makedirs(f"{args.out}/train", exist_ok=True)
    offset = 0
    with open(f"{args.out}/train/audio.bin", "wb") as fa, \
            open(f"{args.out}/train/meta.jsonl", "w", encoding="utf-8") as fm:
        for i in pick:
            m = dict(meta[i])
            fa.write(audio[m["offset"]: m["offset"] + m["length"]].tobytes())
            m["offset"] = offset
            offset += m["length"]
            fm.write(json.dumps(m, ensure_ascii=False) + "\n")
    for split in ["valid", "test"]:
        shutil.copytree(f"{args.src}/{split}", f"{args.out}/{split}", dirs_exist_ok=True)
    print(f"{len(pick)} clips, {offset / SR / 3600:.2f}h -> {args.out}")


if __name__ == "__main__":
    main()
