"""Silero VAD speech time per clip -> <data>/<split>/vad.jsonl ({"i": clip index, "speech_s": seconds}).

clean_malay_subset.py and the clean/suspect test split use it to spot truncated labels
(transcript chars per speech-second far below the corpus median).

Usage: python vad_stats.py D:/data/malay_50h test train
"""
import json
import sys

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

SR = 16000


def main():
    torch.set_num_threads(2)
    vad = load_silero_vad()
    root, splits = sys.argv[1], sys.argv[2:]
    for split in splits:
        d = f"{root}/{split}"
        meta = [json.loads(l) for l in open(f"{d}/meta.jsonl", encoding="utf-8")]
        audio = np.memmap(f"{d}/audio.bin", dtype=np.int16, mode="r")
        with open(f"{d}/vad.jsonl", "w") as f:
            for k, m in enumerate(meta):
                pcm = torch.from_numpy(audio[m["offset"]: m["offset"] + m["length"]].astype(np.float32) / 32768)
                sp = sum(t["end"] - t["start"] for t in get_speech_timestamps(pcm, vad, sampling_rate=SR)) / SR
                f.write(json.dumps({"i": k, "speech_s": round(sp, 2)}) + "\n")
                if k % 2000 == 0:
                    print(split, k, len(meta), flush=True)
        print(split, "done", flush=True)


if __name__ == "__main__":
    main()
