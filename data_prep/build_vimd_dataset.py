"""Build the ViMD North + Central training data in one pass, with little disk use.

Combines download_vimd.py and prepare_data.py: each shard is downloaded,
filtered to --regions, resampled to 16kHz mono int16 and deleted, so only the
final OUT/<split>/audio.bin + meta.jsonl stay on disk (~7.5GB for North +
Central). Output format is the same as prepare_data.py, so train_whisper.py
reads it directly.

Usage: python build_vimd_dataset.py [--out data/prepared] [--regions North Central] [--workers 6]
"""
import argparse
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

from prepare_data import process_shard, SR

REPO = "nguyendv02/ViMD_Dataset"


def fetch_and_process(name, regions, max_sec, tmp_root):
    """Download one shard, keep wanted regions, return converted clips. Runs in a worker."""
    keep = pa.array(regions)
    fs = HfFileSystem()
    remote_regions = pq.read_table(fs.open(f"datasets/{REPO}/data/{name}"), columns=["region"]).column("region")
    if not pc.any(pc.is_in(remote_regions, value_set=keep)).as_py():
        return [], 0, False
    tmp = tempfile.mkdtemp(dir=tmp_root)
    try:
        local = hf_hub_download(REPO, f"data/{name}", repo_type="dataset", local_dir=tmp)
        table = pq.read_table(local)
        filtered = os.path.join(tmp, "filtered.parquet")
        pq.write_table(table.filter(pc.is_in(table.column("region"), value_set=keep)), filtered)
        del table
        rows, n = process_shard(filtered, max_sec)
        return rows, n, True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/prepared")
    ap.add_argument("--regions", nargs="+", default=["North", "Central"])
    ap.add_argument("--max_sec", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    files = sorted(p.rsplit("/", 1)[-1] for p in HfFileSystem().glob(f"datasets/{REPO}/data/*.parquet"))
    tmp_root = os.path.join(args.out, "_tmp")
    os.makedirs(tmp_root, exist_ok=True)

    for split in ["train", "valid", "test"]:
        out_dir = os.path.join(args.out, split)
        if os.path.exists(os.path.join(out_dir, "meta.jsonl")):
            print(f"{split}: already built, skipping", flush=True)
            continue
        os.makedirs(out_dir, exist_ok=True)
        shards = [f for f in files if f.startswith(split + "-")]
        offset, kept, total, downloaded = 0, 0, 0, 0
        with open(os.path.join(out_dir, "audio.bin"), "wb") as fa, \
                open(os.path.join(out_dir, "meta.jsonl.part"), "w", encoding="utf-8") as fm, \
                ProcessPoolExecutor(args.workers) as ex:
            jobs = ex.map(fetch_and_process, shards, [args.regions] * len(shards),
                          [args.max_sec] * len(shards), [tmp_root] * len(shards))
            for i, (name, (rows, n, used)) in enumerate(zip(shards, jobs), 1):
                total += n
                downloaded += used
                for pcm, meta in rows:
                    fa.write(pcm.tobytes())
                    meta.update(offset=offset, length=len(pcm))
                    fm.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    offset += len(pcm)
                    kept += 1
                status = f"kept {len(rows)}" if used else "no wanted regions, not downloaded"
                print(f"{split} [{i}/{len(shards)}] {name}: {status}", flush=True)
        os.replace(os.path.join(out_dir, "meta.jsonl.part"), os.path.join(out_dir, "meta.jsonl"))
        print(f"{split}: {kept}/{total} clips from {downloaded} shards, {offset / SR / 3600:.1f} hours", flush=True)
    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
